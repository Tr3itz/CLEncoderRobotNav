# Torch imports
import torch
import torch.optim as optim
from torchvision.transforms import v2

# Distributed Training
import torch.multiprocessing as mp
import torch.distributed as distr
from torch.distributed import init_process_group, destroy_process_group

# Contrastive imports
from trainer import SingleGPUTrainer, MultiGPUTrainer
from contrastive.datasets import RoomAllAgentsDataset
from contrastive.encoder import ResNetEncoder, MobileNetV3Encoder
MODEL = {
    'resnet50': ResNetEncoder,
    'mbnv3': MobileNetV3Encoder
}
from contrastive.components import SNNCosineSimilarityLoss, SNNSimCLR
LOSS_FN = {
    'scene-transfer': SNNCosineSimilarityLoss,
    'simclr': SNNSimCLR
}

# Utils
import os
import traceback
from datetime import datetime
from configurations import Configurations


def ddp_setup(rank: int, world_size: int):
    """
    Set up distributed process group.
    ----------
    Parameters:
    - rank: int       - process unique ID
    - world_size: int - number of total processes    
    """

    # Setup environment variables 
    os.environ['MASTER_ADDR'] = "localhost" # Master node IP    
    os.environ['MASTER_PORT'] = "21501"     # Master node port

    # Set up CUDA process group
    torch.cuda.set_device(rank)
    init_process_group(backend='gloo', rank=rank, world_size=world_size)

    print(f'[GPU:{rank}] DDP successfully set up!')


def get_dataset(args) -> tuple[RoomAllAgentsDataset]:
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize([232,232]),
        v2.CenterCrop([224,224]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Augmentations for positive examples
    augmentations = [
        # Brighter 
        v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize([232,232]),
            v2.CenterCrop([224,224]),
            v2.ColorJitter(brightness=(1.5,2.0)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),  
    ]
    
    # Training dataset
    train_dataset = RoomAllAgentsDataset(
        dir=f'{args.datasets_path}/{args.dataset}',
        algo=args.algo,
        val_room=args.val_room,
        metric=args.metric,
        mask=args.mask,
        shift = args.shift,
        n_pos=args.n_pos,
        pos_thresh=args.pos_thresh,
        n_neg=args.n_neg,
        neg_thresh=args.neg_thresh,
        batch_size=args.batch_size,
        micro_bsize=args.micro_bsize,
        transforms=transforms,
        augmentations=augmentations,
        mode='train',
        multi_gpu=args.multi_gpu
    )

    # Validation dataset
    val_dataset = RoomAllAgentsDataset(
        dir=f'{args.datasets_path}/{args.dataset}',
        algo=args.algo,
        val_room=args.val_room,
        metric=args.metric,
        mask=args.mask,
        shift = args.shift,
        n_pos=args.n_pos,
        pos_thresh=args.pos_thresh,
        n_neg=args.n_neg,
        neg_thresh=args.neg_thresh,
        batch_size=args.batch_size,
        micro_bsize=args.micro_bsize,
        transforms=transforms,
        augmentations=augmentations,
        mode='val',
        multi_gpu=args.multi_gpu
    )

    return train_dataset, val_dataset


def load_components(args):
    # Datasets    
    train_dataset, val_dataset = get_dataset(args)

    # Model, Loss and Optimizer
    model = MODEL[args.model]()
    loss_fn = LOSS_FN[args.algo](
        args=args,
        tau_min=args.min_tau,
        tau_max=args.max_tau
    )
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

    return model, train_dataset, val_dataset, loss_fn, optimizer


def main_single_gpu(args, exp_dir: str, figs_dir: str):
    # Load training objects
    model, train_ds, val_ds, loss_fn, optimizer = load_components(args)

    # Trainer
    trainer = SingleGPUTrainer(
        args=args,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        loss_fn=loss_fn,
        optimizer=optimizer,
        exp_dir=exp_dir,
        figs_dir=figs_dir
    )

    # Train the model
    trainer.train()


def main_multi_gpu(rank: int, world_size: int, exp_dir: str, figs_dir: str, args):
    try:
        # Set up distributed training
        ddp_setup(rank=rank, world_size=world_size)

        print(f"[GPU:{rank}] Initializing datasets and model...")
        # Load components
        model, train_ds, val_ds, loss_fn, optimizer = load_components(args)

        # Barrier synchronizazion
        distr.barrier(device_ids=[rank])

        if rank > 0:
            print(f"[GPU:{rank}] Retrieving shared similarity scores matrix...")
            val_ds.set_shared_sim_mat()
            if args.algo == 'scene-transfer':
                train_ds.set_shared_sim_mat()

        # Barrier synchronizazion
        distr.barrier(device_ids=[rank])

        print(f"[GPU:{rank}] Building trainer...")
        # Trainer
        trainer = MultiGPUTrainer(
            args=args,
            gpu_id=rank,
            world_size=world_size,
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            loss_fn=loss_fn,
            optimizer=optimizer,
            exp_dir=exp_dir,
            figs_dir=figs_dir
        )

        # Barrier synchronizazion
        distr.barrier(device_ids=[rank])

        # Train
        trainer.train()

        # Free shared memory
        val_ds._free_shm(rank=rank)
        if args.algo == 'scene-transfer':
            train_ds._free_shm(rank=rank)
        
        # End distributed training
        destroy_process_group()

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        traceback.print_exc()
        destroy_process_group()
    except KeyboardInterrupt:
        print(f"[CTRL+C] Destroy process group.")
        destroy_process_group()


if __name__ == '__main__':
    # Configurations
    conf = Configurations()
    args = conf.get_args()

    print(f'{"-"*30}\nContrastive Scene Transfer Encoder training experiment! (Framework: {args.algo})')

    # Create directories of the experiment
    now = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    exp_dir = f"./experiments/{args.algo}/{now}"  
    figs_dir = f'{exp_dir}/val_figs'
    os.makedirs(figs_dir, exist_ok=True)

    # Save the configuration
    conf.save_yaml(dir=exp_dir)

    if args.multi_gpu:
        # Available GPUs
        world_size = torch.cuda.device_count()
        # Adjust batch size to number of available GPUs
        args.batch_size //= world_size

        # Spawn mulitple processes (torch.multiprocessing takes care of assigning rank)
        if world_size > 1:
            mp.spawn(
                main_multi_gpu, 
                args=(world_size, exp_dir, figs_dir, args),
                nprocs=world_size
            )
        else:
            print(f'[WARN] Multi-GPU training requested, but only {world_size} found. Switching to single GPU training...')
            main_single_gpu(args, exp_dir, figs_dir)
    else:
        main_single_gpu(args, exp_dir, figs_dir)