# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision.transforms import v2

# Contrastive imports
from contrastive.datasets import WithAugmentationsDataset
from contrastive.encoder import ResNetEncoder
from contrastive.components import SoftNearestNeighbor

# Utils
import os
import random
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt; plt.switch_backend('agg')
from tqdm import tqdm
from datetime import datetime
from configurations import Configurations


def get_dataset(args) -> tuple[WithAugmentationsDataset]:
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
    train_dataset = WithAugmentationsDataset(
        dir=f'{args.datasets_path}/{args.dataset}',
        val_room=args.val_room,
        metric=args.metric,
        mask=args.mask,
        shift = args.shift,
        n_pos=args.n_pos,
        pos_thresh=args.pos_thresh,
        n_neg=args.n_neg,
        neg_thresh=args.neg_thresh,
        transforms=transforms,
        augmentations=augmentations,
        mode='train'
    )

    # Validation dataset
    val_dataset = WithAugmentationsDataset(
        dir=f'{args.datasets_path}/{args.dataset}',
        val_room=args.val_room,
        metric=args.metric,
        mask=args.mask,
        shift = args.shift,
        n_pos=args.n_pos,
        pos_thresh=args.pos_thresh,
        n_neg=args.n_neg,
        neg_thresh=args.neg_thresh,
        transforms=transforms,
        augmentations=augmentations,
        mode='val'
    )

    return train_dataset, val_dataset


def validate(
        args,
        model: nn.Module,
        dataset: WithAugmentationsDataset,
        loss_fn: SoftNearestNeighbor,
        figs_dir: str,
        epoch: int,
        n_bins: int=20
):
    data_loader = dataset.get_DataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
    device = next(model.parameters()).device

    val_loss = 0
    intra_embeddings = []   # embeddings for intra-consistency analysis
    inter_embeddings = []   # embeddings for inter-consistency analysis
    inter_set = []          # settings included in embeddings observations 
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(tqdm(data_loader, unit='imgs', total= len(data_loader), leave=True)):            
            
            # Move data to the target device
            if args.n_neg > 0:
                anchors, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores = data
                anchors = anchors.to(device)
                pos_ex, pos_sim_scores = pos_ex.to(device), pos_sim_scores.to(device)
                neg_ex, neg_sim_scores = neg_ex.to(device), neg_sim_scores.to(device)       
            else:
                anchors, pos_ex, pos_sim_scores, lidars, gds = data
                anchors = anchors.to(device)
                pos_ex, pos_sim_scores = pos_ex.to(device), pos_sim_scores.to(device)
                lidars, gds = lidars.to(device), gds.to(device)
            
            # Generate embeddings
            anc_embeddings = model(anchors)
            pos_embeddings = model(pos_ex)

            if args.n_neg > 0:
                # Generate embeddings of negative examples
                neg_embeddings = model(neg_ex)
                # Adaptive Contrastive Loss
                val_loss += loss_fn(anc_embeddings, pos_embeddings, pos_sim_scores, neg_batch=neg_embeddings, neg_sim_scores=neg_sim_scores).detach().item()
            else:
                # Adaptive Contrastive Loss
                val_loss += loss_fn(anc_embeddings, pos_embeddings, pos_sim_scores, lidars=lidars, gds=gds).detach().item()
            
            # For intra-consistency analysis just take anchor embeddings
            intra_embeddings.append(anc_embeddings.cpu()) 

            # For inter-scene consistency analysis take anchors and augmentations embeddings
            ancs = anc_embeddings.unsqueeze(1).cpu()
            augs = pos_embeddings[:, :-args.n_pos, ...].cpu() if args.n_pos > 0 else pos_embeddings.cpu()
            embeddings = torch.cat([ancs, augs], dim=1)
            inter_embeddings.append(embeddings)
            for idx in range(0, args.batch_size):
                record_idx = batch*args.batch_size + idx    
                if record_idx < dataset.annot_df.shape[0]:           
                    inter_set.append(dataset.annot_df.iloc[record_idx]['setting'])

            # Free space
            torch.cuda.empty_cache()
            
    val_loss /= len(data_loader)
    print(f'End of VALIDATION - Avg Soft Nearest Neighbor Loss: {val_loss:.5f}')

    # PLOTTING
    fig = plt.figure(figsize=[25,30])
    plt.axis('off')
    fig.suptitle(f'Validation epoch {epoch + 1}')

    ############################## INTRA-SCENE CONSISTENCY ###################################
    bins = [[] for _ in range(n_bins)]
    bin_tol = dataset.sim_scores_range / n_bins
    intra_embeddings = torch.cat(intra_embeddings, dim=0)
    corrs = []
    for i, score in enumerate(dataset.sim_scores_mat):
        anc_embedding = intra_embeddings[i, :]

        # Sort embeddings by descending sample similarity
        sorted_idx = np.argsort(score)[::-1].copy()
        sorted_scores = score[sorted_idx]
        sorted_embeddings = intra_embeddings[sorted_idx, :]
        
        # Measure the embedding similarity between the anchor and the sorted embeddings 
        embedding_sims = F.cosine_similarity(anc_embedding, sorted_embeddings)
        for j, sim in enumerate(embedding_sims):
            bin = int((1.0-score[j].item()) / bin_tol)
            if bin == len(bins): 
                bins[-1].append(sim.item())
            else:
                bins[bin].append(sim.item())

        # Measure the correlation between sample and embedding similarities
        corrs.append(np.corrcoef(x=sorted_scores, y=embedding_sims.numpy())[0, 1])

    # Intra-scene consistency plot
    ax = fig.add_subplot(3,1,1) 
    xticks = [f'{(1.0-(i-1)*bin_tol):.2f}-{(1.0-i*bin_tol):.2f}' for i in range(1, n_bins+1)]
    ax.set_title(f'Intra-scene Consistency (metric={dataset.metric})')
    ax.boxplot(bins, orientation='vertical')
    ax.set_xticks(range(1, n_bins+1), xticks)
    ax.set_ylabel('Embedding Similarity')
    ax.set_xlabel('Sample similarity')

    # Similarities correlation plot
    bx = fig.add_subplot(3,2,3)
    bx.set_title(f'Similarities Correlation')
    bx.boxplot(corrs, orientation='vertical')
    bx.set_ylabel('Pearson Coefficient')
    bx.set_xticks([])
    ##########################################################################################  

    ####################################### INTER-SCENE CONSISTENCY ############################################
    inter_embeddings = torch.cat(inter_embeddings, dim=0)
    augs_sim = F.cosine_similarity(inter_embeddings[:, 0, ...].unsqueeze(1), inter_embeddings[:, 1:, ...], dim=2)
    augs_sim = augs_sim.mean(dim=0)
    cx = fig.add_subplot(3,2,4)
    cx.set_title('Inter-scene Consistency')
    cx.set_ylabel('Embedding Similarity')
    cx.boxplot(augs_sim, orientation='vertical')
    cx.set_xticks([])
    #############################################################################################################

    ############################# Augmentations 2D t-SNE visualization ################################
    # TODO: create different visualizations per augmentation type (background, warehouse, etc.)
    dx = fig.add_subplot(3,2,5)
    dx.set_title('t-SNE Embedding Space (Augmentations)')
    dx.grid()

    aug_embeddings = []
    tsne_idx = []
    for aug_idx in range(inter_embeddings.shape[1]):
        aug = inter_embeddings[:, aug_idx, ...]
        aug_embeddings.append(aug)
        tsne_idx.extend([aug_idx for _ in range(aug.shape[0])])

    # Colormap
    aug_idx = np.unique(tsne_idx)
    cmap = plt.get_cmap('tab10')
    idx_to_color = {idx: cmap(idx) for idx in aug_idx}
    colors = [idx_to_color[i] for i in tsne_idx]

    print('Augmentations t-SNE visualization...')
    aug_embeddings = torch.cat(aug_embeddings, dim=0)
    aug_embeddings = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='random',
        random_state=42
    ).fit_transform(aug_embeddings)      
    dx.scatter(aug_embeddings[:,0], aug_embeddings[:,1], c=colors)
    dx.set_xticks([])
    dx.set_yticks([])
    
    # Create a legend manually
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Anchor' if i == 0 else f'Augmentation {i}',
                   markerfacecolor=idx_to_color[i], markersize=8)
        for i in aug_idx
    ]
    dx.legend(handles=handles)
    ###################################################################################################

   ################################ Episodes 2D t-SNE visualization ###################################
    print('Episodes t-SNE visualization...')
    rgb_embeddings = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='random',
        random_state=42
    ).fit_transform(intra_embeddings)

    ex = fig.add_subplot(3,2,6)
    ex.set_title(f"t-SNE Embedding Space (Settings)")
    ex.scatter(rgb_embeddings[:,0], rgb_embeddings[:,1], c=inter_set, cmap='tab10')
    ex.set_xticks([])
    ex.set_yticks([])
    ##################################################################################################

    # Save and close figure
    fig.savefig(f'{figs_dir}/epoch_{epoch + 1}.png', format='png')
    plt.close(fig)

    return val_loss


def train(
        args,
        model: nn.Module,
        train_dataset: WithAugmentationsDataset,
        val_dataset: WithAugmentationsDataset,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        exp_dir: str,
        figs_dir: str,
        lr_scheduler: optim.lr_scheduler.LRScheduler | str='manual',
):
    # Experiments seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Target device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)
    data_loader = train_dataset.get_DataLoader(batch_size=args.batch_size, num_workers=args.num_workers)

    # Logging
    log = open(f'{exp_dir}/log.txt', 'w')
    batch_update = len(data_loader) // 5

    train_loss_h, val_loss_h = [], []
    for epoch in range(args.epochs):

        header = f"{'*'*20}\nEPOCH {epoch} {f'(on {device}): {torch.cuda.get_device_name(device)}' if torch.cuda.is_available() else {f'on ({device})'}}"
        print(header)
        log.write(header + '\n')

        ### TRAINING ###
        model.train()
        running_loss = 0.0
        for batch, data in enumerate(tqdm(data_loader, unit='batch', leave=True)):

            # Move data to the target device
            if args.n_neg > 0:
                anchors, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores = data
                anchors = anchors.to(device)
                pos_ex, pos_sim_scores = pos_ex.to(device), pos_sim_scores.to(device)
                neg_ex, neg_sim_scores = neg_ex.to(device), neg_sim_scores.to(device)       
            else:
                anchors, pos_ex, pos_sim_scores, lidars, gds = data
                anchors = anchors.to(device)
                pos_ex, pos_sim_scores = pos_ex.to(device), pos_sim_scores.to(device)
                lidars, gds = lidars.to(device), gds.to(device)

            # Generate embeddings
            anc_embeddings = model(anchors)
            pos_embeddings = model(pos_ex)

            if args.n_neg > 0:
                # Generate embeddings of negative examples
                neg_embeddings = model(neg_ex)
                # Adaptive Contrastive Loss
                loss = loss_fn(anc_embeddings, pos_embeddings, pos_sim_scores, neg_batch=neg_embeddings, neg_sim_scores=neg_sim_scores)
            else:
                # Adaptive Contrastive Loss
                loss = loss_fn(anc_embeddings, pos_embeddings, pos_sim_scores, lidars=lidars, gds=gds)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.detach().item()

            if batch % batch_update == (batch_update - 1):
                update = f'Computed {(batch):4}/{len(data_loader)} batches - Avg Adaptive Contrastive Loss: {(running_loss/batch):.5f}'
                tqdm.write(update)
                log.write(update + '\n')

            # Free space
            torch.cuda.empty_cache()

        # Update the learning rate
        if lr_scheduler == 'manual':
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * (1 - (epoch / args.epochs))
        else:
            lr_scheduler.step()

        running_loss /= len(data_loader)
        foot = f"\nEnd of EPOCH {epoch} - Avg Loss: {running_loss:5f}"
        print(foot)
        log.write(foot + '\n')

        ### VALIDATION ###
        if (epoch + 1) % args.val_freq == 0:
            print('\nVALIDATION...')
            val_loss = validate(
                args=args,
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                figs_dir=figs_dir,
                epoch=epoch
            )
            train_loss_h.append(running_loss)
            val_loss_h.append(val_loss)

    # Free space
    model = model.cpu()
    torch.cuda.empty_cache()

    # Close the log file
    log.close()

    # Train/validation loss plot
    ticks = [str(i*args.val_freq) for i in range(1, len(val_loss_h) + 1)]
    fig = plt.figure(figsize=[15,10])
    fig.suptitle('SNN Loss History')
    ax = fig.gca()
    ax.plot(train_loss_h, label='Train')
    ax.plot(val_loss_h, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('SNN')
    ax.set_xticks([i for i in range(len(val_loss_h))], ticks)
    ax.legend()
    fig.savefig(f'{exp_dir}/loss_h.png', format='png')
    plt.close(fig)

    # Save the model
    torch.save(model.state_dict(), f'{exp_dir}/ResNetEncoder_state_dict.pt')


if __name__ == '__main__':
    print(f'{"-"*30}\nContrastive Scene Transfer Encoder training experiment!')

    # Configurations
    conf = Configurations()
    args = conf.get_args()

    # Set global seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create directories of the experiment
    now = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    exp_dir = f"./experiments/{'sampled' if args.n_neg > 0 else 'batch'}/{now}"  
    figs_dir = f'{exp_dir}/val_figs'
    os.makedirs(figs_dir)

    # Save the configuration
    conf.save_yaml(dir=exp_dir)

    # Datasets    
    train_dataset, val_dataset = get_dataset(args)

    # Model, Loss and Optimizer
    model = ResNetEncoder()
    loss_fn = SoftNearestNeighbor(
        args=args,
        tau_min=args.min_tau,
        tau_max=args.max_tau
    )
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    # Train the model
    train(
        args=args,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        exp_dir=exp_dir,
        figs_dir=figs_dir
    )