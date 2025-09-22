# Torch imports
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

# Distributed Training
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP

# Contrastive imports
from contrastive.datasets import ContrastiveDataset
from contrastive.encoder import ResNetEncoder
from contrastive.components import SoftNearestNeighbor

# Utils
import os, gc, shutil
import numpy as np
from math import ceil
from abc import ABC, abstractmethod
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt; plt.switch_backend('agg')
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm
from contrastive import utils


class ContrastiveTrainer(ABC):
    def __init__(
            self, 
            args,
            model: ResNetEncoder,
            train_ds: ContrastiveDataset,
            val_ds: ContrastiveDataset,
            loss_fn: SoftNearestNeighbor,
            optimizer: optim.Optimizer,
            exp_dir: str
        ):
        super().__init__()

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Datasets
        self.train_ds = train_ds
        self.val_ds = val_ds

        # Directories for logging
        self.exp_dir = exp_dir
        self.figs_dir = f'{self.exp_dir}/val_figs'
        self.check_dir = f'{self.exp_dir}/checkpoints'
        os.makedirs(self.figs_dir, exist_ok=True)
        os.makedirs(self.check_dir, exist_ok=True)

        # Training settings
        self.epochs = args.epochs
        self.lr = args.learning_rate
        self.val_freq = args.val_freq

    @abstractmethod
    def train(): pass

    @abstractmethod
    def _train_step(): pass

    @abstractmethod
    def _val_epoch(): pass

    def _plot(self, intra_embeddings: torch.Tensor, inter_train: torch.Tensor, inter_ho: torch.Tensor, epoch: int):
        # Figure
        fig = plt.figure(figsize=[25,30])
        plt.axis('off')
        fig.suptitle(f'Validation epoch {epoch + 1}')

        # Plots
        self._intra_consistency(embeddings=intra_embeddings, fig=fig)
        self._inter_consistency(train_embeds=inter_train, ho_embeds=inter_ho, fig=fig)

        # Save and close figure
        fig.savefig(f'{self.figs_dir}/epoch_{epoch + 1}.png', format='png')
        plt.close(fig)

    def _intra_consistency(self, embeddings: torch.Tensor, fig: Figure, n_bins: int=10):
        print('Intra-scene consistency visualization...')

        # Inter-scene consistency
        bins = [[] for _ in range(n_bins)]
        bin_tol = self.val_ds.sim_scores_range / n_bins
        corrs = []
        for i, score in enumerate(self.val_ds.sim_scores_mat):
            anc_embedding = embeddings[i, :]

            # Sort embeddings by descending sample similarity
            sorted_idx = np.argsort(score)[::-1].copy()
            sorted_scores = score[sorted_idx]
            sorted_embeddings = embeddings[sorted_idx, :]
            
            # Measure the embedding similarity between the anchor and the sorted embeddings 
            embedding_sims = F.cosine_similarity(anc_embedding, sorted_embeddings)
            for j, sim in enumerate(embedding_sims):
                bin = int((1.0-sorted_scores[j].item()) / bin_tol)
                if bin == len(bins): 
                    bins[-1].append(sim.item())
                else:
                    bins[bin].append(sim.item())

            # Measure the correlation between sample and embedding similarities
            corrs.append(np.corrcoef(x=sorted_scores, y=embedding_sims.numpy())[0, 1])

        # Intra-scene consistency box-plots
        ax = fig.add_subplot(3,1,1) 
        xticks = [f'{(1.0-(i-1)*bin_tol):.2f}-{(1.0-i*bin_tol):.2f}' for i in range(1, n_bins+1)]
        ax.set_title(f'Intra-scene Consistency')
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

    def _inter_consistency(self, train_embeds: torch.Tensor, ho_embeds: torch.Tensor, fig: Figure):
        print('Inter-scene consistency visualization...')

        # Augmentation embeddings similarities
        train_sim = F.cosine_similarity(train_embeds[:, 0, ...].unsqueeze(1), train_embeds[:, 1:, ...], dim=2)
        ho_sim = F.cosine_similarity(ho_embeds[:, 0, ...].unsqueeze(1), ho_embeds[:, 1:, ...], dim=2)
        train_sim = train_sim.mean(dim=0)
        ho_sim = ho_sim.mean(dim=0)
        
        # Augmentation embeddings similarity box-plots
        cx = fig.add_subplot(3,2,4)
        cx.set_title('Inter-scene Consistency')
        cx.set_ylabel('Embedding Similarity')
        cx.boxplot([train_sim, ho_sim], tick_labels=['Training', 'Hold-out'], orientation='vertical')

        # Train scenes t-SNE
        dx = fig.add_subplot(3,2,5)
        dx.set_title('t-SNE Embedding Space (Training Scenes)')
        dx.grid()
        self._inter_tsne(embeddings=train_embeds, ax=dx)

        # Hold-out scenes t-SNE
        ex = fig.add_subplot(3,2,6)
        ex.set_title('t-SNE Embedding Space (Hold-out Scenes)')
        ex.grid()
        self._inter_tsne(embeddings=ho_embeds, ax=ex, scenes='val')

    def _inter_tsne(self, embeddings: torch.Tensor, ax: Axes, scenes:str ='train'):
        N_SAMPLES, N_AUGS, DIM = embeddings.shape

        # Prepare embeddings for t-SNE
        flattened_embeds = embeddings.reshape(-1, DIM).numpy()

        # Perform t-SNE
        print('Augmentations t-SNE visualization...')
        tsne_embeds = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate='auto',
            init='random',
            random_state=42
        ).fit_transform(flattened_embeds)

        # t-SNE plot
        tsne_idx = np.repeat(np.arange(N_AUGS), N_SAMPLES)
        cmap = plt.get_cmap('Set1', N_AUGS)      
        ax.scatter(tsne_embeds[:,0], tsne_embeds[:,1], c=tsne_idx, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])

        # Create a legend manually
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=self.val_ds.SCENE_MAP[scenes][i],
                       markerfacecolor=cmap(i), markersize=8)
            for i in range(N_AUGS)
        ]
        ax.legend(handles=handles, title='Scenes')


class SingleGPUTrainer(ContrastiveTrainer):
    def __init__(
            self, 
            args,
            model: ResNetEncoder,
            train_ds: ContrastiveDataset,
            val_ds: ContrastiveDataset,
            loss_fn: SoftNearestNeighbor,
            optimizer: optim.Optimizer,
            exp_dir: str
        ):
        super().__init__(
            args,
            model,
            train_ds,
            val_ds,
            loss_fn,
            optimizer,
            exp_dir
        )

        # Retrieve DataLoaders
        self.train_dataloader = train_ds.get_DataLoader(num_workers=args.num_workers)
        self.val_dataloader = val_ds.get_DataLoader(num_workers=args.num_workers)

        # Move model to the target device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler(device=self.device)
        self.model = self.model.to(self.device)

    def train(self):
        # Environment variable setting
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Logging
        log = open(f'{self.exp_dir}/log.txt', 'w')
        batch_update = len(self.train_dataloader) // 5

        # Pipeline
        train_loss_h, val_loss_h = [], []
        min_val_loss, best_epoch = np.inf, 0
        for epoch in range(self.epochs):

            # Epoch header
            header = f"{'*'*20}\nEPOCH {epoch} {f'(on {self.device}): {torch.cuda.get_device_name(self.device)}'}"
            print(header)
            log.write(header + '\n')

            self.model.train()
            running_loss = 0.0
            for batch, data in enumerate(tqdm(self.train_dataloader, unit='batch', leave=True)):
                # TRAIN step
                running_loss += self._train_step(data)

                if batch % batch_update == (batch_update - 1):
                    update = f'Computed {(batch):4}/{len(self.train_dataloader)} batches - Avg Adaptive Contrastive Loss: {(running_loss/batch):.5f}'
                    tqdm.write(update)
                    log.write(update + '\n')

                # Free space
                torch.cuda.empty_cache()    

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr * (1 - (epoch / self.epochs))

            # Epoch foot
            running_loss /= len(self.train_dataloader)
            foot = f"\nEnd of EPOCH {epoch} - Avg Loss: {running_loss:5f}"
            print(foot)
            log.write(foot + '\n')

            ### VALIDATION ###
            if (epoch + 1) % self.val_freq == 0:
                print('\nVALIDATION...')
                val_loss = self._val_epoch(epoch)
                train_loss_h.append(running_loss)
                val_loss_h.append(val_loss)
                
                # Checkpoint if validation loss decreased
                if val_loss < min_val_loss:
                    torch.save(self.model.state_dict(), f'{self.check_dir}/{self.model.__class__.__name__}_epoch{epoch}.pt')
                    min_val_loss = val_loss
                    best_epoch = epoch

        # Close the log file
        log.close()

        # Train/validation loss plot
        ticks = [str(i*self.val_freq) for i in range(1, len(val_loss_h) + 1)]
        places = [i for i in range(len(val_loss_h))]
        step = max(1, len(val_loss_h)//10)
        fig = plt.figure(figsize=[15,10])
        fig.suptitle('SNN Loss History')
        ax = fig.gca()
        ax.plot(train_loss_h, label='Train')
        ax.plot(val_loss_h, label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('SNN')
        ax.set_xticks(places[::step], ticks[::step])
        ax.legend()
        fig.savefig(f'{self.exp_dir}/loss_h.png', format='png')
        plt.close(fig)

        # Save best model
        # torch.save(self.model.state_dict(), f'{self.exp_dir}/{self.model.__class__.__name__}_state_dict.pt')
        shutil.copy2(
            f'{self.check_dir}/{self.model.__class__.__name__}_epoch{best_epoch}.pt',
            f'{self.exp_dir}/{self.model.__class__.__name__}_state_dict.pt'
        )

    def _forward_and_clear(self, x: torch.Tensor):
        # Move tensor to the device
        x = x.to(self.device)

        # Perform forwarding and clear space
        out = self.model(x)
        del x

        # Return embedding on the GPU
        return out

    def _train_step(self, data):
        # Clear gradients
        self.optimizer.zero_grad()

        # Unpack data
        if self.train_ds.algo == 'scene-transfer':
            anchors, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores = data       
        else:
            anchors, pos_ex, info = data
            if self.train_ds.__class__.__name__ == 'RoomAllAgentsDataset':
                lidars, gds, angles = info
            else:
                positions, velocities, quaternions = info

        # Calculate gradient accumulation steps
        B = anchors.shape[0]
        accumulation_steps = ceil(B / self.train_ds.micro_bsize)
        
        running_loss = 0.0
        for i in range(accumulation_steps):
            start = i * self.train_ds.micro_bsize
            end = min(B, start + self.train_ds.micro_bsize)

            with autocast(device_type='cuda'):       
                # Generate embeddings of anchors
                anc_embeddings = self._forward_and_clear(x=anchors[start:end, ...])
                # Generate embeddings of postive examples
                pos_embeddings = self._forward_and_clear(x=pos_ex[start:end, ...])

                if self.train_ds.algo == 'scene-transfer':
                    # Generate embeddings of negative examples
                    neg_embeddings = self._forward_and_clear(x=neg_ex[start:end, ...])

                    # Move scores on the GPU
                    b_pos_sim_scores = pos_sim_scores[start:end, ...].to(self.device)
                    b_neg_sim_scores = neg_sim_scores[start:end, ...].to(self.device)

                    # Adaptive Contrastive Loss
                    loss = self.loss_fn(anc_embeddings, pos_embeddings, b_pos_sim_scores, neg_batch=neg_embeddings, neg_sim_scores=b_neg_sim_scores)

                    # Move back on the CPU and free space
                    del anc_embeddings, pos_embeddings, b_pos_sim_scores, \
                        neg_embeddings, b_neg_sim_scores
                else:
                    if self.train_ds.__class__.__name__ == 'RoomAllAgentsDataset':
                        # Move info on the GPU
                        b_lidars, b_gds, b_angles = lidars[start:end, ...].to(self.device), gds[start:end, ...].to(self.device), angles[start:end, ...].to(self.device)

                        # Adaptive Contrastive Loss
                        # loss = self.loss_fn(anc_embeddings, pos_embeddings, lidars=b_lidars, gds=b_gds, angles=b_angles)
                        loss = self.loss_fn(
                            anc_embeddings, pos_embeddings, 
                            utils.robot_nav_scores,
                            self.train_ds.mask, self.train_ds.shift,
                            lidars=b_lidars,
                            gds=b_gds,
                            angles=b_angles,
                            metric=self.train_ds.metric
                        )

                        # Move back on the CPU and free space
                        del anc_embeddings, pos_embeddings, b_lidars, b_gds, b_angles
                    else:
                        # Move info on the GPU
                        b_positions, b_velocities, b_quaternions = positions[start:end, ...].to(self.device), velocities[start:end, ...].to(self.device), quaternions[start:end, ...].to(self.device)

                        # Adaptive Contrastive Loss
                        loss = self.loss_fn(
                            anc_embeddings, pos_embeddings, 
                            utils.airsim_scores,
                            0.25, 0.75, 0.6, 0.4,  # Wp, Wv, Wpos, Wrot
                            positions=b_positions,
                            velocities=b_velocities,
                            quaternions=b_quaternions
                        )

                        # Move back on the CPU and free space
                        del anc_embeddings, pos_embeddings, b_positions, b_velocities, b_quaternions

                # Backpropagation
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                running_loss += loss.detach()
        
        # Optimizer/scaler update
        self.scaler.unscale_(self.optimizer) # Uscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return running_loss.item()

    def _val_epoch(self, epoch):
        running_loss = 0.0
        intra_embeddings = [] # embeddings for intra-consistency analysis
        inter_train = []      # embeddings for inter-consistency analysis (training augmentations)
        inter_ho = []         # embeddings for inter-consistency analysis (hold-out augmentations)
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.val_dataloader, unit='imgs', total= len(self.val_dataloader), leave=True)):       
                
                # Unpack data
                if self.val_ds.algo == 'scene-transfer':
                    anchors, scenes, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores = data
                else:
                    anchors, scenes, pos_ex, info = data
                    if self.val_ds.__class__.__name__ == 'RoomAllAgentsDataset':
                        lidars, gds, angles = info
                    else:
                        positions, velocities, quaternions = info       

                # Calculate accumulation steps
                B = anchors.shape[0]
                accumulation_steps = ceil(B / self.val_ds.micro_bsize)

                for i in range(accumulation_steps):
                    start = i * self.val_ds.micro_bsize
                    end = min(B, start + self.val_ds.micro_bsize)
                    
                    with autocast(device_type='cuda'):
                        # Generate embeddings of anchors
                        anc_embeddings = self._forward_and_clear(x=anchors[start:end, ...])
                        # Generate embeddings of postive examples
                        pos_embeddings = self._forward_and_clear(x=pos_ex[start:end, ...])
                        # Generate embeddings of scenes
                        scene_embeddings = self._forward_and_clear(x=scenes[start:end, ...])

                        if self.val_ds.algo == 'scene-transfer':
                            # Generate embeddings of negative examples
                            neg_embeddings = self._forward_and_clear(x=neg_ex[start:end, ...])

                            # Move scores on the GPU
                            b_pos_sim_scores = pos_sim_scores[start:end, ...].to(self.device)
                            b_neg_sim_scores = neg_sim_scores[start:end, ...].to(self.device)

                            # Adaptive Contrastive Loss
                            loss = self.loss_fn(anc_embeddings, pos_embeddings, b_pos_sim_scores, neg_batch=neg_embeddings, neg_sim_scores=b_neg_sim_scores).detach().item()
                            running_loss += loss / accumulation_steps

                            # Free space
                            del anc_embeddings, pos_embeddings, b_pos_sim_scores, \
                                neg_embeddings, b_neg_sim_scores
                        else:
                            if self.train_ds.__class__.__name__ == 'RoomAllAgentsDataset':
                                # Move info on the GPU
                                b_lidars, b_gds, b_angles = lidars[start:end, ...].to(self.device), gds[start:end, ...].to(self.device), angles[start:end, ...].to(self.device)

                                # Adaptive Contrastive Loss
                                # loss = self.loss_fn(anc_embeddings, pos_embeddings, lidars=b_lidars, gds=b_gds, angles=b_angles)
                                loss = self.loss_fn(
                                    anc_embeddings, pos_embeddings, 
                                    utils.robot_nav_scores,
                                    self.train_ds.mask, self.train_ds.shift,
                                    lidars=b_lidars,
                                    gds=b_gds,
                                    angles=b_angles,
                                    metric=self.train_ds.metric
                                ).detach().item()
                                running_loss += loss / accumulation_steps

                                # Move back on the CPU and free space
                                del anc_embeddings, pos_embeddings, b_lidars, b_gds, b_angles
                            else:
                                 # Move info on the GPU
                                b_positions, b_velocities, b_quaternions = positions[start:end, ...].to(self.device), velocities[start:end, ...].to(self.device), quaternions[start:end, ...].to(self.device)

                                # Adaptive Contrastive Loss
                                loss = self.loss_fn(
                                    anc_embeddings, pos_embeddings, 
                                    utils.airsim_scores,
                                    0.25, 0.75, 0.6, 0.4,  # Wp, Wv, Wpos, Wrot
                                    positions=b_positions,
                                    velocities=b_velocities,
                                    quaternions=b_quaternions
                                )

                                # Move back on the CPU and free space
                                del anc_embeddings, pos_embeddings, b_positions, b_velocities, b_quaternions             
                    
                    # Move back to the CPU
                    scene_embeddings = scene_embeddings.cpu()

                    # Dissect scenes from embeddings
                    if self.val_ds.__class__.__name__ == 'RoomAllAgentsDataset':
                        anc_scene = scene_embeddings[:, 5, ...] # Office scene as anchor
                        train_scenes = scene_embeddings[:, :len(self.val_ds.SCENES['train']), ...]
                        val_scenes = scene_embeddings[:, len(self.val_ds.SCENES['train']):, ...]
                    else:
                        anc_scene = scene_embeddings[:, 1, ...] # Digital as anchor
                        train_scenes = scene_embeddings[:, 1:len(self.val_ds.SCENES['train'])+1, ...]
                        val_scenes = scene_embeddings[:, len(self.val_ds.SCENES['train'])+1:, ...]

                    # For intra-scene consistency take the 1st scene (office)
                    intra_embeddings.append(anc_scene.squeeze(dim=1))
                        
                    # For inter-scene consistency analysis take all scenes
                    inter_train.append(train_scenes)
                    inter_ho.append(val_scenes)

                    del scene_embeddings, anc_scene, train_scenes, val_scenes

                # Free space
                torch.cuda.empty_cache()
                
        running_loss /= len(self.val_dataloader)
        print(f'End of VALIDATION - Avg Soft Nearest Neighbor Loss: {running_loss:.5f}')

        # Prepare tensors for plotting
        intra_embeddings = torch.cat(intra_embeddings, dim=0)
        inter_train = torch.cat(inter_train, dim=0)
        inter_ho = torch.cat(inter_ho, dim=0)

        # Plotting
        self._plot(intra_embeddings, inter_train, inter_ho, epoch)

        return running_loss   


class MultiGPUTrainer(ContrastiveTrainer):
    def __init__(
            self, 
            args,
            gpu_id: int,
            world_size: int,
            model: ResNetEncoder,
            train_ds: ContrastiveDataset,
            val_ds: ContrastiveDataset,
            loss_fn: SoftNearestNeighbor,
            optimizer: optim.Optimizer,
            exp_dir: str
        ):
        super().__init__(
            args,
            model,
            train_ds,
            val_ds,
            loss_fn,
            optimizer,
            exp_dir
        )
    
        # Node info
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.scaler = GradScaler(device=self.device)
        self.is_master = gpu_id == 0

        # Adapt the model to Multi-GPU training
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # Avoid divergence

        # Move model to the target device  
        self.model = model.to(self.device)
        print(f"[GPU:{self.gpu_id}] Moved the model to target deivce: {torch.cuda.get_device_name(self.gpu_id)}!")
        self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id)
        print(f"[GPU:{self.gpu_id}] Wrapped model with DDP!")

        # Retrieve distributed DataLoaders
        self.train_dataloader = train_ds.get_DataLoader(num_workers=args.num_workers)
        self.val_dataloader = val_ds.get_DataLoader(num_workers=args.num_workers)
        print(f"[GPU:{self.gpu_id}] Retrieved distributed DataLoader!")

    def train(self):
        # Logging
        if self.is_master:
            train_loss_h, val_loss_h = [], []

        # Pipeline
        batch_update = len(self.train_dataloader) // 5
        for epoch in range(self.epochs):
            print(f"[GPU:{self.gpu_id}] Started EPOCH {epoch} on {torch.cuda.get_device_name(self.gpu_id)}")

            self.model.train()
            running_loss = torch.tensor(0.0)
            self.train_dataloader.sampler.set_epoch(epoch)
            for batch, data in enumerate(self.train_dataloader):
                # TRAIN step
                running_loss += self._train_step(data)
                if batch % batch_update == (batch_update - 1):
                    print(f'[GPU:{self.gpu_id}, EPOCH: {epoch}] Computed {(batch+1):4}/{len(self.train_dataloader)} batches - Avg Adaptive Contrastive Loss: {(running_loss/(batch+1)):.5f}')
                
                # Free space
                gc.collect()                
                torch.cuda.empty_cache()                    

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr * (1 - (epoch / self.epochs))

            # Global average loss
            distr.all_reduce(running_loss, op=distr.ReduceOp.SUM)
            avg_train_loss = running_loss / len(self.train_dataloader)
            if self.is_master:
                print(f"\nEnd of EPOCH {epoch} - Avg Loss: {avg_train_loss:5f}\n")

            ### VALIDATION ###
            if (epoch + 1) % self.val_freq == 0:
                print(f'[GPU:{self.gpu_id}, EPOCH: {epoch}] VALIDATION...')
                avg_val_loss = self._val_epoch(epoch)
                if self.is_master:
                    train_loss_h.append(avg_train_loss)
                    val_loss_h.append(avg_val_loss)

        if self.is_master:
            # Train/validation loss plot
            ticks = [str(i*self.val_freq) for i in range(1, len(val_loss_h) + 1)]
            fig = plt.figure(figsize=[15,10])
            fig.suptitle('SNN Loss History')
            ax = fig.gca()
            ax.plot(train_loss_h, label='Train')
            ax.plot(val_loss_h, label='Validation')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('SNN')
            ax.set_xticks([i for i in range(len(val_loss_h))], ticks)
            ax.legend()
            fig.savefig(f'{self.exp_dir}/loss_h.png', format='png')
            plt.close(fig)

            # Save the model
            torch.save(self.model.module.state_dict(), f'{self.exp_dir}/ResNetEncoder_state_dict.pt')

    def _forward_and_clear(self, x: torch.Tensor):
        # Move tensor to the device
        x = x.to(self.device)

        # Perform forwarding and clear space
        out = self.model(x)
        del x

        # Return embedding on the GPU
        return out

    def _train_step(self, data):
        # Clear gradients
        self.optimizer.zero_grad()

        # Unpack data
        if self.train_ds.algo == 'scene-transfer':
            anchors, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores = data       
        else:
            anchors, pos_ex, lidars, gds, angles = data

        # Calculate gradient accumulation steps
        B = anchors.shape[0]
        accumulation_steps = ceil(B / self.train_ds.micro_bsize)
        
        running_loss = 0.0
        for i in range(accumulation_steps):
            start = i * self.train_ds.micro_bsize
            end = min(B, start + self.train_ds.micro_bsize)

            with autocast(device_type='cuda'):       
                # Generate embeddings of anchors
                anc_embeddings = self._forward_and_clear(x=anchors[start:end, ...])
                # Generate embeddings of postive examples
                pos_embeddings = self._forward_and_clear(x=pos_ex[start:end, ...])

                if self.train_ds.algo == 'scene-transfer':
                    # Generate embeddings of negative examples
                    neg_embeddings = self._forward_and_clear(x=neg_ex[start:end, ...])

                    # Move scores on the GPU
                    b_pos_sim_scores = pos_sim_scores[start:end, ...].to(self.device)
                    b_neg_sim_scores = neg_sim_scores[start:end, ...].to(self.device)

                    # Adaptive Contrastive Loss
                    loss = self.loss_fn(anc_embeddings, pos_embeddings, b_pos_sim_scores, neg_batch=neg_embeddings, neg_sim_scores=b_neg_sim_scores)

                    # Move back on the CPU and free space
                    del anc_embeddings, pos_embeddings, b_pos_sim_scores, \
                        neg_embeddings, b_neg_sim_scores
                else:
                    # Move info on the GPU
                    b_lidars, b_gds, b_angles = lidars[start:end, ...].to(self.device), gds[start:end, ...].to(self.device), angles[start:end, ...].to(self.device)

                    # Adaptive Contrastive Loss
                    loss = self.loss_fn(anc_embeddings, pos_embeddings, lidars=b_lidars, gds=b_gds, angles=b_angles)

                    # Move back on the CPU and free space
                    del anc_embeddings, pos_embeddings, b_lidars, b_gds, b_angles

                # Backpropagation
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                running_loss += loss.detach().cpu()
        
        # Optimizer/scaler update
        self.scaler.unscale_(self.optimizer) # Uscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return running_loss

    def _val_epoch(self, epoch):
        running_loss = torch.tensor(0.0)
        intra_embeddings = [] # embeddings for intra-consistency analysis
        inter_train = []      # embeddings for inter-consistency analysis (training augmentations)
        inter_ho = []         # embeddings for inter-consistency analysis (hold-out augmentations)
        self.model.eval()
        with torch.no_grad():
            self.val_dataloader.sampler.set_epoch(epoch)
            for batch, data in enumerate(self.val_dataloader):             
                
                # Unpack data
                if self.train_ds.algo == 'scene-transfer':
                    anchors, scenes, pos_ex, pos_sim_scores, neg_ex, neg_sim_scores = data
                elif self.train_ds.algo == 'simclr':
                    anchors, scenes, pos_ex, lidars, gds, angles = data        
                else:
                    anchors, pos_ex, pos_sim_scores, lidars, gds, angles = data

                # Calculate accumulation steps
                B = anchors.shape[0]
                accumulation_steps = ceil(B / self.val_ds.micro_bsize)

                for i in range(accumulation_steps):
                    start = i * self.val_ds.micro_bsize
                    end = min(B, start + self.val_ds.micro_bsize)
                    
                    with autocast(device_type='cuda'):
                        # Generate embeddings of anchors
                        anc_embeddings = self._forward_and_clear(x=anchors[start:end, ...])
                        # Generate embeddings of postive examples
                        pos_embeddings = self._forward_and_clear(x=pos_ex[start:end, ...])
                        # Generate embeddings of scenes
                        scene_embeddings = self._forward_and_clear(x=scenes[start:end, ...])

                        if self.val_ds.algo == 'scene-transfer':
                            # Generate embeddings of negative examples
                            neg_embeddings = self._forward_and_clear(x=neg_ex[start:end, ...])

                            # Move scores on the GPU
                            b_pos_sim_scores = pos_sim_scores[start:end, ...].to(self.device)
                            b_neg_sim_scores = neg_sim_scores[start:end, ...].to(self.device)

                            # Adaptive Contrastive Loss
                            loss = self.loss_fn(anc_embeddings, pos_embeddings, b_pos_sim_scores, neg_batch=neg_embeddings, neg_sim_scores=b_neg_sim_scores).detach().cpu()
                            running_loss += loss / accumulation_steps

                            # Free space
                            del anc_embeddings, pos_embeddings, b_pos_sim_scores, \
                                neg_embeddings, b_neg_sim_scores
                        else:
                            # Move info on the GPU
                            b_lidars, b_gds, b_angles = lidars[start:end, ...].to(self.device), gds[start:end, ...].to(self.device), angles[start:end, ...].to(self.device)

                            # Adaptive Contrastive Loss (consider only hold-out augmentations)                            
                            loss = self.loss_fn(anc_embeddings, pos_embeddings, lidars=b_lidars, gds=b_gds, angles=b_angles).detach().cpu()
                            running_loss += loss / accumulation_steps

                            # Free space
                            del anc_embeddings, pos_embeddings, b_lidars, b_gds, b_angles                
                    
                    # Move back to the CPU
                    scene_embeddings = scene_embeddings.cpu()

                    # Dissect scenes from embeddings
                    anc_scene = scene_embeddings[:, 5, ...] # Office as anchor scene
                    train_scenes = scene_embeddings[:, len(self.val_ds.SCENES['train']), ...]
                    val_scenes = scene_embeddings[:, len(self.val_ds.SCENES['train']):, ...]

                    # For intra-scene consistency take the 1st scene (office)
                    intra_embeddings.append(anc_scene.squeeze(dim=1))
                        
                    # For inter-scene consistency analysis take all scenes
                    inter_train.append(torch.cat([anc_scene, train_scenes], dim=1))
                    inter_ho.append(torch.cat([anc_scene, val_scenes], dim=1))

                    del scene_embeddings, anc_scene, train_scenes, val_scenes

                pct = round(batch / len(self.val_dataloader), 2) * 100
                if pct % 10 == 0:
                    print(f'[GPU:{self.gpu_id}, EPOCH: {epoch}] Validation progress {int(pct):.2f}%')

                # Free space
                gc.collect()
                torch.cuda.empty_cache()

        # Global average loss
        distr.all_reduce(running_loss, op=distr.ReduceOp.SUM)
        avg_val_loss = running_loss / len(self.val_dataloader)

        if self.is_master:
            print(f'\n[EPOCH: {epoch}] End of VALIDATION - Avg Soft Nearest Neighbor Loss: {avg_val_loss:.5f}\n')

        # Prepare tensors for gathering
        intra_embeddings = torch.cat(intra_embeddings, dim=0)
        inter_train = torch.cat(inter_train, dim=0)
        inter_ho = torch.cat(inter_ho, dim=0)

        # Gather tensors accross GPUs
        if self.is_master:
            full_intra = [torch.empty_like(intra_embeddings) for _ in range(self.world_size)]
            full_inter_train = [torch.empty_like(inter_train) for _ in range(self.world_size)]
            full_inter_ho = [torch.empty_like(inter_ho) for _ in range(self.world_size)]
        else:
            full_intra = full_inter_train = full_inter_ho = None

        distr.gather(intra_embeddings, full_intra, dst=0)
        distr.gather(inter_train, full_inter_train, dst=0)
        distr.gather(inter_ho, full_inter_ho, dst=0)

        # Plotting
        if self.is_master:
            print(f'[GPU:{self.gpu_id}, EPOCH {epoch}] Master plotting...')
            full_intra = torch.cat(full_intra, dim=0)
            full_inter_train = torch.cat(full_inter_train, dim=0)
            full_inter_ho = torch.cat(full_inter_ho, dim=0)
            self._plot(full_intra, full_inter_train, full_inter_ho, epoch)

        # Barrier synchronization
        distr.barrier(device_ids=[self.gpu_id])

        return avg_val_loss
        