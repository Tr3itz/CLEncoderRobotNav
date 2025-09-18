import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class SoftNearestNeighbor(nn.Module, ABC):
    def __init__(self, args, tau_min: float=0.1, tau_max: float=1.0):
        """
        Adaptive Soft Nearest Neighbor objective base class.
        ----------
        Parameters:
        - metric: str        - metric to use for computing similarity between examples (lidar, goal, both)
        - tau_min: float     - adaptive temperature minimum value
        - tau_max: float     - adaptive temperature maximum value
        """
        super().__init__()

        # Metric
        self.metric = args.metric
        if self.metric in ['lidar', 'both']:
            self.mask = args.mask
            self.shift = args.shift

        # Temperatures range
        self.tau_min, self.tau_max = tau_min, tau_max

    @abstractmethod
    def __call__(self): pass
    
    def _lidar_dists(self, lidars: torch.Tensor):
        """
        Compute in-batch lidar distances.
        """

        if not hasattr(self, 'mask_w') and hasattr(self, 'mask'):
            # Define LiDAR readings mask
            w = torch.zeros(size=(lidars.shape[1],))
            match self.mask:
                case 'naive':
                    w += 1
                case 'binary':
                    # In FOV readings
                    w[64:164] += 1
                case 'soft':
                    # In FOV readings
                    w[64:164] += 1
                    # Out of FOV readings
                    x = torch.linspace(0.0, 1.0, w[164:].shape[0])
                    s_right = 1 - 0.9*(1 / (1+torch.exp(-x + self.shift))) # Sigmoid 1.0 -> 0.1
                    s_left = 0.1 + 0.9*(1 / (1+torch.exp(-x + self.shift))) # Sigmoid 0.1 -> 1.0
                    w[164:] += s_right
                    w[:64] += s_left
                
            # Mask and Normalizer    
            self.mask_w = w.to(lidars.get_device())
            self.norm = torch.sqrt(w.sum())

        # Weighted eucledian distances between in-batch anchors lidar readings
        lid_dists = (lidars.unsqueeze(0) - lidars.unsqueeze(1)).pow(2) * self.mask_w        
        lid_dists = lid_dists.sum(dim=-1).sqrt() 
        # Normalize distances to [0, 1]
        lid_dists /= self.norm

        return lid_dists
    
    def _goal_diffs(self, gds: torch.Tensor, angles: torch.Tensor):
        """
        Compute in-batch position differences w.r.t. the goal.
        """

        # Differences between in-batch anchors goal distances
        gd_diffs = torch.abs(gds.unsqueeze(0) - gds.unsqueeze(1))

        # Differences between in-batch anchors orientations w.r.t. the goal
        ori_diffs = ((angles.unsqueeze(0) - angles.unsqueeze(1)) + torch.pi) % (2 * torch.pi) - torch.pi
        ori_diffs = torch.abs(ori_diffs) / torch.pi

        return gd_diffs * ori_diffs

    
    def _in_batch_scores(self, lidars: torch.Tensor, gds: torch.Tensor, angles: torch.Tensor):
        """
        Compute in-batch negative scores for anchors.
        """

        # Distances between in-batch examples
        if self.metric == 'lidar':
            batch_scores = self._lidar_dists(lidars)
        elif self.metric == 'goal':
            batch_scores = self._goal_diffs(gds, angles)
        else:
            batch_scores = self._lidar_dists(lidars) *  self._goal_diffs(gds, angles)      

        return batch_scores
       

class SNNCosineSimilarityLoss(SoftNearestNeighbor):
    def __init__(self, args, tau_min: float=0.1, tau_max: float=1.0):
        """
        SNN objective based on Cosine Similarity of embeddings.
        """
        super().__init__(
            args=args,
            tau_min=tau_min,
            tau_max=tau_max
        )

    def __call__(
            self,
            anc_batch: torch.Tensor, 
            pos_batch: torch.Tensor,
            pos_sim_scores: torch.Tensor, 
            lidars: torch.Tensor=None, 
            gds: torch.Tensor=None, 
            angles: torch.Tensor=None,
            neg_batch: torch.Tensor=None, 
            neg_sim_scores: torch.Tensor=None
        ):
        """
        New implementation maps similarities/distances in [tau_min, tau_max]:

                tau(i,j) = tau_min + (tau_max-tau_min) * d_ij/d_max 
        """ 
        if torch.isnan(anc_batch).any() or torch.isnan(pos_batch).any():
            print("FATAL: NaN detected in network outputs (anc_batch or pos_batch).")

        # Embedding similarities between anchors and positive examples
        pos_sims = F.cosine_similarity(anc_batch.unsqueeze(1), pos_batch, dim=-1)
        pos_tau = self.tau_min + (self.tau_max - self.tau_min) * (1-pos_sim_scores)
        pos_sims = torch.exp(pos_sims / pos_tau)  # removed '-' in front of similarity

        if neg_batch is not None:
            # Embedding similarities between anchors and positive examples
            neg_sims = F.cosine_similarity(anc_batch.unsqueeze(1), neg_batch, dim=-1)
            # Adaptive temperatures
            neg_tau = self.tau_min + (self.tau_max - self.tau_min) * (1-neg_sim_scores)
            neg_sims = torch.exp(neg_sims / neg_tau)  # removed '-' in front of similarity

            # Compute SNN loss with sampled negative examples
            num = pos_sims.sum(dim=1)
            den = torch.cat([pos_sims, neg_sims], dim=1).sum(dim=1)
            snn = -torch.log(num / den)
        else:
            # Compute SNN loss with in-batch negative examples
            num = pos_sims.sum(dim=1)
            den = self._in_batch_negatives(anc_batch, lidars, gds, angles)
            snn = -torch.log(num / (num + den))

        return snn.mean()
    
    def _in_batch_negatives(self, anc_batch, lidars, gds, angles):
        # In-batch distance scores
        batch_scores = self._in_batch_scores(lidars, gds, angles)

        # Embedding similarities between anchors and in-batch negative examples
        batch_sims = F.cosine_similarity(anc_batch.unsqueeze(1), anc_batch.unsqueeze(0), dim=2)

        # Mask out self-similarities
        B = anc_batch.shape[0]
        mask = ~torch.eye(B, device=anc_batch.device, dtype=torch.bool)
        batch_sims = batch_sims[mask].view(B, B-1)
        batch_scores = batch_scores[mask].view(B, B-1)

        # Adaptive temperatures
        batch_tau = self.tau_min + (self.tau_max - self.tau_min)*batch_scores
        batch_sims = torch.exp(batch_sims / batch_tau)  # removed '-' in front of similarity

        return batch_sims.sum(dim=1)

 
class SNNSimCLR(SoftNearestNeighbor):
    def __init__(self, args, tau_min: float=0.1, tau_max: float=1.0):
        """
        SNN objective based on SimCLR framework.
        """
        super().__init__(
            args=args,
            tau_min=tau_min,
            tau_max=tau_max
        )

    def __call__(
            self,
            anc_batch: torch.Tensor, 
            pos_batch: torch.Tensor, 
            lidars: torch.Tensor, 
            gds: torch.Tensor, 
            angles: torch.Tensor
        ):

        # Compute overall similarity matrix
        features = torch.cat([anc_batch, pos_batch], dim=0)        
        sim_mat = F.cosine_similarity(features.unsqueeze(0), features.unsqueeze(1), dim=2)        

        # Retrieve dimensions
        B = anc_batch.shape[0] # Batch size
        N = features.shape[0]  # 2 * Batch size

        # Similarity masks
        device = features.device
        mask = ~torch.eye(N, dtype=bool, device=device)
        labels = torch.arange(B, device=device).repeat(2)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & mask
        neg_mask = (labels.unsqueeze(0)!= labels.unsqueeze(1))

        # Positive and negative similarities
        pos_sims = sim_mat[pos_mask].view(N, 1)
        neg_sims = sim_mat[neg_mask].view(N, -1)

        # In-batch negative scores
        batch_tau = self._in_batch_scores(lidars, gds, angles)
        batch_tau_mat = batch_tau.repeat(2, 2)
        batch_tau_mat = batch_tau_mat[neg_mask].view(N, -1)
        batch_tau_mat = self.tau_min + (self.tau_max - self.tau_min)*batch_tau_mat
        
        # Adaptive temperature scaling
        pos_logits = pos_sims / self.tau_min
        neg_logits = neg_sims / batch_tau_mat

        # Loss computation
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        ce_labels = torch.zeros(N, dtype=torch.long, device=device)
        
        return F.cross_entropy(logits, ce_labels)