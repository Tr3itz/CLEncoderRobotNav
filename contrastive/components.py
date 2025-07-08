import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class SoftNearestNeighbor(nn.Module):
    def __init__(self, metric:str, tau_min: float=0.1, tau_max: float=1.0):
        """
        Adaptive contrastive learning objective implementation.
        ----------
        Parameters:
        - metric: str        - metric to use for computing similarity between examples (lidar, goal, both)
        - tau_min: float     - adaptive temperature minimum value
        - tau_max: float     - adaptive temperature maximum value
        """
        super().__init__()

        # Metric
        assert metric in ['lidar', 'goal', 'both']
        self.metric = metric

        # Temperatures range
        self.tau_min, self.tau_max = tau_min, tau_max

    def __call__(
            self,
            anc_batch: torch.Tensor, 
            pos_batch: torch.Tensor,
            pos_sim_scores: torch.Tensor, 
            lidars: torch.Tensor=None, 
            gds: torch.Tensor=None, 
            neg_batch: torch.Tensor=None, 
            neg_sim_scores: torch.Tensor=None
        ):
        """
        New implementation maps similarities/distances in [tau_min, tau_max]:

                tau(i,j) = tau_min + (tau_max-tau_min) * d_ij/d_max
        ----------
        IMPORTANT:
        - positive examples use distances as adaptive temperatures 
          (the closer the sample, the higher the numerator)
        - negative examples use similarities as adaptive temperatures 
          (the more similar, the less impact on the denominator) 
        """ 

        # Embedding similarities between anchors and positive examples
        pos_sims = F.cosine_similarity(anc_batch.unsqueeze(1), pos_batch, dim=-1)
        pos_tau = self.tau_min + (self.tau_max - self.tau_min) * (1-pos_sim_scores)
        pos_sims = torch.exp(pos_sims / pos_tau)  # removed '-' in front of similarity

        if neg_batch is not None:
            # Embedding similarities between anchors and positive examples
            neg_sims = F.cosine_similarity(anc_batch.unsqueeze(1), neg_batch, dim=-1)
            # Adaptive temperatures
            neg_tau = self.tau_min + (self.tau_max - self.tau_min) * neg_sim_scores
            neg_sims = torch.exp(neg_sims / neg_tau)  # removed '-' in front of similarity

            # Compute SNN loss with sampled negative examples
            num = pos_sims.sum(dim=1)
            den = torch.cat([pos_sims, neg_sims], dim=1).sum(dim=1)
            snn = -torch.log(num / den)
        else:
            # Compute SNN loss with in-batch negative examples
            num = pos_sims.sum(dim=1)
            den = self._in_batch_negatives(anc_batch, lidars, gds)
            snn = -torch.log(num / (num + den))

        return snn.mean()
    
    def _in_batch_negatives(self, anc_batch: torch.Tensor, lidars: torch.Tensor, gds: torch.Tensor):
        """
        Compute in-batch negatives for anchors.
        """

        # Eucledian distances between in-batch anchors lidar readings
        lid_dists = (lidars.unsqueeze(0) - lidars.unsqueeze(1)).pow(2).sum(dim=-1).sqrt() 
        # Normalize distances to [0, 1]
        lid_dists /= sqrt(lidars.shape[1])

        # Differences between in-batch anchors goal distances
        gd_diffs = torch.abs(gds.unsqueeze(0) - gds.unsqueeze(1))

        # Distances between in-batch examples
        if self.metric == 'lidar':
            batch_dists = lid_dists
        elif self.metric == 'goal':
            batch_dists = gd_diffs
        else:
            batch_dists = lid_dists * gd_diffs       

        # Embedding similarities between anchors and in-batch negative examples
        batch_sims = F.cosine_similarity(anc_batch.unsqueeze(1), anc_batch.unsqueeze(0), dim=2)

        # Mask out self-similarities
        B = anc_batch.shape[0]
        mask = ~torch.eye(B, device=anc_batch.device, dtype=torch.bool)
        batch_sims = batch_sims[mask].view(B, B-1)
        batch_dists = batch_dists[mask].view(B, B-1)

        # Adaptive temperatures
        batch_tau = self.tau_min + (self.tau_max - self.tau_min)*((1-batch_dists) / self.tau_max)
        batch_sims = torch.exp(batch_sims / batch_tau)  # removed '-' in front of similarity

        return batch_sims.sum(dim=1)