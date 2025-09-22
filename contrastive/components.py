import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
       

class SNNCosineSimilarityLoss(nn.Module):
    def __init__(self, tau_min: float=0.1, tau_max: float=1.0):
        """
        SNN objective based on Cosine Similarity of embeddings.
        ----------
        Parameters:
            - tau_min: float - minimum adaptive temperature value
            - tau_max: float - maximum adaptive temperature value
        """

        # Temperatures range
        self.tau_min, self.tau_max = tau_min, tau_max

    def __call__(
            self,
            anc_batch: torch.Tensor, 
            pos_batch: torch.Tensor,
            pos_sim_scores: torch.Tensor, 
            neg_batch: torch.Tensor, 
            neg_sim_scores: torch.Tensor
        ):
        """
        Parameters:
            - anc_batch: torch.Tensor      - anchor images
            - pos_batch: torch.Tensor      - positive examples
            - pos_sim_scores: torch.Tensor - psimilarity scores between anchors and positives
            - neg_batch: torch.Tensor      - negative examples
            - neg_sim_scores: torch.Tensor - psimilarity scores between anchors and negatives
        ----------
        New implementation maps similarities/distances in [tau_min, tau_max]:

                tau(i,j) = tau_min + (tau_max-tau_min) * d_ij/d_max 
        """ 
        if torch.isnan(anc_batch).any() or torch.isnan(pos_batch).any():
            print("FATAL: NaN detected in network outputs (anc_batch or pos_batch).")

        # Embedding similarities between anchors and positive examples
        pos_sims = F.cosine_similarity(anc_batch.unsqueeze(1), pos_batch, dim=-1)
        pos_tau = self.tau_min + (self.tau_max - self.tau_min) * (1-pos_sim_scores)
        pos_sims = torch.exp(pos_sims / pos_tau)  # removed '-' in front of similarity
        
        # Embedding similarities between anchors and positive examples
        neg_sims = F.cosine_similarity(anc_batch.unsqueeze(1), neg_batch, dim=-1)
        neg_tau = self.tau_min + (self.tau_max - self.tau_min) * neg_sim_scores
        neg_sims = torch.exp(neg_sims / neg_tau)  # removed '-' in front of similarity

        # Compute SNN loss with sampled negative examples
        num = pos_sims.sum(dim=1)
        den = torch.cat([pos_sims, neg_sims], dim=1).sum(dim=1)
        snn = -torch.log(num / den)

        return snn.mean()

 
class SNNSimCLR(nn.Module):
    def __init__(self, tau_min: float=0.1, tau_max: float=1.0):
        """
        SNN objective based on SimCLR framework.
        ----------
        Parameters:
            - tau_min: float - minimum adaptive temperature value
            - tau_max: float - maximum adaptive temperature value
        """
        # Temperatures range
        self.tau_min, self.tau_max = tau_min, tau_max

    def __call__(
            self,
            anc_batch: torch.Tensor, 
            pos_batch: torch.Tensor,
            tau_fn: Callable,
            *args,
            **kwargs
        ):
        """
        Parameters:
            - anc_batch: torch.Tensor - anchor images
            - pos_batch: torch.Tensor - positive examples
            - tau_fn: Callable        - function for computing in-batch similarity scores
            - *args, **kwargs         - `tau_fn` arguments
        ----------
        New implementation maps similarities/distances in [tau_min, tau_max]:

                tau(i,j) = tau_min + (tau_max-tau_min) * d_ij/d_max 
        """

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
        batch_tau = tau_fn(*args, **kwargs)
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