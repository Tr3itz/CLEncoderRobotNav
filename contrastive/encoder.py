import torch
import torch.nn as nn
from torchvision.models import resnet50


class ReprojectionLayer(nn.Module):
    """
    Reprojection layer implementation for vision encoder.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        # Components
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: [#ex, in_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        if x.shape[0] > 1:
            x = self.bn(x)

        return self.fc2(x)
    
    
class ResNetEncoder(nn.Module):
    """
    ResNet50 encoder for Scene Tranfer Contrastive Learning.
    """
    
    def __init__(self, hidden_dim: int=512, out_dim: int=128) -> None:
        super().__init__()

        # ResNet Backbone
        self.encoder = resnet50()

        # Introduce the reprojection layer
        mlp_in = self.encoder.fc.in_features
        self.encoder.fc = ReprojectionLayer(
            in_dim=mlp_in,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)        
        if x.dim() > 4:
            # Input shape: [batch_size, #ex, C, H, W] (e.g., B=32, N=5)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            
            # Shape: [B*N, out_dim]
            embeddings = self.encoder(x) 
            
            # Output shape: [B, N, out_dim]
            return embeddings.view(B, N, -1)
        
        # Shape: [B, C, H, W]
        return self.encoder(x)