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
        if x.dim() > 4:
            """
            Input shape: [batch_size, #ex, C, H, W]
            ------------
            Treat each set of positive examples as a batch of images.
            """
            
            # Positive/negative examples encodings
            encodings = [self.encoder(ex) for ex in x]

            # Output shape: [batch_size, #ex, out_dim]
            return torch.stack(encodings, dim=0)
        elif x.dim() < 4:
            # Single image encoding
            return self.encoder(x.unsqueeze(0))

        # Anchor encodings
        return self.encoder(x)