# models/discriminator.py
import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    """
    A simple discriminator for adversarial domain adaptation.
    It takes a feature vector and outputs a single logit indicating whether
    the feature is from the real or synthetic domain [1].
    """
    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classifies the domain of the input features.

        Args:
            features (torch.Tensor): The feature vector to classify (B, in_features).

        Returns:
            torch.Tensor: A single logit for each feature vector in the batch (B, 1).
        """
        return self.discriminator(features)
