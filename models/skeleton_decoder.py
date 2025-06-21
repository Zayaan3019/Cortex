# models/skeleton_decoder.py
import torch
import torch.nn as nn

class SkeletonDecoder(nn.Module):
    """
    Decodes a global feature vector into 3D joint coordinates and per-joint features.
    This module is described as an MLP in the architecture diagram of the paper [1].
    """
    def __init__(self, in_features: int, hidden_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        
        # The MLP that predicts joint coordinates and features
        self.decoder_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_joints * (3 + hidden_dim)) # 3 for xyz, hidden_dim for features
        )
        self.feature_dim = hidden_dim

    def forward(self, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the skeleton decoder.

        Args:
            global_features (torch.Tensor): The latent vector from the encoder (B, in_features).

        Returns:
            A tuple containing:
            - pred_joints (torch.Tensor): Predicted 3D joint coordinates (B, N, 3).
            - joint_features (torch.Tensor): Features for each joint (B, N, feature_dim).
        """
        B = global_features.shape[0]
        
        # Predict flattened joint data
        decoded_output = self.decoder_mlp(global_features)
        
        # Reshape the output to separate coordinates and features
        decoded_output = decoded_output.view(B, self.num_joints, 3 + self.feature_dim)
        
        # Split into coordinates and features
        pred_joints = decoded_output[..., :3]
        joint_features = decoded_output[..., 3:]
        
        return pred_joints, joint_features
