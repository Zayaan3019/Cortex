# models/pointcloud_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius

class PointNetSetAbstraction(nn.Module):
    """A single layer of the PointNet++ hierarchy."""
    def __init__(self, npoint, radius_val, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius_val
        self.nsample = nsample
        self.group_all = group_all
        self.conv = PointNetConv(
            local_nn=nn.Sequential(*[
                nn.Sequential(nn.Linear(c_in, c_out), nn.ReLU(inplace=True))
                for c_in, c_out in zip(mlp[:-1], mlp[1:])
            ]),
            global_nn=None
        )

    def forward(self, pos, x):
        if self.group_all:
            # Group all points to get a single global feature vector
            new_pos = pos.mean(dim=1, keepdim=True)
            new_x = self.conv((pos, x), (new_pos, None))[1]
            return new_pos, new_x
        
        # Furthest Point Sampling (FPS) to select centroids
        idx = fps(pos, ratio=self.npoint / pos.shape[1])
        row, col = radius(pos, pos[idx], self.radius, max_num_neighbors=self.nsample)
        edge_index = torch.stack([col, row], dim=0)
        
        # PointNetConv for local feature aggregation
        new_pos = pos[idx]
        new_x = self.conv((pos, x), (new_pos, None), edge_index)[1]
        
        return new_pos, new_x

class PointNetPlusPlus(nn.Module):
    """
    Enhanced PointNet++ Encoder for extracting a global feature vector from a point cloud.
    The architecture follows the original PointNet++ paper, which is a state-of-the-art
    choice for robust point cloud feature learning [1].
    """
    def __init__(self, num_classes):
        super().__init__()
        # Three levels of set abstraction
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        
        # Final classifier/regressor head
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, pos):
        """
        Args:
            pos (torch.Tensor): Input point cloud coordinates (B, N, 3).
        
        Returns:
            torch.Tensor: Global feature vector (B, num_classes).
        """
        # PointNet++ operates on (num_points, num_dims)
        B, N, _ = pos.shape
        pos_flat = pos.view(B*N, 3)
        batch = torch.arange(B, device=pos.device).repeat_interleave(N)

        # PyG expects (pos, features)
        # Here, initial features are the coordinates themselves.
        sa0_out = (pos_flat, pos_flat)
        
        sa1_pos, sa1_x = self.sa1(sa0_out[0], sa0_out[1])
        sa2_pos, sa2_x = self.sa2(sa1_pos, sa1_x)
        _, sa3_x = self.sa3(sa2_pos, sa2_x)
        
        # Reshape to (B, C)
        global_feature = sa3_x.view(B, -1)
        
        # Pass through final MLP to get the final latent vector
        latent_vector = self.mlp(global_feature)
        
        return latent_vector
