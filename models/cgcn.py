# models/cgcn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from utils.sinkhorn import sinkhorn_algorithm

class CGCN(nn.Module):
    """
    Continuous Graph Construction Network (CGCN).
    This module predicts skeletal topology using a novel differentiable approach.
    It combines an MLP for edge prediction with the Sinkhorn algorithm and GATs for refinement [1].
    """
    def __init__(self, feature_dim: int, num_joints: int, gat_layers: int, gat_heads: int, sinkhorn_lambda: float, sinkhorn_iter: int):
        super().__init__()
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iter = sinkhorn_iter

        # MLP to predict edge costs from joint features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Hierarchical Graph Attention (GAT) for refinement
        self.gat_layers = nn.ModuleList()
        in_channels = feature_dim
        for _ in range(gat_layers):
            self.gat_layers.append(GATConv(in_channels, in_channels, heads=gat_heads, concat=False))

    def forward(self, joint_features: torch.Tensor, joint_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CGCN.

        Args:
            joint_features (torch.Tensor): Features for each predicted joint (B, N, F).
            joint_coords (torch.Tensor): 3D coordinates of each predicted joint (B, N, 3).

        Returns:
            torch.Tensor: The refined adjacency matrix A' (B, N, N).
        """
        B, N, F = joint_features.shape
        
        # 1. Compute pairwise cost matrix C
        # Feature concatenation [fi, fj]
        feat_a = joint_features.unsqueeze(2).repeat(1, 1, N, 1)
        feat_b = joint_features.unsqueeze(1).repeat(1, N, 1, 1)
        
        # Pairwise distance ||xi - xj||
        dist = torch.cdist(joint_coords, joint_coords, p=2).unsqueeze(-1)
        
        pairwise_features = torch.cat([feat_a, feat_b, dist], dim=-1)
        cost_matrix = self.edge_mlp(pairwise_features).squeeze(-1) # (B, N, N)

        # 2. Sinkhorn-based Differentiable Connectivity
        # This is the continuous relaxation of the assignment problem [1].
        adj_matrix = sinkhorn_algorithm(cost_matrix, self.sinkhorn_lambda, self.sinkhorn_iter)
        
        # 3. Hierarchical GAT Refinement
        # Use the soft adjacency matrix for message passing
        edge_index = adj_matrix.to_sparse().indices()
        edge_weight = adj_matrix[edge_index[0], edge_index[1]]

        refined_features = joint_features
        for gat_layer in self.gat_layers:
            refined_features = gat_layer(refined_features.view(B*N, -1), edge_index)
            refined_features = refined_features.view(B, N, -1)
        
        # Recompute adjacency from refined features for final output
        # (This step could also be a direct output of adj_matrix depending on interpretation)
        final_adj = torch.bmm(refined_features, refined_features.transpose(1, 2))
        final_adj = torch.sigmoid(final_adj) # Normalize to [0, 1]
        
        return final_adj
