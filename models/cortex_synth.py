# models/cortex_synth.py
import torch
import torch.nn as nn
from .image_processor import ImageProcessor
from .pointcloud_encoder import PointNetPlusPlus
from .skeleton_decoder import SkeletonDecoder
from .cgcn import CGCN

class CortexSynth(nn.Module):
    """
    The complete Cortex-Synth model, integrating all four modules [1].
    """
    def __init__(self, config):
        super().__init__()
        # 1. Image Processor (Segmentation + Depth Estimation)
        self.image_processor = ImageProcessor()
        
        # 2. Enhanced PointNet++ Encoder
        self.pointcloud_encoder = PointNetPlusPlus(num_classes=config['encoder_channels'][-1])
        
        # 3. Skeleton Decoder (MLP for joint prediction)
        self.skeleton_decoder = SkeletonDecoder(
            in_features=config['encoder_channels'][-1],
            hidden_dim=config['decoder_hidden_dim'],
            num_joints=config['num_joints']
        )
        
        # 4. Continuous Graph Construction Network (CGCN)
        self.cgcn = CGCN(
            feature_dim=config['decoder_hidden_dim'],
            num_joints=config['num_joints'],
            gat_layers=config['gat_layers'],
            gat_heads=config['gat_heads'],
            sinkhorn_lambda=config['sinkhorn_lambda'],
            sinkhorn_iter=config['sinkhorn_iterations']
        )

    def forward(self, image: torch.Tensor):
        """
        End-to-end forward pass from a 2D image to a 3D skeleton.

        Args:
            image (torch.Tensor): Input RGB image (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - pred_joints (torch.Tensor): Predicted 3D joint coordinates (B, N, 3).
            - pred_adj (torch.Tensor): Predicted adjacency matrix (B, N, N).
            - global_features (torch.Tensor): Encoded features for domain adaptation loss.
        """
        # 1. Generate pseudo-3D point cloud
        point_cloud = self.image_processor(image)

        # 2. Encode point cloud to get latent features
        global_features = self.pointcloud_encoder(point_cloud)

        # 3. Decode features into joint positions and features
        pred_joints, joint_features = self.skeleton_decoder(global_features)

        # 4. Synthesize the graph topology
        pred_adj = self.cgcn(joint_features, pred_joints)

        return pred_joints, pred_adj, global_features
