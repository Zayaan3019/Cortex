# models/image_processor.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms import ToTensor, Normalize

class ImageProcessor(nn.Module):
    """
    Processes a single 2D RGB image to generate a pseudo-3D point cloud.
    This module combines:
    1. Semantic Segmentation (U-Net style, here using a pre-trained FCN) to create a foreground mask.
    2. Monocular Depth Estimation (MiDaS) to get a depth map.
    The mask and depth map are then combined and back-projected to form a point cloud [1].
    """
    def __init__(self, num_points=2048):
        super().__init__()
        self.num_points = num_points

        # 1. Load pre-trained MiDaS model for depth estimation
        # Using a robust, publicly available model as is common practice.
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform_depth = self.midas_transforms.dpt_transform

        # 2. Load pre-trained FCN model for semantic segmentation
        weights = FCN_ResNet50_Weights.DEFAULT
        self.segmentation_model = fcn_resnet50(weights=weights)
        self.transform_seg = weights.transforms()

        # Set models to evaluation mode as we are using them for inference
        self.depth_model.eval()
        self.segmentation_model.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of images to a batch of point clouds.

        Args:
            image (torch.Tensor): Input RGB image tensor (B, 3, H, W).

        Returns:
            torch.Tensor: A pseudo-3D point cloud (B, N, 3).
        """
        B, _, H, W = image.shape
        device = image.device

        # --- Depth Estimation ---
        # MiDaS requires specific input normalization
        transformed_image_depth = torch.stack([self.transform_depth(img) for img in image]).to(device)
        depth_map = self.depth_model(transformed_image_depth)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        # --- Semantic Segmentation ---
        processed_image_seg = self.transform_seg(image)
        seg_output = self.segmentation_model(processed_image_seg)['out']
        # Use argmax to get the class for each pixel. We are interested in the main foreground object.
        # This can be made more sophisticated by targeting specific class IDs.
        foreground_mask = (torch.argmax(seg_output, dim=1) > 0).float() # Simple foreground/background

        # --- Point Cloud Generation ---
        # Create a meshgrid of pixel coordinates
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x, y = x.to(device), y.to(device)
        
        # Normalize pixel coordinates to create camera rays
        focal_length = max(H, W) # A common heuristic
        x_cam = (x - W / 2) / focal_length
        y_cam = (y - H / 2) / focal_length

        point_clouds = []
        for i in range(B):
            mask_i = foreground_mask[i]
            depth_i = depth_map[i]
            
            # Apply mask to get foreground pixels
            valid_pixels = mask_i > 0.5
            if valid_pixels.sum() == 0: # Handle cases with no detected foreground
                valid_pixels = torch.ones_like(mask_i, dtype=torch.bool)

            # Back-project pixels to 3D space
            z_cam_i = depth_i[valid_pixels]
            x_cam_i = x_cam[valid_pixels] * z_cam_i
            y_cam_i = y_cam[valid_pixels] * z_cam_i
            
            pc_i = torch.stack([x_cam_i, y_cam_i, z_cam_i], dim=-1)

            # Subsample or pad to a fixed number of points
            if pc_i.shape[0] > self.num_points:
                indices = torch.randperm(pc_i.shape[0])[:self.num_points]
                pc_i = pc_i[indices]
            elif pc_i.shape[0] < self.num_points and pc_i.shape[0] > 0:
                padding_indices = torch.randint(pc_i.shape[0], (self.num_points - pc_i.shape[0],))
                pc_i = torch.cat([pc_i, pc_i[padding_indices]], dim=0)
            else: # If no points were generated
                pc_i = torch.zeros((self.num_points, 3), device=device)
            
            point_clouds.append(pc_i)

        return torch.stack(point_clouds)



