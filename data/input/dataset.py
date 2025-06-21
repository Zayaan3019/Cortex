# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticSkeletonDataset(Dataset):
    """Placeholder dataset for synthetic data with image-skeleton pairs."""
    def __init__(self, num_samples=1000, img_size=256, num_joints=17):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_joints = num_joints

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In a real scenario, you would load an image and its corresponding skeleton data
        image = torch.rand(3, self.img_size, self.img_size)
        gt_joints = torch.rand(self.num_joints, 3)
        # Create a plausible random adjacency matrix (e.g., a line graph for simplicity)
        adj = torch.zeros(self.num_joints, self.num_joints)
        for i in range(self.num_joints - 1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
        gt_adj = adj
        
        return image, (gt_joints, gt_adj)

class RealImageDataset(Dataset):
    """Placeholder dataset for real-world images without annotations."""
    def __init__(self, num_samples=500, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In a real scenario, you would load an image from COCO, Pascal3D+, etc.
        image = torch.rand(3, self.img_size, self.img_size)
        # Real data for domain adaptation doesn't need ground truth labels
        return image, torch.empty(0)


def create_dataloaders(config):
    """Creates dataloaders for both synthetic and real data."""
    synthetic_dataset = SyntheticSkeletonDataset()
    real_dataset = RealImageDataset()

    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    real_loader = DataLoader(
        real_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return real_loader, synthetic_loader
