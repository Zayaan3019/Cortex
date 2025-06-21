# utils/losses.py
import torch
import torch.nn.functional as F
from utils.laplacian import compute_laplacian

def geometry_loss(pred_joints, gt_joints, loss_type='mpjpe'):
    """Calculates Mean Per Joint Position Error (MPJPE) or Chamfer Distance."""
    if loss_type == 'mpjpe':
        return torch.mean(torch.norm(pred_joints - gt_joints, dim=-1))
    elif loss_type == 'chamfer':
        # A simplified Chamfer implementation
        dist1 = torch.cdist(pred_joints, gt_joints).min(dim=2)[0]
        dist2 = torch.cdist(gt_joints, pred_joints).min(dim=2)[0]
        return torch.mean(dist1) + torch.mean(dist2)
    else:
        raise ValueError("Unsupported geometry loss type")

def spectral_loss(pred_adj, gt_adj):
    """
    Computes the spectral loss based on the Frobenius norm of Laplacian difference.
    This loss enforces topological consistency, as described in the paper [1].
    """
    laplacian_pred = compute_laplacian(pred_adj, normalize=True)
    laplacian_gt = compute_laplacian(gt_adj, normalize=True)
    return F.mse_loss(laplacian_pred, laplacian_gt)

def adversarial_loss_g(discriminator_output):
    """Generator's adversarial loss to fool the discriminator."""
    return F.binary_cross_entropy_with_logits(discriminator_output, torch.ones_like(discriminator_output))

def adversarial_loss_d(d_real, d_fake):
    """Discriminator's loss to distinguish real vs. synthetic features."""
    loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return (loss_real + loss_fake) / 2
