# utils/evaluation_metrics.py
import torch
import networkx as nx

def mean_per_joint_position_error(pred_joints, gt_joints):
    """Calculates Mean Per Joint Position Error (MPJPE)."""
    return torch.mean(torch.norm(pred_joints - gt_joints, dim=-1))

def chamfer_distance(pc1, pc2):
    """Calculates the Chamfer Distance between two point clouds."""
    dist1 = torch.cdist(pc1, pc2).min(dim=2)[0]
    dist2 = torch.cdist(pc2, pc1).min(dim=2)[0]
    return (torch.mean(dist1) + torch.mean(dist2)) / 2

def graph_edit_distance(pred_adj, gt_adj, threshold=0.5):
    """
    Calculates the Graph Edit Distance (GED) between two graphs.
    This is computationally expensive and is often estimated or calculated on a per-sample basis.
    """
    pred_adj_binary = (pred_adj > threshold).cpu().numpy()
    gt_adj_binary = (gt_adj > threshold).cpu().numpy()
    
    ged_scores = []
    for i in range(pred_adj_binary.shape[0]):
        g1 = nx.from_numpy_array(pred_adj_binary[i])
        g2 = nx.from_numpy_array(gt_adj_binary[i])
        # Use an approximation for speed, as exact GED is NP-hard.
        ged = nx.graph_edit_distance(g1, g2, timeout=5)
        ged_scores.append(ged if ged is not None else 50) # Use a high value for timeout
        
    return sum(ged_scores) / len(ged_scores)
