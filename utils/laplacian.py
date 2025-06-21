# utils/laplacian.py
import torch

def compute_laplacian(adj_matrix: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Computes the graph Laplacian from a batch of adjacency matrices.

    Args:
        adj_matrix (torch.Tensor): The adjacency matrix (B, N, N).
        normalize (bool): If True, computes the normalized Laplacian.

    Returns:
        torch.Tensor: The graph Laplacian (B, N, N).
    """
    degree_matrix = torch.diag_embed(torch.sum(adj_matrix, dim=-1))
    laplacian = degree_matrix - adj_matrix
    
    if normalize:
        # Compute D^(-1/2)
        deg_inv_sqrt = torch.pow(torch.sum(adj_matrix, dim=-1), -0.5)
        # Replace infs with 0s for isolated nodes
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt_matrix = torch.diag_embed(deg_inv_sqrt)
        
        # L_norm = D^(-1/2) * L * D^(-1/2)
        normalized_laplacian = deg_inv_sqrt_matrix @ laplacian @ deg_inv_sqrt_matrix
        return normalized_laplacian
        
    return laplacian
