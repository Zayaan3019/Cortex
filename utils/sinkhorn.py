# utils/sinkhorn.py
import torch

def sinkhorn_algorithm(cost_matrix: torch.Tensor, lambda_reg: float, iterations: int) -> torch.Tensor:
    """
    Differentiable Sinkhorn algorithm to transform a cost matrix into a doubly stochastic matrix.
    This function implements the iterative projection described in the Cortex-Synth paper [1].

    Args:
        cost_matrix (torch.Tensor): The input cost matrix (B, N, N).
        lambda_reg (float): Entropy regularization strength.
        iterations (int): Number of Sinkhorn iterations.

    Returns:
        torch.Tensor: The resulting soft assignment matrix P (B, N, N).
    """
    # Using log-space for numerical stability
    log_alpha = -lambda_reg * cost_matrix
    for _ in range(iterations):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)
