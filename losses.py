from __future__ import annotations
import torch
import torch.nn.functional as F


def recon_loss_bce(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reconstruction loss using binary cross-entropy.
    
    For composition data on simplex, BCE treats each element independently
    and avoids numerical issues with log(0) for tiny fractions.
    
    Args:
        x_hat: Reconstructed composition (after Softmax), shape (batch, D)
        x: Original composition, values in [0, 1], shape (batch, D)
    
    Returns:
        Mean BCE loss
    """
    return F.binary_cross_entropy(x_hat, x, reduction="mean")


@torch.jit.script
def _pairwise_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances between two sets of points.
    
    Args:
        a: Tensor of shape (n, d)
        b: Tensor of shape (m, d)
    
    Returns:
        Distance matrix of shape (n, m)
    """
    a_norm = (a**2).sum(1, keepdim=True)
    b_norm = (b**2).sum(1, keepdim=True)
    return a_norm + b_norm.t() - 2.0 * (a @ b.t())


def mmd_imq(z: torch.Tensor, z_prior: torch.Tensor, h_dim: int) -> torch.Tensor:
    """Maximum Mean Discrepancy with Inverse Multiquadric (IMQ) kernel.
    
    Computes MMD between latent distribution and Gaussian prior using
    multi-scale IMQ kernels. This regularizes the encoder to produce
    latent codes that match the prior distribution.
    
    Args:
        z: Latent codes from encoder, shape (batch, latent_dim)
        z_prior: Samples from Gaussian prior, shape (batch, latent_dim)
        h_dim: Latent dimension (used for kernel bandwidth scaling)
    
    Returns:
        Unbiased MMD estimate aggregated over multiple kernel scales
    """
    b = z.size(0)
    d_xx = _pairwise_dists(z, z)
    d_yy = _pairwise_dists(z_prior, z_prior)
    d_xy = _pairwise_dists(z, z_prior)
    
    stats = 0.0
    # Multi-scale bandwidths for robust MMD estimation
    for scale in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        C = 2 * h_dim * scale
        # IMQ kernel: k(x,y) = C / (C + ||x-y||^2)
        k_xx = C / (C + d_xx)
        k_yy = C / (C + d_yy)
        k_xy = C / (C + d_xy)
        
        # Unbiased U-statistic estimator
        eye = torch.eye(b, device=z.device)
        stats = stats + ((k_xx + k_yy) * (1 - eye)).sum() / (b - 1) - 2.0 * k_xy.sum() / b
    
    return stats


def kendall_weight(loss_sum: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Apply Kendall uncertainty weighting for multi-task learning.
    
    Implements the uncertainty weighting scheme from Kendall et al. (2018):
    weighted_loss = (1/(2σ²)) * loss + log(σ)
    
    The learnable parameter σ automatically balances task difficulty,
    where σ = exp(log_sigma).
    
    Args:
        loss_sum: Sum of task-specific losses (L_recon + L_pro + L_MMD)
        log_sigma: Learnable log uncertainty parameter
    
    Returns:
        Uncertainty-weighted loss
    """
    return 0.5 * torch.exp(-2.0 * log_sigma) * loss_sum + log_sigma