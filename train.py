from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from .losses import recon_loss_bce, mmd_imq, kendall_weight


def train_epoch(model, loaders: Dict[str, DataLoader], optimizer, device,
                sigma: float, lambda_mmd: float, task_weights: Dict[str, float]):
    """Train MTWAE for one epoch across all tasks.
    
    Implements the resampling strategy: iterates through max(batch_counts)
    mini-batches, cycling through smaller datasets to equalize training frequency.
    
    Args:
        model: MTWAE model
        loaders: Dict of DataLoaders for each task ('Bs', 'Hc', 'Dc')
        optimizer: Optimizer
        device: Training device (cuda/cpu)
        sigma: Std of Gaussian prior for MMD 
        lambda_mmd: Weight for MMD loss
        task_weights: Task-specific weights for loss balancing
    
    Returns:
        Dict of average losses per task
    """
    model.train()
    
    # Determine number of batches (use max for resampling strategy)
    num_batches = {t: len(ld) for t, ld in loaders.items()}
    max_batches = max(num_batches.values())
    iters = {t: iter(ld) for t, ld in loaders.items()}

    logs = {"Bs": 0.0, "Hc": 0.0, "Dc": 0.0, "MMD": 0.0}
    
    for _ in range(max_batches):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        
        # Process one batch from each task
        for task in ("Bs", "Hc", "Dc"):
            # Resampling: restart iterator if exhausted (for smaller datasets)
            try:
                x, y = next(iters[task])
            except StopIteration:
                iters[task] = iter(loaders[task])
                x, y = next(iters[task])
            
            x, y = x.to(device), y.to(device)
            x_hat, z, yb, yh, yd = model(x)

            # Compute loss components: L_task = L_recon + L_pro + L_MMD
            recon = recon_loss_bce(x_hat, x)  # Reconstruction loss
            pred = yb if task == "Bs" else yh if task == "Hc" else yd
            prop = F.mse_loss(pred, y, reduction="mean")  # Property prediction loss
            
            # MMD loss: regularize latent distribution to match Gaussian prior
            z_prior = sigma * torch.randn_like(z)
            mmd = lambda_mmd * mmd_imq(z, z_prior, z.size(1))

            loss_sum = recon + prop + mmd
            
            # Apply weighting strategy
            if model.use_uncertainty:
                # Kendall uncertainty weighting: learnable task difficulty
                log_sigma = (model.log_sigma_Bs if task == "Bs" else 
                           model.log_sigma_Hc if task == "Hc" else 
                           model.log_sigma_Dc)
                loss = kendall_weight(loss_sum, log_sigma)
            else:
                # Fixed weighting (equal or inverse sample size)
                loss = task_weights[task] * loss_sum

            total = total + loss
            logs[task] += loss.detach().item()
            logs["MMD"] += mmd.detach().item()
        
        # Update parameters with combined loss
        total.backward()
        optimizer.step()

    # Average losses across all batches
    for k in logs:
        logs[k] /= max_batches
    return logs


@torch.no_grad()
def evaluate(model, loaders: Dict[str, DataLoader], device,
            sigma: float, lambda_mmd: float):
    """Evaluate MTWAE on validation/test set.
    
    Args:
        model: MTWAE model
        loaders: Dict of DataLoaders for each task
        device: Evaluation device
        sigma: Std of Gaussian prior for MMD
        lambda_mmd: Weight for MMD loss
    
    Returns:
        Tuple of (losses_dict, MAE_dict, R2_dict)
    """
    model.eval()
    logs = {"Bs": 0.0, "Hc": 0.0, "Dc": 0.0, "MMD": 0.0}
    preds = {"Bs": [], "Hc": [], "Dc": []}
    trues = {"Bs": [], "Hc": [], "Dc": []}
    counts = {t: 0 for t in ("Bs", "Hc", "Dc")}

    for task in ("Bs", "Hc", "Dc"):
        for x, y in loaders[task]:
            x, y = x.to(device), y.to(device)
            x_hat, z, yb, yh, yd = model(x)
            
            # Compute losses
            recon = recon_loss_bce(x_hat, x)
            pred = yb if task == "Bs" else yh if task == "Hc" else yd
            prop = F.mse_loss(pred, y, reduction="mean")
            z_prior = sigma * torch.randn_like(z)
            mmd = lambda_mmd * mmd_imq(z, z_prior, z.size(1))

            logs[task] += (recon + prop + mmd).item()
            logs["MMD"] += mmd.item()
            
            # Collect predictions and ground truth for metrics
            preds[task].append(pred.cpu().numpy())
            trues[task].append(y.cpu().numpy())
            counts[task] += 1

    # Average losses
    for k in ("Bs", "Hc", "Dc"):
        logs[k] /= max(1, counts[k])
    logs["MMD"] /= sum(counts.values())

    # Compute MAE and R² metrics
    maes, r2s = {}, {}
    for t in ("Bs", "Hc", "Dc"):
        yhat = np.concatenate(preds[t], axis=0)
        ytru = np.concatenate(trues[t], axis=0)
        maes[t] = float(mean_absolute_error(ytru, yhat))
        r2s[t] = float(r2_score(ytru, yhat))
    
    return logs, maes, r2s