from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 8):
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state_dict, path: str):
    """Save model checkpoint to file.
    
    Args:
        state_dict: Model state dictionary
        path: Save path (creates parent directories if needed)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def inverse_sample_size_weights(nB: int, nH: int, nD: int):
    """Compute inverse sample size weighting for multi-task learning.
    
    Implements the weighting strategy:
    w_task^raw = T / (N_task × K)
    w_task = w_task^raw / Σ w_task^raw
    
    where:
    - T: total number of samples across all tasks
    - N_task: number of samples for specific task
    - K: number of tasks (3)
    
    This weighting boosts the influence of smaller datasets (Hc, Dc)
    and reduces dominance by the largest dataset (Bs).
    
    Args:
        nB: Number of Bs samples (574)
        nH: Number of Hc samples (383)
        nD: Number of Dc samples (311)
    
    Returns:
        Dict of normalized weights for each task
    """
    T = nB + nH + nD  # Total samples
    K = 3  # Number of tasks
    
    # Compute raw weights (inversely proportional to dataset size)
    raw = {
        'Bs': T / (nB * K),
        'Hc': T / (nH * K),
        'Dc': T / (nD * K),
    }
    
    # Normalize weights to sum to 1
    s = sum(raw.values())
    return {k: raw[k] / s for k in raw}

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_checkpoint(model, path: str, device='cpu'):
    """
    Load model checkpoint from file.
    
    Args:
        model: Model instance
        path: Path to checkpoint file
        device: Device to load to ('cpu' or 'cuda')
    """
    import torch
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def standardize_targets(Bs, lnHc, Dc):
    """
    Standardize target properties using StandardScaler.
    
    Args:
        Bs, lnHc, Dc: Target arrays
    
    Returns:
        Tuple of (Bs_scaled, lnHc_scaled, Dc_scaled, scalers_dict)
    """
    from sklearn.preprocessing import StandardScaler
    
    scalers = {}
    
    sc_Bs = StandardScaler()
    Bs_s = sc_Bs.fit_transform(Bs.reshape(-1, 1))
    scalers['Bs'] = sc_Bs
    
    sc_Hc = StandardScaler()
    lnHc_s = sc_Hc.fit_transform(lnHc.reshape(-1, 1))
    scalers['Hc'] = sc_Hc
    
    sc_Dc = StandardScaler()
    Dc_s = sc_Dc.fit_transform(Dc.reshape(-1, 1))
    scalers['Dc'] = sc_Dc
    
    return Bs_s, lnHc_s, Dc_s, scalers


def make_scalers_from_train_split(Bs, lnHc, Dc, split_seed=8, test_size=0.2):
    """
    Create scalers fitted on training split only.
    
    Args:
        Bs, lnHc, Dc: Full target arrays
        split_seed: Random seed for train/test split
        test_size: Fraction for test set
    
    Returns:
        Dict of scalers fitted on training data only
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Split each property
    Bs_tr, _ = train_test_split(Bs, test_size=test_size, random_state=split_seed)
    lnHc_tr, _ = train_test_split(lnHc, test_size=test_size, random_state=split_seed)
    Dc_tr, _ = train_test_split(Dc, test_size=test_size, random_state=split_seed)
    
    # Fit scalers on training data only
    scalers = {}
    
    sc_Bs = StandardScaler()
    sc_Bs.fit(Bs_tr.reshape(-1, 1))
    scalers['Bs'] = sc_Bs
    
    sc_Hc = StandardScaler()
    sc_Hc.fit(lnHc_tr.reshape(-1, 1))
    scalers['Hc'] = sc_Hc
    
    sc_Dc = StandardScaler()
    sc_Dc.fit(Dc_tr.reshape(-1, 1))
    scalers['Dc'] = sc_Dc
    
    return scalers


def plot_kde_overlap(existing, generated, target, xlabel, save_path):
    """
    Plot KDE overlaps for existing, generated, and target distributions.
    
    Args:
        existing: Existing property values
        generated: Generated property values
        target: Target subset property values
        xlabel: X-axis label
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    plt.figure(figsize=(8, 5))
    
    # KDE for each distribution
    if len(existing) > 1:
        kde_exist = stats.gaussian_kde(existing)
        x_range = np.linspace(existing.min(), existing.max(), 200)
        plt.plot(x_range, kde_exist(x_range), 'r-', linewidth=2, label='Existing', alpha=0.7)
    
    if len(generated) > 1:
        kde_gen = stats.gaussian_kde(generated)
        x_range = np.linspace(generated.min(), generated.max(), 200)
        plt.plot(x_range, kde_gen(x_range), 'b-', linewidth=2, label='Generated', alpha=0.7)
    
    if len(target) > 1:
        kde_target = stats.gaussian_kde(target)
        x_range = np.linspace(target.min(), target.max(), 200)
        plt.plot(x_range, kde_target(x_range), 'g-', linewidth=2, label='Target', alpha=0.7)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()