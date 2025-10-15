"""
5-Fold Cross-Validation for MTWAE Model

This script performs systematic cross-validation experiments:
1. Compare weighting strategies (inverse, equal, uncertainty) with k=8
2. Evaluate latent space dimensionality k ∈ {2, 4, 8, 16}
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from models import MTWAE
from data import load_text_data
from train import train_epoch, evaluate
from utils import set_seed, inverse_sample_size_weights


def _to_loader(X, y, batch, shuffle=True, drop_last=True):
    """Convert numpy arrays to PyTorch DataLoader."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, 
                     shuffle=shuffle, drop_last=drop_last)


def train_and_evaluate_fold(data_splits, D, args, task_weights, device):
    """Train multi-task model on one fold and return validation metrics.
    
    Args:
        data_splits: Dict containing train/val splits for all tasks
                    {'Bs': (X_tr, y_tr, X_val, y_val), 'Hc': ..., 'Dc': ...}
        D: Input feature dimension
        args: Training arguments
        task_weights: Dict of task weights {'Bs': w1, 'Hc': w2, 'Dc': w3}
        device: Training device
    
    Returns:
        Dict of validation metrics (MAE and R² for each task)
    """
    # Standardize targets for each task
    scalers = {}
    train_loaders = {}
    val_loaders = {}
    
    for task in ['Bs', 'Hc', 'Dc']:
        X_tr, y_tr, X_val, y_val = data_splits[task]
        
        # Standardize targets
        sc = StandardScaler()
        y_tr_s = sc.fit_transform(y_tr)
        y_val_s = sc.transform(y_val)
        scalers[task] = sc
        
        # Create dataloaders
        train_loaders[task] = _to_loader(X_tr, y_tr_s, args.batch, 
                                         shuffle=True, drop_last=True)
        val_loaders[task] = _to_loader(X_val, y_val_s, args.batch, 
                                       shuffle=False, drop_last=False)
    
    # Initialize model
    use_uncert = (args.weighting == 'uncertainty')
    model = MTWAE(in_features=D, latent_size=args.latent, 
                  use_uncertainty_weighting=use_uncert).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Training loop - ALL THREE TASKS TRAINED JOINTLY
    for epoch in range(args.epochs):
        _ = train_epoch(model, train_loaders, opt, device, 
                       args.sigma, args.lambda_mmd, task_weights)
    
    # Final evaluation on all tasks
    _, mae, r2 = evaluate(model, val_loaders, device, 
                         args.sigma, args.lambda_mmd)
    
    return {'MAE': mae, 'R2': r2}


def multi_task_cross_validate(data_dict, D, args, task_weights, device, n_folds=5):
    """Perform k-fold cross-validation with JOINT multi-task training.
    
    All three tasks are trained together in each fold, sharing the same latent space.
    
    Args:
        data_dict: Dict containing data for all tasks
                  {'Bs': (X, y), 'Hc': (X, y), 'Dc': (X, y)}
        D: Feature dimension
        args: Training arguments
        task_weights: Task weights {'Bs': w1, 'Hc': w2, 'Dc': w3}
        device: Device for training
        n_folds: Number of CV folds
    
    Returns:
        Dict with mean and std of MAE and R² for each task
    """
    # Prepare KFold for each task (different sample sizes)
    kfolds = {}
    for task in ['Bs', 'Hc', 'Dc']:
        X, y = data_dict[task]
        kfolds[task] = list(KFold(n_splits=n_folds, shuffle=True, 
                                  random_state=args.seed).split(X))
    
    # Store results for each fold
    results = {task: {'MAE': [], 'R2': []} for task in ['Bs', 'Hc', 'Dc']}
    
    print(f"  Running {n_folds}-fold CV: ", end='', flush=True)
    
    for fold_idx in range(n_folds):
        # Prepare data splits for all tasks in this fold
        data_splits = {}
        for task in ['Bs', 'Hc', 'Dc']:
            X, y = data_dict[task]
            train_idx, val_idx = kfolds[task][fold_idx]
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            data_splits[task] = (X_tr, y_tr, X_val, y_val)
        
        # Train and evaluate with JOINT multi-task learning
        metrics = train_and_evaluate_fold(
            data_splits, D, args, task_weights, device
        )
        
        # Store results
        for task in ['Bs', 'Hc', 'Dc']:
            results[task]['MAE'].append(metrics['MAE'][task])
            results[task]['R2'].append(metrics['R2'][task])
        
        print(f"{fold_idx+1}", end=' ', flush=True)
    
    print("✓")
    
    # Compute statistics
    output = {}
    for task in ['Bs', 'Hc', 'Dc']:
        output[task] = {
            'MAE_mean': float(np.mean(results[task]['MAE'])),
            'MAE_std': float(np.std(results[task]['MAE'])),
            'R2_mean': float(np.mean(results[task]['R2'])),
            'R2_std': float(np.std(results[task]['R2'])),
            'MAE_folds': [float(m) for m in results[task]['MAE']],
            'R2_folds': [float(r) for r in results[task]['R2']]
        }
    
    return output


def experiment_weighting_strategies(data_dict, D, args, device, n_folds=5):
    """Compare different weighting strategies using 5-fold CV.
    
    Args:
        data_dict: Dict containing data for all tasks
        D: Feature dimension
        args: Training arguments
        device: Device
        n_folds: Number of CV folds
    
    Returns:
        Dict of results for each weighting strategy
    """
    print("Experiment 1: Comparing Weighting Strategies (k=8)")
    strategies = ['inverse', 'equal', 'uncertainty']
    results = {}
    
    # Compute dataset sizes for weighting
    nB = len(data_dict['Bs'][0])
    nH = len(data_dict['Hc'][0])
    nD = len(data_dict['Dc'][0])
    
    for strategy in strategies:
        print(f"\nWeighting strategy: {strategy}")
        args.weighting = strategy
        
        # Compute weights based on strategy
        if strategy == 'inverse':
            weights = inverse_sample_size_weights(nB, nH, nD)
        elif strategy == 'equal':
            weights = {'Bs': 1.0, 'Hc': 1.0, 'Dc': 1.0}
        else:  # uncertainty
            weights = {'Bs': 1.0, 'Hc': 1.0, 'Dc': 1.0}  # Learnable during training
        
        print(f"  Task weights: Bs={weights['Bs']:.3f}, Hc={weights['Hc']:.3f}, Dc={weights['Dc']:.3f}")
        
        # Run multi-task CV
        cv_results = multi_task_cross_validate(
            data_dict, D, args, weights, device, n_folds
        )
        
        results[strategy] = cv_results
        
        # Compute mean R² across all tasks
        mean_r2 = np.mean([cv_results[t]['R2_mean'] for t in ['Bs', 'Hc', 'Dc']])
        results[strategy]['Mean_R2'] = float(mean_r2)
        
        print(f"  Results:")
        for task in ['Bs', 'Hc', 'Dc']:
            print(f"    {task}: R²={cv_results[task]['R2_mean']:.4f}±{cv_results[task]['R2_std']:.4f}, "
                  f"MAE={cv_results[task]['MAE_mean']:.4f}±{cv_results[task]['MAE_std']:.4f}")
        print(f"  Mean R² across tasks: {mean_r2:.4f}")
    
    return results


def experiment_latent_dimensions(data_dict, D, args, device, n_folds=5):
    """Evaluate impact of latent space dimensionality.
    
    Args:
        data_dict: Dict containing data for all tasks
        D: Feature dimension
        args: Training arguments
        device: Device
        n_folds: Number of CV folds
    
    Returns:
        Dict of results for each latent dimension
    """
    print("Experiment 2: Impact of Latent Space Dimensionality")

    
    latent_dims = [2, 4, 8, 16]
    results = {}
    
    # Use inverse weighting
    args.weighting = 'inverse'
    nB = len(data_dict['Bs'][0])
    nH = len(data_dict['Hc'][0])
    nD = len(data_dict['Dc'][0])
    weights = inverse_sample_size_weights(nB, nH, nD)
    
    for k in latent_dims:
        print(f"\nLatent dimension k={k}")
        args.latent = k
        
        # Run multi-task CV
        cv_results = multi_task_cross_validate(
            data_dict, D, args, weights, device, n_folds
        )
        
        results[f'k{k}'] = cv_results
        
        # Compute mean metrics
        mean_r2 = np.mean([cv_results[t]['R2_mean'] for t in ['Bs', 'Hc', 'Dc']])
        mean_mae = np.mean([cv_results[t]['MAE_mean'] for t in ['Bs', 'Hc', 'Dc']])
        results[f'k{k}']['Mean_R2'] = float(mean_r2)
        results[f'k{k}']['Mean_MAE'] = float(mean_mae)
        
        print(f"  Results:")
        for task in ['Bs', 'Hc', 'Dc']:
            print(f"    {task}: R²={cv_results[task]['R2_mean']:.4f}±{cv_results[task]['R2_std']:.4f}, "
                  f"MAE={cv_results[task]['MAE_mean']:.4f}±{cv_results[task]['MAE_std']:.4f}")
        print(f"  Mean R²: {mean_r2:.4f}, Mean MAE: {mean_mae:.4f}")
    
    return results


def save_results(results, output_dir):
    """Save cross-validation results to JSON and CSV files.
    
    Args:
        results: Dict of experimental results
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results as JSON
    json_path = os.path.join(output_dir, 'cv_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Save weighting strategy comparison as CSV
    if 'weighting' in results:
        rows = []
        for strategy in ['inverse', 'equal', 'uncertainty']:
            for task in ['Bs', 'Hc', 'Dc']:
                rows.append({
                    'Strategy': strategy,
                    'Task': task,
                    'R2_mean': results['weighting'][strategy][task]['R2_mean'],
                    'R2_std': results['weighting'][strategy][task]['R2_std'],
                    'MAE_mean': results['weighting'][strategy][task]['MAE_mean'],
                    'MAE_std': results['weighting'][strategy][task]['MAE_std'],
                })
            rows.append({
                'Strategy': strategy,
                'Task': 'Mean',
                'R2_mean': results['weighting'][strategy]['Mean_R2'],
                'R2_std': np.nan,
                'MAE_mean': np.nan,
                'MAE_std': np.nan,
            })
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'weighting_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"Weighting comparison saved to {csv_path}")
    
    # Save latent dimension comparison as CSV
    if 'latent_dim' in results:
        rows = []
        for k in [2, 4, 8, 16]:
            for task in ['Bs', 'Hc', 'Dc']:
                rows.append({
                    'Latent_k': k,
                    'Task': task,
                    'R2_mean': results['latent_dim'][f'k{k}'][task]['R2_mean'],
                    'R2_std': results['latent_dim'][f'k{k}'][task]['R2_std'],
                    'MAE_mean': results['latent_dim'][f'k{k}'][task]['MAE_mean'],
                    'MAE_std': results['latent_dim'][f'k{k}'][task]['MAE_std'],
                })
            rows.append({
                'Latent_k': k,
                'Task': 'Mean',
                'R2_mean': results['latent_dim'][f'k{k}']['Mean_R2'],
                'R2_std': np.nan,
                'MAE_mean': results['latent_dim'][f'k{k}']['Mean_MAE'],
                'MAE_std': np.nan,
            })
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'latent_dimension_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"Latent dimension comparison saved to {csv_path}")


def main():
    ap = argparse.ArgumentParser(description='5-fold cross-validation for MTWAE')
    
    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    
    # Training parameters
    ap.add_argument('--epochs', type=int, default=800,
                   help='Number of training epochs per fold')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--sigma', type=float, default=8.0)
    ap.add_argument('--lambda_mmd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=8)
    
    # Experiment selection
    ap.add_argument('--experiment', choices=['weighting', 'latent_dim', 'both'], 
                   default='both',
                   help='Which experiment to run')
    ap.add_argument('--n_folds', type=int, default=5,
                   help='Number of cross-validation folds')
    ap.add_argument('--output_dir', default='cv_results',
                   help='Directory to save results')
    
    args = ap.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )
    print(f"Dataset sizes - Bs: {len(Xb)}, Hc: {len(Xh)}, Dc: {len(Xd)}")
    
    data_dict = {
        'Bs': (Xb, yb),
        'Hc': (Xh, yh),
        'Dc': (Xd, yd)
    }
    
    # Run experiments
    all_results = {}
    
    if args.experiment in ['weighting', 'both']:
        args.latent = 8  # Fixed k=8 for weighting comparison
        weighting_results = experiment_weighting_strategies(
            data_dict, D, args, device, args.n_folds
        )
        all_results['weighting'] = weighting_results
    
    if args.experiment in ['latent_dim', 'both']:
        latent_results = experiment_latent_dimensions(
            data_dict, D, args, device, args.n_folds
        )
        all_results['latent_dim'] = latent_results
    
    # Save results
    save_results(all_results, args.output_dir)
    
    print("Cross-validation complete!")



if __name__ == '__main__':
    main()