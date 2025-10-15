"""
Random Search Hyperparameter Optimization for MTWAE

"These hyperparameters were selected through 
30 iterations of cross-validation with random search, ensuring robust 
model performance."

This script performs random search over:
- epochs: Training epochs
- batch_size: Mini-batch size
- learning_rate: Adam optimizer learning rate
- sigma: Gaussian prior standard deviation
- lambda_mmd: MMD loss weight

Each configuration is evaluated using 5-fold cross-validation.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, List, Tuple
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


def sample_hyperparameters(search_space: Dict) -> Dict:
    """Sample hyperparameters from search space.
    
    Args:
        search_space: Dict defining search space for each hyperparameter
        
    Returns:
        Dict of sampled hyperparameters
    """
    config = {}
    
    for param, space in search_space.items():
        if space['type'] == 'int':
            # Sample integer uniformly
            config[param] = np.random.randint(space['low'], space['high'] + 1)
        elif space['type'] == 'float':
            # Sample float uniformly
            config[param] = np.random.uniform(space['low'], space['high'])
        elif space['type'] == 'log':
            # Sample in log space
            log_low = np.log10(space['low'])
            log_high = np.log10(space['high'])
            config[param] = 10 ** np.random.uniform(log_low, log_high)
        elif space['type'] == 'choice':
            # Sample from discrete choices
            config[param] = np.random.choice(space['choices'])
    
    return config


def _to_loader(X, y, batch, shuffle=True, drop_last=True):
    """Convert numpy arrays to PyTorch DataLoader."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, 
                     shuffle=shuffle, drop_last=drop_last)


def cross_validate_config(data_dict, D, config, device, n_folds=5, seed=8):
    """Evaluate a hyperparameter configuration using k-fold CV.
    
    Args:
        data_dict: Dict containing data for all tasks
        D: Input feature dimension
        config: Hyperparameter configuration to evaluate
        device: Training device
        n_folds: Number of CV folds
        seed: Random seed
        
    Returns:
        Dict with mean R² across tasks and per-task metrics
    """
    # Compute task weights (using inverse sample size weighting)
    nB = len(data_dict['Bs'][0])
    nH = len(data_dict['Hc'][0])
    nD = len(data_dict['Dc'][0])
    task_weights = inverse_sample_size_weights(nB, nH, nD)
    
    # Prepare KFold for each task
    kfolds = {}
    for task in ['Bs', 'Hc', 'Dc']:
        X, y = data_dict[task]
        kfolds[task] = list(KFold(n_splits=n_folds, shuffle=True, 
                                  random_state=seed).split(X))
    
    # Store results for each fold
    results = {task: {'R2': [], 'MAE': []} for task in ['Bs', 'Hc', 'Dc']}
    
    for fold_idx in range(n_folds):
        # Prepare data splits for all tasks
        data_splits = {}
        train_loaders = {}
        val_loaders = {}
        
        for task in ['Bs', 'Hc', 'Dc']:
            X, y = data_dict[task]
            train_idx, val_idx = kfolds[task][fold_idx]
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Standardize targets
            sc = StandardScaler()
            y_tr_s = sc.fit_transform(y_tr)
            y_val_s = sc.transform(y_val)
            
            # Create dataloaders
            train_loaders[task] = _to_loader(X_tr, y_tr_s, config['batch_size'], 
                                            shuffle=True, drop_last=True)
            val_loaders[task] = _to_loader(X_val, y_val_s, config['batch_size'], 
                                          shuffle=False, drop_last=False)
        
        # Initialize model
        model = MTWAE(in_features=D, latent_size=config['latent_dim']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        
        # Training loop
        for epoch in range(config['epochs']):
            _ = train_epoch(model, train_loaders, opt, device, 
                          config['sigma'], config['lambda_mmd'], task_weights)
        
        # Final evaluation
        _, mae, r2 = evaluate(model, val_loaders, device, 
                            config['sigma'], config['lambda_mmd'])
        
        # Store results
        for task in ['Bs', 'Hc', 'Dc']:
            results[task]['R2'].append(r2[task])
            results[task]['MAE'].append(mae[task])
    
    # Compute statistics
    output = {}
    r2_values = []
    for task in ['Bs', 'Hc', 'Dc']:
        output[task] = {
            'R2_mean': float(np.mean(results[task]['R2'])),
            'R2_std': float(np.std(results[task]['R2'])),
            'MAE_mean': float(np.mean(results[task]['MAE'])),
            'MAE_std': float(np.std(results[task]['MAE'])),
        }
        r2_values.append(output[task]['R2_mean'])
    
    # Mean R² across all tasks (optimization objective)
    output['mean_R2'] = float(np.mean(r2_values))
    
    return output


def random_search(data_dict, D, search_space, n_iter, device, n_folds=5, seed=8):
    """Perform random search over hyperparameter space.
    
    Args:
        data_dict: Dict containing data for all tasks
        D: Input feature dimension
        search_space: Dict defining search space
        n_iter: Number of random search iterations
        device: Training device
        n_folds: Number of CV folds per configuration
        seed: Random seed
        
    Returns:
        List of all evaluated configurations with results
    """
    results = []
    best_score = -np.inf
    best_config = None
    
    print("\n" + "="*80)
    print(f"Random Search: {n_iter} iterations with {n_folds}-fold CV")
    print("="*80)
    
    for iteration in range(1, n_iter + 1):
        # Sample hyperparameters
        config = sample_hyperparameters(search_space)
        
        print(f"\n[Iteration {iteration}/{n_iter}]")
        print(f"  Config: {config}")
        
        start_time = time.time()
        
        # Evaluate configuration
        try:
            cv_results = cross_validate_config(
                data_dict, D, config, device, n_folds, seed
            )
            
            elapsed_time = time.time() - start_time
            
            # Store results
            result_entry = {
                'iteration': iteration,
                'config': config,
                'mean_R2': cv_results['mean_R2'],
                'Bs_R2': cv_results['Bs']['R2_mean'],
                'Hc_R2': cv_results['Hc']['R2_mean'],
                'Dc_R2': cv_results['Dc']['R2_mean'],
                'Bs_MAE': cv_results['Bs']['MAE_mean'],
                'Hc_MAE': cv_results['Hc']['MAE_mean'],
                'Dc_MAE': cv_results['Dc']['MAE_mean'],
                'time_seconds': elapsed_time,
                'status': 'success'
            }
            results.append(result_entry)
            
            # Check if this is the best configuration
            if cv_results['mean_R2'] > best_score:
                best_score = cv_results['mean_R2']
                best_config = config.copy()
                print(f"  ⭐ New best! Mean R² = {best_score:.4f}")
            
            print(f"  Results: Mean R² = {cv_results['mean_R2']:.4f}")
            print(f"    Bs: R²={cv_results['Bs']['R2_mean']:.4f}, MAE={cv_results['Bs']['MAE_mean']:.4f}")
            print(f"    Hc: R²={cv_results['Hc']['R2_mean']:.4f}, MAE={cv_results['Hc']['MAE_mean']:.4f}")
            print(f"    Dc: R²={cv_results['Dc']['R2_mean']:.4f}, MAE={cv_results['Dc']['MAE_mean']:.4f}")
            print(f"  Time: {elapsed_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            result_entry = {
                'iteration': iteration,
                'config': config,
                'status': 'failed',
                'error': str(e)
            }
            results.append(result_entry)
    
    print("\n" + "="*80)
    print("Random Search Complete!")
    print(f"Best configuration (Mean R² = {best_score:.4f}):")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    print("="*80)
    
    return results, best_config


def save_results(results, best_config, output_dir):
    """Save random search results to files.
    
    Args:
        results: List of all evaluated configurations
        best_config: Best hyperparameter configuration
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results as JSON
    json_path = os.path.join(output_dir, 'random_search_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_config
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Save successful runs as CSV for easy analysis
    successful_runs = [r for r in results if r.get('status') == 'success']
    if successful_runs:
        df = pd.DataFrame(successful_runs)
        
        # Expand config dict into separate columns
        config_df = pd.json_normalize(df['config'])
        config_df.columns = ['config_' + col for col in config_df.columns]
        
        # Combine with results
        df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
        
        csv_path = os.path.join(output_dir, 'random_search_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to {csv_path}")
    
    # Save best config separately
    best_path = os.path.join(output_dir, 'best_config.json')
    with open(best_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved to {best_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Random search hyperparameter optimization for MTWAE'
    )
    
    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    
    # Random search parameters
    ap.add_argument('--n_iter', type=int, default=30,
                   help='Number of random search iterations (30)')
    ap.add_argument('--n_folds', type=int, default=5,
                   help='Number of CV folds per configuration')
    ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--output_dir', default='random_search_results')
    
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
    
    # Define search space
    search_space = {
        'epochs': {
            'type': 'int',
            'low': 400,
            'high': 1200,
        },
        'batch_size': {
            'type': 'choice',
            'choices': [2, 4, 8, 16],
        },
        'lr': {
            'type': 'log',
            'low': 1e-4,
            'high': 1e-2,
        },
        'latent_dim': {
            'type': 'choice',
            'choices': [2, 4, 8, 16],
        },
        'sigma': {
            'type': 'float',
            'low': 4.0,
            'high': 12.0,
        },
        'lambda_mmd': {
            'type': 'log',
            'low': 1e-5,
            'high': 1e-3,
        },
    }
    
    print("\nSearch space:")
    for param, space in search_space.items():
        print(f"  {param}: {space}")
    
    # Run random search
    results, best_config = random_search(
        data_dict, D, search_space, args.n_iter, device, args.n_folds, args.seed
    )
    
    # Save results
    save_results(results, best_config, args.output_dir)
    
    print("\nRandom search complete! Check results directory for details.")


if __name__ == '__main__':
    main()