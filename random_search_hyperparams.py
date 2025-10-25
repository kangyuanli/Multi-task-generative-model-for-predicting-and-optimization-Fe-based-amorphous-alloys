"""
This corrected version properly implements:
1. Random 80-20 train-test split for each property dataset
2. Hyperparameter tuning using 5-fold cross-validation ONLY on training set
3. Final evaluation of best model on held-out test set
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from models import MTWAE
from data import load_text_data
from train import train_epoch, evaluate
from utils import set_seed, inverse_sample_size_weights


def sample_hyperparameters(search_space: Dict) -> Dict:
    """Sample hyperparameters from search space."""
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


def split_data(data_dict, test_size=0.2, random_state=8):
    """Split each task's data into training and testing sets."""
    train_dict = {}
    test_dict = {}
    
    for task in ['Bs', 'Hc', 'Dc']:
        X, y = data_dict[task]
        
        # Perform 80-20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        train_dict[task] = (X_train, y_train)
        test_dict[task] = (X_test, y_test)
        
        print(f"  {task}: Train={len(X_train)}, Test={len(X_test)} "
              f"(split ratio: {len(X_train)/(len(X_train)+len(X_test)):.2%})")
    
    return train_dict, test_dict


def cross_validate_config(train_dict, D, config, device, n_folds=5, seed=8):
    """Evaluate a hyperparameter configuration using k-fold CV on TRAINING SET ONLY."""
    # Compute task weights (using inverse sample size weighting)
    nB = len(train_dict['Bs'][0])
    nH = len(train_dict['Hc'][0])
    nD = len(train_dict['Dc'][0])
    task_weights = inverse_sample_size_weights(nB, nH, nD)
    
    # Prepare KFold for each task - ONLY on training data
    kfolds = {}
    for task in ['Bs', 'Hc', 'Dc']:
        X_train, y_train = train_dict[task]
        kfolds[task] = list(KFold(n_splits=n_folds, shuffle=True, 
                                  random_state=seed).split(X_train))
    
    # Store results for each fold
    results = {task: {'R2': [], 'MAE': []} for task in ['Bs', 'Hc', 'Dc']}
    
    for fold_idx in range(n_folds):
        # Prepare data splits for all tasks
        train_loaders = {}
        val_loaders = {}
        
        for task in ['Bs', 'Hc', 'Dc']:
            X_train, y_train = train_dict[task]
            train_idx, val_idx = kfolds[task][fold_idx]
            
            # Split training data into train/val for this fold
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
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
                          config.get('sigma', 10.0), 
                          config.get('lambda_mmd', 1e-4), 
                          task_weights)
        
        # Final evaluation on validation fold
        _, mae, r2 = evaluate(model, val_loaders, device, 
                            config.get('sigma', 10.0), 
                            config.get('lambda_mmd', 1e-4))
        
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


def evaluate_on_test_set(train_dict, test_dict, D, config, device):
    """Train model on entire training set and evaluate on test set."""
    # Compute task weights
    nB = len(train_dict['Bs'][0])
    nH = len(train_dict['Hc'][0])
    nD = len(train_dict['Dc'][0])
    task_weights = inverse_sample_size_weights(nB, nH, nD)
    
    # Prepare data loaders
    train_loaders = {}
    test_loaders = {}
    scalers = {}
    
    for task in ['Bs', 'Hc', 'Dc']:
        X_train, y_train = train_dict[task]
        X_test, y_test = test_dict[task]
        
        # Standardize targets using training set statistics
        sc = StandardScaler()
        y_train_s = sc.fit_transform(y_train)
        y_test_s = sc.transform(y_test)
        scalers[task] = sc
        
        # Create dataloaders
        train_loaders[task] = _to_loader(X_train, y_train_s, config['batch_size'], 
                                        shuffle=True, drop_last=True)
        test_loaders[task] = _to_loader(X_test, y_test_s, config['batch_size'], 
                                       shuffle=False, drop_last=False)
    
    # Initialize and train model on entire training set
    model = MTWAE(in_features=D, latent_size=config['latent_dim']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    print("\n  Training final model on entire training set...")
    for epoch in range(config['epochs']):
        loss = train_epoch(model, train_loaders, opt, device, 
                         config.get('sigma', 10.0), 
                         config.get('lambda_mmd', 1e-4), 
                         task_weights)
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{config['epochs']}: Loss={loss:.4f}")
    
    # Evaluate on test set
    test_loss, test_mae, test_r2 = evaluate(model, test_loaders, device, 
                                           config.get('sigma', 10.0), 
                                           config.get('lambda_mmd', 1e-4))
    
    # Prepare results
    results = {
        'test_loss': float(test_loss),
        'mean_R2': float(np.mean([test_r2[task] for task in ['Bs', 'Hc', 'Dc']])),
        'tasks': {}
    }
    
    for task in ['Bs', 'Hc', 'Dc']:
        results['tasks'][task] = {
            'R2': float(test_r2[task]),
            'MAE': float(test_mae[task])
        }
    
    return results


def random_search(train_dict, test_dict, D, search_space, n_iter, device, n_folds=5, seed=8):
    """
    Perform random search over hyperparameter space.
    Cross-validation is performed ONLY on training set.
    """
    results = []
    best_score = -np.inf
    best_config = None
    
    print("\n" + "="*80)
    print(f"Random Search: {n_iter} iterations with {n_folds}-fold CV on TRAINING SET")
    print("="*80)
    
    for iteration in range(1, n_iter + 1):
        # Sample hyperparameters
        config = sample_hyperparameters(search_space)
        
        print(f"\n[Iteration {iteration}/{n_iter}]")
        print(f"  Config: {config}")
        
        start_time = time.time()
        
        # Evaluate configuration using CV on training set only
        try:
            cv_results = cross_validate_config(
                train_dict, D, config, device, n_folds, seed
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
            
            print(f"  Results: Mean R² = {cv_results['mean_R2']:.4f}")
            print(f"    Bs: R²={cv_results['Bs']['R2_mean']:.4f}, MAE={cv_results['Bs']['MAE_mean']:.4f}")
            print(f"    Hc: R²={cv_results['Hc']['R2_mean']:.4f}, MAE={cv_results['Hc']['MAE_mean']:.4f}")
            print(f"    Dc: R²={cv_results['Dc']['R2_mean']:.4f}, MAE={cv_results['Dc']['MAE_mean']:.4f}")
            print(f"  Time: {elapsed_time:.1f}s")
            
        except Exception as e:

            result_entry = {
                'iteration': iteration,
                'config': config,
                'status': 'failed',
                'error': str(e)
            }
            results.append(result_entry)
    
    print(f"Best configuration from CV (Mean R² = {best_score:.4f}):")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    
    # Evaluate best model on test set

    test_results = evaluate_on_test_set(train_dict, test_dict, D, best_config, device)
    
    print(f"\nTEST SET RESULTS:")
    print(f"  Mean R² across tasks: {test_results['mean_R2']:.4f}")
    for task in ['Bs', 'Hc', 'Dc']:
        print(f"  {task}: R²={test_results['tasks'][task]['R2']:.4f}, "
              f"MAE={test_results['tasks'][task]['MAE']:.4f}")
    print("="*80)
    
    return results, best_config, test_results


def save_results(results, best_config, test_results, output_dir):
    """Save random search results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results as JSON
    json_path = os.path.join(output_dir, 'random_search_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_config,
            'test_set_performance': test_results
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
    
    # Save best config and test results separately
    best_path = os.path.join(output_dir, 'best_config.json')
    with open(best_path, 'w') as f:
        json.dump({
            'best_config': best_config,
            'cv_score': max([r['mean_R2'] for r in successful_runs]),
            'test_performance': test_results
        }, f, indent=2)
    print(f"Best config saved to {best_path}")


def main():
    ap = argparse.ArgumentParser(
        description='Random search hyperparameter optimization for MTWAE with proper train-test split'
    )
    
    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    
    # Random search parameters
    ap.add_argument('--n_iter', type=int, default=30,
                   help='Number of random search iterations (default: 30)')
    ap.add_argument('--n_folds', type=int, default=5,
                   help='Number of CV folds per configuration (default: 5)')
    ap.add_argument('--test_size', type=float, default=0.2,
                   help='Proportion of data for testing (default: 0.2)')
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
    print(f"Full dataset sizes - Bs: {len(Xb)}, Hc: {len(Xh)}, Dc: {len(Xd)}")
    
    data_dict = {
        'Bs': (Xb, yb),
        'Hc': (Xh, yh),
        'Dc': (Xd, yd)
    }
    
    # CRITICAL: Perform 80-20 train-test split
    print(f"\nPerforming {100*(1-args.test_size):.0f}-{100*args.test_size:.0f} train-test split...")
    train_dict, test_dict = split_data(data_dict, test_size=args.test_size, random_state=args.seed)
    
    # Define search space
    search_space = {
        'epochs': {
            'type': 'choice',
            'choices': [400, 800, 1600],            
        },
        'batch_size': {
            'type': 'choice',
            'choices': [4, 8, 16, 32],
        },
        'lr': {
            'type': 'choice',
            'choices': [1e-4, 1e-3, 1e-2],
        },
        'latent_dim': {
            'type': 'choice',
            'choices': [2, 4, 8, 16],
        },
        # Uncomment to include these in search
        #'sigma': {
        #    'type': 'float',
        #    'low': 8.0,
        #    'high': 12.0,
        #},
        #'lambda_mmd': {
        #    'type': 'log',
        #    'low': 1e-5,
        #    'high': 1e-3,
        #},
    }
    
    for param, space in search_space.items():
        print(f"  {param}: {space}")
    
    # Run random search (CV on training set only)
    results, best_config, test_results = random_search(
        train_dict, test_dict, D, search_space, args.n_iter, device, args.n_folds, args.seed
    )
    
    # Save results
    save_results(results, best_config, test_results, args.output_dir)


if __name__ == '__main__':
    main()