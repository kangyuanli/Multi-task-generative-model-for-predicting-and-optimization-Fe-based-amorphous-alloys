
"""
Multi-objective Inverse Design using NSGA-III

NSMTWAE Framework:
- Uses trained MTWAE model with NSGA-III algorithm
- Searches latent space for Pareto optimal compositions
- Optimizes trade-offs: maximize Bs & Dc, minimize ln(Hc)
- Target criteria: Bs>1.5T, ln(Hc)<1.5, Dc>1mm
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
import torch
import geatpy as ea

from models import MTWAE
from data import load_text_data
from utils import set_seed, ensure_dir


def composition_to_string(composition, element_table, threshold=0.001):
    """
    Convert composition vector to chemical formula string.
    
    Args:
        composition: Composition vector (simplex)
        element_table: List of element symbols
        threshold: Minimum fraction to include (default: 0.001)
    
    Returns:
        String like "Fe77.13B13.45Si7.53..."
    """
    composition_string = ''
    for i, frac in enumerate(composition):
        if frac > threshold:
            percentage = round(frac * 100, 2)
            composition_string += f'{element_table[i]}{percentage:.2f}'
    return composition_string


def draw_ellipse_background(ax, x_values, y_values, color, scale=2.2):
    """
    Draw ellipse background around scatter points for visualization.
    
    Args:
        ax: Matplotlib axis
        x_values, y_values: Data coordinates
        color: Ellipse color
        scale: Size scaling factor (default: 2.2)
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    x_std = np.std(x_values)
    y_std = np.std(y_values)
    
    ellipse = Ellipse(
        (x_mean, y_mean), 
        width=2*x_std*scale, 
        height=2*y_std*scale, 
        color=color, 
        alpha=0.3
    )
    ax.add_patch(ellipse)


class MultiObjectiveProblem(ea.Problem):
    """
    Multi-objective optimization problem for NSGA-III.
    
    Objectives:
        f1 = -Bs (maximize saturation magnetization)
        f2 = ln(Hc) (minimize coercivity)
        f3 = -Dc (maximize critical diameter)
    """
    
    def __init__(self, model, scalers, latent_dim=8, n_samples=3000, sigma=8.0):
        """
        Initialize optimization problem.
        
        Args:
            model: Trained MTWAE model
            scalers: Dict of StandardScalers for inverse transform
            latent_dim: Latent space dimension (default: 8)
            n_samples: Samples for determining bounds (default: 3000)
            sigma: Prior standard deviation (default: 8.0)
        """
        self.model = model
        self.scalers = scalers
        self.sigma = sigma
        
        name = 'MTWAE_MultiObj'
        M = 3  # Number of objectives
        maxormins = [-1, 1, -1]  # Maximize Bs(-), minimize Hc(+), maximize Dc(-)
        Dim = latent_dim
        varTypes = [0] * Dim  # Continuous variables
        
        # Determine bounds from prior sampling
        prior_samples = sigma * torch.randn(n_samples, latent_dim)
        lb = prior_samples.min(dim=0)[0].tolist()
        ub = prior_samples.max(dim=0)[0].tolist()
        
        lbin = [1] * Dim  # All variables have lower bounds
        ubin = [1] * Dim  # All variables have upper bounds
        
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        
        print(f"Initialized {name}: M={M}, Dim={Dim}, bounds=[{lb[0]:.2f}, {ub[0]:.2f}]")
    
    def evalVars(self, Vars):
        """
        Evaluate objective functions for given latent variables.
        
        Args:
            Vars: Latent space variables (batch, latent_dim)
        
        Returns:
            Objective values (batch, 3): [Bs, ln(Hc), Dc]
        """
        Z_tensor = torch.from_numpy(Vars).float()
        
        with torch.no_grad():
            # Predict properties (standardized)
            Bs_pred, Hc_pred, Dc_pred = self.model.predict(Z_tensor)
            
            # Inverse standardization to physical scale
            Bs_pred = self.scalers['Bs'].inverse_transform(Bs_pred.numpy())
            Hc_pred = self.scalers['Hc'].inverse_transform(Hc_pred.numpy())
            Dc_pred = self.scalers['Dc'].inverse_transform(Dc_pred.numpy())
        
        # Stack objectives: minimize -Bs, ln(Hc), -Dc
        f = np.hstack([Bs_pred, Hc_pred, Dc_pred])
        
        return f


def plot_2d_pareto_comparisons(obj_values, experimental_data, selected_examples, 
                                output_dir, bs_thresh=1.5, lnhc_thresh=1.5, dc_thresh=1.0):
    """
    Plot 2D Pareto front comparisons with experimental data.
    
    Args:
        obj_values: Optimized objective values
        experimental_data: Dict with 'Bs', 'Hc', 'Dc' arrays
        selected_examples: Dict with example compositions
        output_dir: Output directory for figures
        bs_thresh, lnhc_thresh, dc_thresh: Target thresholds
    """
    fig_configs = [
        ('Bs', 'ln(Hc)', 0, 1, [0, 2.5], [-3, 4], True),
        ('Bs', 'Dc', 0, 2, [0, 2.5], [0, 8], False),
        ('Dc', 'ln(Hc)', 2, 1, [0, 8], [-2, 4], True)
    ]
    
    for idx, (xlabel, ylabel, x_idx, y_idx, xlim, ylim, invert_y) in enumerate(fig_configs):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Ellipse backgrounds
        draw_ellipse_background(ax, obj_values[:, x_idx], obj_values[:, y_idx], 
                                color='lightcoral')
        draw_ellipse_background(ax, experimental_data[xlabel.split('(')[0]], 
                                experimental_data[ylabel.split('(')[0]], 
                                color='lightblue')
        
        # Scatter plots
        ax.scatter(obj_values[:, x_idx], obj_values[:, y_idx], 
                  c='red', marker='o', alpha=0.8, edgecolor='k', 
                  linewidth=1.5, s=100, label='Pareto Front')
        ax.scatter(experimental_data[xlabel.split('(')[0]], 
                  experimental_data[ylabel.split('(')[0]], 
                  c='blue', marker='o', alpha=0.8, edgecolor='k', 
                  linewidth=1.5, s=100, label='Experimental Data')
        
        # Selected examples
        if selected_examples:
            ax.scatter(selected_examples[xlabel.split('(')[0]], 
                      selected_examples[ylabel.split('(')[0]], 
                      c='yellow', marker='o', alpha=1.0, edgecolor='k', 
                      linewidth=1.5, s=150, label='Selected Examples')
        
        # Threshold lines
        if 'Bs' in xlabel:
            ax.axvline(x=bs_thresh, color='black', linestyle='--', linewidth=2)
        if 'Dc' in xlabel:
            ax.axvline(x=dc_thresh, color='black', linestyle='--', linewidth=2)
        if 'ln(Hc)' in ylabel:
            ax.axhline(y=lnhc_thresh, color='black', linestyle='--', linewidth=2)
        if 'Dc' in ylabel:
            ax.axhline(y=dc_thresh, color='black', linestyle='--', linewidth=2)
        
        # Formatting
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if invert_y:
            ax.invert_yaxis()
        
        ax.tick_params(axis='x', labelsize=18, width=2)
        ax.tick_params(axis='y', labelsize=18, width=2)
        ax.set_xlabel(f'{xlabel} ({"T" if "Bs" in xlabel else "mm"})', 
                     fontsize=22, weight='bold')
        ax.set_ylabel(f'{ylabel} ({"A/m" if "Hc" in ylabel else "mm"})', 
                     fontsize=22, weight='bold')
        
        ax.legend(loc='upper right', frameon=True, prop={'weight': 'bold', 'size': 12})
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        fig.savefig(os.path.join(output_dir, f'pareto_comparison_{idx+1}.png'), 
                   dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved pareto_comparison_{idx+1}.png")


def plot_3d_pareto_front(obj_values, output_dir, 
                         bs_thresh=1.5, lnhc_thresh=1.5, dc_thresh=1.0):
    """
    Plot 3D Pareto front with projections.
    
    Args:
        obj_values: Objective values (N, 3): [Bs, ln(Hc), Dc]
        output_dir: Output directory
        bs_thresh, lnhc_thresh, dc_thresh: Target thresholds
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = obj_values[:, 0], obj_values[:, 1], obj_values[:, 2]
    
    # Separate target and non-target solutions
    target_condition = (x > bs_thresh) & (y < lnhc_thresh) & (z > dc_thresh)
    
    # Plot non-target solutions
    ax.scatter(x[~target_condition], y[~target_condition], z[~target_condition], 
              c='tab:gray', marker='^', s=60, alpha=0.6, label='Other MGs')
    
    # Plot target solutions
    ax.scatter(x[target_condition], y[target_condition], z[target_condition], 
              c='tab:red', marker='o', s=80, edgecolor='k', 
              depthshade=True, label='Target MGs')
    
    # Plot projections
    ax.scatter(x[target_condition], y[target_condition], 
              np.full_like(z[target_condition], z.min()), 
              c='tab:blue', marker='o', s=40, alpha=0.5)
    ax.scatter(x[target_condition], np.full_like(y[target_condition], y.max()), 
              z[target_condition], 
              c='tab:green', marker='o', s=40, alpha=0.5)
    ax.scatter(np.full_like(x[target_condition], x.min()), 
              y[target_condition], z[target_condition], 
              c='tab:orange', marker='o', s=40, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Bs (T)', fontsize=18, weight='bold', labelpad=15)
    ax.set_ylabel('ln(Hc) (A/m)', fontsize=18, weight='bold', labelpad=15)
    ax.set_zlabel('Dc (mm)', fontsize=18, weight='bold', labelpad=15)
    
    ax.tick_params(axis='x', labelsize=14, width=1.5)
    ax.tick_params(axis='y', labelsize=14, width=1.5)
    ax.tick_params(axis='z', labelsize=14, width=1.5)
    
    ax.view_init(elev=20, azim=-35)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=14, frameon=True)
    
    plt.savefig(os.path.join(output_dir, 'pareto_3d.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved pareto_3d.png")


def main():
    ap = argparse.ArgumentParser(
        description='Multi-objective inverse design using NSGA-III')
    
    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    
    # Model settings
    ap.add_argument('--ckpt', type=str, 
                   default='checkpoints/mtwae_k8_final_inverse.pth',
                   help='Path to trained MTWAE model')
    ap.add_argument('--latent', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=8.0)
    
    # NSGA-III settings
    ap.add_argument('--nind', type=int, default=200,
                   help='Population size (200)')
    ap.add_argument('--maxgen', type=int, default=500,
                   help='Maximum generations (500)')
    ap.add_argument('--seed', type=int, default=1)
    
    # Output
    ap.add_argument('--outdir', default='nsga3_results')
    
    args = ap.parse_args()
    
    ensure_dir(args.outdir)
    set_seed(args.seed)
    
    print("="*80)
    print("Multi-objective Inverse Design using NSGA-III")
    print("="*80)
    
    # Load data and prepare scalers
    print("\nLoading data...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Split and create scalers (consistent with training)
    _, _, yb_tr, _ = train_test_split(Xb, yb, test_size=0.2, random_state=8)
    _, _, yh_tr, _ = train_test_split(Xh, yh, test_size=0.2, random_state=8)
    _, _, yd_tr, _ = train_test_split(Xd, yd, test_size=0.2, random_state=8)
    
    scalers = {}
    scalers['Bs'] = StandardScaler().fit(yb_tr)
    scalers['Hc'] = StandardScaler().fit(yh_tr)
    scalers['Dc'] = StandardScaler().fit(yd_tr)
    
    # Prepare experimental data for plotting
    exp_data = {
        'Bs': yb.ravel(),
        'ln(Hc)': yh.ravel(),
        'Dc': yd.ravel()
    }
    
    # Load trained model
    print(f"Loading model from {args.ckpt}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTWAE(in_features=D, latent_size=args.latent)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()
    
    # Define optimization problem
    print("\nSetting up NSGA-III optimization...")
    problem = MultiObjectiveProblem(model, scalers, args.latent, sigma=args.sigma)
    
    # Create NSGA-III algorithm
    algorithm = ea.moea_NSGA3_templet(
        problem,
        ea.Population(Encoding='RI', NIND=args.nind),
        MAXGEN=args.maxgen,
        logTras=1,
        seed=args.seed
    )
    
    # Run optimization
    print(f"\nRunning NSGA-III: {args.nind} individuals, {args.maxgen} generations...")
    print("This may take several minutes...")
    
    res = ea.optimize(algorithm, verbose=False, drawLog=True, outputMsg=True, 
                     saveFlag=True, dirName=os.path.join(args.outdir, 'evolution'))
    
    print("\nOptimization complete!")
    
    # Extract results
    obj_values = res['ObjV']
    latent_vars = res['Vars']
    
    # Perform non-dominated sorting
    NDSet = ea.ndsortDED(obj_values)[0]
    
    # Extract Pareto fronts
    first_pareto = obj_values[NDSet == 1]
    second_pareto = obj_values[NDSet == 2]
    other_pareto = obj_values[NDSet > 2]
    
    first_vars = latent_vars[NDSet == 1]
    second_vars = latent_vars[NDSet == 2]
    other_vars = latent_vars[NDSet > 2]
    
    print(f"\nPareto fronts:")
    print(f"  First front: {len(first_pareto)} solutions")
    print(f"  Second front: {len(second_pareto)} solutions")
    print(f"  Other fronts: {len(other_pareto)} solutions")
    
    # Decode compositions
    print("\nDecoding compositions...")
    from data import PERIODIC_30
    
    with torch.no_grad():
        first_comps = model.decode(torch.from_numpy(first_vars).float().to(device)).cpu().numpy()
        second_comps = model.decode(torch.from_numpy(second_vars).float().to(device)).cpu().numpy()
        other_comps = model.decode(torch.from_numpy(other_vars).float().to(device)).cpu().numpy()
    
    # Apply threshold and convert to strings
    threshold = 0.001
    first_comps[first_comps < threshold] = 0
    second_comps[second_comps < threshold] = 0
    other_comps[other_comps < threshold] = 0
    
    first_comp_strs = [composition_to_string(c, PERIODIC_30) for c in first_comps]
    second_comp_strs = [composition_to_string(c, PERIODIC_30) for c in second_comps]
    other_comp_strs = [composition_to_string(c, PERIODIC_30) for c in other_comps]
    
    # Save results to CSV
    print("\nSaving results...")
    
    # First Pareto front
    df_first = pd.DataFrame({
        'Composition': first_comp_strs,
        'Bs (T)': first_pareto[:, 0],
        'ln(Hc) (A/m)': first_pareto[:, 1],
        'Dc (mm)': first_pareto[:, 2]
    })
    
    # Filter target compositions
    df_first_target = df_first[
        (df_first['Bs (T)'] > 1.5) & 
        (df_first['ln(Hc) (A/m)'] < 1.5) & 
        (df_first['Dc (mm)'] > 1.0)
    ]
    
    df_first.to_csv(os.path.join(args.outdir, 'first_pareto_all.csv'), index=False)
    df_first_target.to_csv(os.path.join(args.outdir, 'first_pareto_target.csv'), index=False)
    
    # Second Pareto front
    df_second = pd.DataFrame({
        'Composition': second_comp_strs,
        'Bs (T)': second_pareto[:, 0],
        'ln(Hc) (A/m)': second_pareto[:, 1],
        'Dc (mm)': second_pareto[:, 2]
    })
    df_second.to_csv(os.path.join(args.outdir, 'second_pareto_all.csv'), index=False)
    
    # Other fronts
    df_other = pd.DataFrame({
        'Composition': other_comp_strs,
        'Bs (T)': other_pareto[:, 0],
        'ln(Hc) (A/m)': other_pareto[:, 1],
        'Dc (mm)': other_pareto[:, 2]
    })
    df_other.to_csv(os.path.join(args.outdir, 'other_pareto.csv'), index=False)
    
    print(f"  Saved CSV files to {args.outdir}/")
    print(f"  Target compositions (1st front): {len(df_first_target)}")
    
    # Plot results
    print("\nGenerating visualizations...")
    
    # Selected examples
    selected = {
        'Bs': [1.596, 1.716, 1.909, 1.808, 1.710],
        'ln(Hc)': [-0.838, 0.371, 0.029, 0.598, 2.068],
        'Dc': [1.466, 2.62, 1.213, 2.026, 4.904]
    }
    
    plot_2d_pareto_comparisons(obj_values, exp_data, selected, args.outdir)
    plot_3d_pareto_front(obj_values, args.outdir)
    
    print(f"\n{'='*80}")
    print(f"Inverse design complete!")
    print(f"Results saved to {args.outdir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()