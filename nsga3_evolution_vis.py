
"""
Evolutionary Process Visualization

Tracks how composition population evolves across generations:
- Generation 0: Random initialization
- Generation 10: Initial convergence
- Generation 200: Refined Pareto front

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import torch
import geatpy as ea
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

from models import MTWAE
from data import load_text_data, PERIODIC_30
from utils import set_seed, ensure_dir


def draw_ellipse(ax, x_values, y_values, color, scale=2.2):
    """Draw ellipse background for visualization."""
    x_mean, y_mean = np.mean(x_values), np.mean(y_values)
    x_std, y_std = np.std(x_values), np.std(y_values)
    
    ellipse = Ellipse(
        (x_mean, y_mean),
        width=2*x_std*scale,
        height=2*y_std*scale,
        color=color,
        alpha=0.3
    )
    ax.add_patch(ellipse)


class MultiObjectiveProblem(ea.Problem):
    """Multi-objective optimization problem for evolution tracking."""
    
    def __init__(self, model, scalers, latent_dim=8, n_samples=3000, sigma=8.0):
        self.model = model
        self.scalers = scalers
        self.sigma = sigma
        
        name = 'MTWAE_Evolution'
        M = 3
        maxormins = [-1, 1, -1]
        Dim = latent_dim
        varTypes = [0] * Dim
        
        # Generate prior samples for bounds
        self.prior_samples = sigma * torch.randn(n_samples, latent_dim)
        lb = self.prior_samples.min(dim=0)[0].tolist()
        ub = self.prior_samples.max(dim=0)[0].tolist()
        
        lbin, ubin = [1] * Dim, [1] * Dim
        
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def evalVars(self, Vars):
        Z_tensor = torch.from_numpy(Vars).float()
        
        with torch.no_grad():
            Bs_pred, Hc_pred, Dc_pred = self.model.predict(Z_tensor)
            
            Bs_pred = self.scalers['Bs'].inverse_transform(Bs_pred.numpy())
            Hc_pred = self.scalers['Hc'].inverse_transform(Hc_pred.numpy())
            Dc_pred = self.scalers['Dc'].inverse_transform(Dc_pred.numpy())
        
        return np.hstack([Bs_pred, Hc_pred, Dc_pred])


def visualize_tsne_evolution(generations_data, output_dir):
    """
    Visualize t-SNE embeddings for different generations.
    
    Args:
        generations_data: List of (gen_num, compositions, pareto_comps)
        output_dir: Output directory
    """
    for gen_num, random_comps, pareto_comps in generations_data:
        # Compute t-SNE
        combined = np.vstack([random_comps, pareto_comps])
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        combined_2d = tsne.fit_transform(combined)
        
        random_2d = combined_2d[:len(random_comps)]
        pareto_2d = combined_2d[len(random_comps):]
        
        # Compute density for heatmap
        xy = np.vstack([random_2d[:, 0], random_2d[:, 1]])
        z = gaussian_kde(xy)(xy)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Random compositions with density heatmap
        sc = ax.scatter(random_2d[:, 0], random_2d[:, 1], 
                       c=z*1e4, s=10, cmap='Blues', 
                       label='Randomly Generated', alpha=0.6)
        
        # Pareto solutions
        ax.scatter(pareto_2d[:, 0], pareto_2d[:, 1], 
                  c='yellow', marker='o', s=50, 
                  edgecolor='k', linewidth=1, 
                  label='Pareto Solutions')
        
        # Colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Density (×10⁴)', fontsize=18, weight='bold')
        cbar.ax.tick_params(labelsize=16, width=2)
        
        # Formatting
        ax.set_xlim([-90, 90])
        ax.set_ylim([-90, 90])
        ax.tick_params(axis='both', labelsize=18, width=2)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=24, weight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=24, weight='bold')
        ax.legend(loc='upper right', frameon=True, 
                 prop={'weight': 'bold', 'size': 12})
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_gen_{gen_num}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved tsne_gen_{gen_num}.png")


def plot_property_evolution(generations_objectives, output_dir):
    """
    Plot property evolution across generations.
    
    Args:
        generations_objectives: List of (gen_num, obj_values)
        output_dir: Output directory
    """
    generations = [g[0] for g in generations_objectives]
    colors = sns.color_palette("Set1", n_colors=len(generations))
    
    fig_configs = [
        ('Bs', 'ln(Hc)', 0, 1, [0, 2.5], [-2, 4], True),
        ('Bs', 'Dc', 0, 2, [0, 2.5], [0, 8], False),
        ('Dc', 'ln(Hc)', 2, 1, [0, 8], [-2, 4], True)
    ]
    
    for idx, (xlabel, ylabel, x_idx, y_idx, xlim, ylim, invert_y) in enumerate(fig_configs):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, (gen_num, obj_vals) in enumerate(generations_objectives):
            color = colors[i]
            label = f'Generation {gen_num}'
            
            x = obj_vals[:, x_idx]
            y = obj_vals[:, y_idx]
            
            draw_ellipse(ax, x, y, color=color)
            ax.scatter(x, y, c=[color], marker='o', alpha=0.6, 
                      edgecolor='k', linewidth=1.0, s=50, label=label)
        
        # Threshold lines
        if 'Bs' in xlabel:
            ax.axvline(x=1.5, color='black', linestyle='--', linewidth=2)
        if 'Dc' in xlabel:
            ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2)
        if 'ln(Hc)' in ylabel:
            ax.axhline(y=1.5, color='black', linestyle='--', linewidth=2)
        if 'Dc' in ylabel:
            ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
        
        # Formatting
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if invert_y:
            ax.invert_yaxis()
        
        ax.tick_params(axis='x', labelsize=18, width=2)
        ax.tick_params(axis='y', labelsize=18, width=2)
        ax.set_xlabel(f'{xlabel} ({"T" if "Bs" in xlabel else "mm"})', 
                     fontsize=24, weight='bold')
        ax.set_ylabel(f'{ylabel} ({"A/m" if "Hc" in ylabel else "mm"})', 
                     fontsize=24, weight='bold')
        
        ax.legend(loc='upper right', frameon=True, 
                 prop={'weight': 'bold', 'size': 12})
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'evolution_properties_{idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved evolution_properties_{idx+1}.png")


def main():
    ap = argparse.ArgumentParser(
        description='Visualize NSGA-III evolutionary process')
    
    # Data and model
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    ap.add_argument('--ckpt', default='checkpoints/mtwae_k8_final_inverse.pth')
    ap.add_argument('--latent', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=8.0)
    
    # Evolution settings
    ap.add_argument('--nind', type=int, default=200)
    ap.add_argument('--generations', type=int, nargs='+', default=[0, 10, 200],
                   help='Generations to evaluate (0, 10, 200)')
    ap.add_argument('--seed', type=int, default=1)
    
    # Output
    ap.add_argument('--outdir', default='evolution_results')
    
    args = ap.parse_args()
    
    ensure_dir(args.outdir)
    set_seed(args.seed)
    
    print("="*80)
    print("NSGA-III Evolutionary Process Visualization")
    print("="*80)
    
    # Load data and prepare
    print("\nLoading data and model...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    _, _, yb_tr, _ = train_test_split(Xb, yb, test_size=0.2, random_state=8)
    _, _, yh_tr, _ = train_test_split(Xh, yh, test_size=0.2, random_state=8)
    _, _, yd_tr, _ = train_test_split(Xd, yd, test_size=0.2, random_state=8)
    
    scalers = {
        'Bs': StandardScaler().fit(yb_tr),
        'Hc': StandardScaler().fit(yh_tr),
        'Dc': StandardScaler().fit(yd_tr)
    }
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTWAE(in_features=D, latent_size=args.latent)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()
    
    # Setup problem
    problem = MultiObjectiveProblem(model, scalers, args.latent, sigma=args.sigma)
    
    # Track evolution
    print(f"\nTracking evolution at generations: {args.generations}")
    
    generations_data_tsne = []
    generations_data_props = []
    
    for max_gen in args.generations:
        print(f"\n  Running generation {max_gen}...")
        
        algorithm = ea.moea_NSGA3_templet(
            problem,
            ea.Population(Encoding='RI', NIND=args.nind),
            MAXGEN=max_gen,
            logTras=0,
            seed=args.seed
        )
        
        res = ea.optimize(algorithm, verbose=False, drawLog=False, 
                         outputMsg=False, saveFlag=False)
        
        obj_values = res['ObjV']
        latent_vars = res['Vars']
        
        # Decode compositions
        with torch.no_grad():
            pareto_comps = model.decode(
                torch.from_numpy(latent_vars).float().to(device)
            ).cpu().numpy()
            
            random_comps = model.decode(
                problem.prior_samples.to(device)
            ).cpu().numpy()
        
        generations_data_tsne.append((max_gen, random_comps, pareto_comps))
        generations_data_props.append((max_gen, obj_values))
        
        print(f"    Generated {len(obj_values)} solutions")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_tsne_evolution(generations_data_tsne, args.outdir)
    plot_property_evolution(generations_data_props, args.outdir)
    
    print(f"\n{'='*80}")
    print(f"Evolution visualization complete!")
    print(f"Results saved to {args.outdir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()