
"""
SHAP Analysis for Elemental Design Principles

Analyzes element contributions to properties using Shapley Additive Explanations:
- SHAP summary plots for experimental data (top 12 elements)
- SHAP dependence plots for individual elements
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap

from models import MTWAE
from data import load_text_data, PERIODIC_30
from utils import set_seed, ensure_dir


def compute_shap_values(model, property_name, X, device='cpu'):
    """
    Compute SHAP values for given model and data.
    
    Args:
        model: Trained MTWAE model
        property_name: 'Bs', 'Hc', or 'Dc'
        X: Composition features (n_samples, 30)
        device: Device for computation
    
    Returns:
        SHAP Explanation object
    """
    def model_predict(compositions):
        """Wrapper for SHAP explainer."""
        comp_tensor = torch.tensor(compositions, dtype=torch.float32).to(device)
        with torch.no_grad():
            z = model.encoder(comp_tensor)
            
            # Get the appropriate predictor
            if property_name == 'Bs':
                predictions = model.pred_Bs(z)
            elif property_name == 'Hc':
                predictions = model.pred_Hc(z)
            else:  # Dc
                predictions = model.pred_Dc(z)
                
        return predictions.cpu().numpy()
    
    # Create SHAP explainer with independent masker
    masker = shap.maskers.Independent(X)
    explainer = shap.Explainer(model_predict, masker)
    
    print("  Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer(X)
    
    return shap_values


def save_shap_summary_plot(shap_values, features, feature_names, property_name, 
                           output_path, top_k=12):
    """
    Save SHAP summary plot.
    
    Args:
        shap_values: SHAP explanation object
        features: Composition features
        feature_names: Element names
        property_name: 'Bs', 'Hc', or 'Dc'
        output_path: Save path
        top_k: Number of top elements to show
    """
    # Select top k elements by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values[:, top_indices],
        features[:, top_indices],
        feature_names=[feature_names[i] for i in top_indices],
        show=False
    )
    
    # Format plot
    plt.xlabel(f"SHAP value for {property_name}", fontsize=28, weight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Format colorbar
    fig = plt.gcf()
    cb_ax = fig.axes[-1]  # Colorbar axis
    cb_ax.set_ylabel("Feature value", fontsize=26, weight='bold')
    cb_ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def save_shap_dependence_plot(shap_values, features, element_name, element_idx,
                              property_name, output_path):
    """
    Save SHAP dependence plot for single element.
    
    Args:
        shap_values: SHAP explanation object
        features: Composition features
        element_name: Element symbol
        element_idx: Element index in features
        property_name: 'Bs', 'Hc', or 'Dc'
        output_path: Save path
    """
    # Property-specific y-axis labels
    ylabels = {
        'Bs': 'SHAP value (T)',
        'Hc': 'SHAP value (A/m)',
        'Dc': 'SHAP value (mm)'
    }
    ylabel = ylabels.get(property_name, 'SHAP value')
    
    # Filter non-zero content
    mask = features[:, element_idx] > 0
    if mask.sum() < 10:  # Skip if too few samples
        print(f"  Skipped {element_name} (insufficient data)")
        return
    
    xvals = features[mask, element_idx] * 100  # Convert to at%
    yvals = shap_values.values[mask, element_idx]
    
    # Color by SHAP sign (positive: red, negative: blue)
    colors = np.where(yvals > 0, 'red', 'blue')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xvals, yvals, c=colors, s=200, alpha=0.8, edgecolor='k')
    
    # Formatting
    ax.set_xlabel(f'{element_name} content (at.%)', fontsize=36, weight='bold')
    ax.set_ylabel(ylabel, fontsize=36, weight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description='SHAP analysis for elemental design principles')
    
    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    
    # Model
    ap.add_argument('--ckpt', default='checkpoints/mtwae_k8_final_inverse.pth',
                   help='Path to trained MTWAE model')
    ap.add_argument('--latent', type=int, default=8)
    
    # SHAP settings
    ap.add_argument('--top_k', type=int, default=12,
                   help='Number of top elements for summary plot (12)')
    ap.add_argument('--elements', nargs='+',
                   default=['Fe', 'B', 'Si', 'P', 'C', 'Co', 'Nb', 
                           'Ni', 'Mo', 'Zr', 'Ga', 'Al'],
                   help='Elements for dependence plots')
    
    # Optional: Generate random samples for validation
    ap.add_argument('--generate_samples', type=int, default=0,
                   help='Generate N random samples for SHAP (10000)')
    ap.add_argument('--sigma', type=float, default=8.0)
    
    # Output
    ap.add_argument('--outdir', default='shap_results')
    ap.add_argument('--seed', type=int, default=11)
    
    args = ap.parse_args()
    
    ensure_dir(args.outdir)
    set_seed(args.seed)
    
    print("="*80)
    print("SHAP Analysis for Elemental Design Principles")
    print("="*80)
    
    # Load data
    print("\nLoading experimental data...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )
    
    print(f"  Bs: {len(Xb)} samples")
    print(f"  Hc: {len(Xh)} samples")
    print(f"  Dc: {len(Xd)} samples")
    
    # Load model
    print(f"\nLoading model from {args.ckpt}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTWAE(in_features=D, latent_size=args.latent)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()
    
    # ========== Part 1: SHAP Summary Plots==========
    print("\n" + "="*80)
    print("Part 1: SHAP Summary Plots on Experimental Data")
    print("="*80)
    
    # Bs
    print("\nComputing SHAP for Bs...")
    shap_bs = compute_shap_values(model, 'Bs', Xb, device)
    save_shap_summary_plot(
        shap_bs, Xb, PERIODIC_30, 'Bs',
        os.path.join(args.outdir, 'shap_summary_Bs.png'),
        top_k=args.top_k
    )
    
    # Hc
    print("\nComputing SHAP for Hc...")
    shap_hc = compute_shap_values(model, 'Hc', Xh, device)
    save_shap_summary_plot(
        shap_hc, Xh, PERIODIC_30, 'Hc',
        os.path.join(args.outdir, 'shap_summary_Hc.png'),
        top_k=args.top_k
    )
    
    # Dc
    print("\nComputing SHAP for Dc...")
    shap_dc = compute_shap_values(model, 'Dc', Xd, device)
    save_shap_summary_plot(
        shap_dc, Xd, PERIODIC_30, 'Dc',
        os.path.join(args.outdir, 'shap_summary_Dc.png'),
        top_k=args.top_k
    )
    
    # ========== Part 2: SHAP Dependence Plots (Supplementary Figs S9-S11) ==========
    print("\n" + "="*80)
    print("Part 2: SHAP Dependence Plots")
    print("="*80)
    
    # Create subdirectories
    dep_dir = os.path.join(args.outdir, 'dependence_plots')
    ensure_dir(dep_dir)
    
    elem_to_idx = {e: i for i, e in enumerate(PERIODIC_30)}
    
    for element in args.elements:
        if element not in elem_to_idx:
            continue
        
        idx = elem_to_idx[element]
        print(f"\nGenerating dependence plots for {element}...")
        
        # Bs
        save_shap_dependence_plot(
            shap_bs, Xb, element, idx, 'Bs',
            os.path.join(dep_dir, f'shap_scatter_Bs_{element}.png')
        )
        
        # Hc
        save_shap_dependence_plot(
            shap_hc, Xh, element, idx, 'Hc',
            os.path.join(dep_dir, f'shap_scatter_Hc_{element}.png')
        )
        
        # Dc
        save_shap_dependence_plot(
            shap_dc, Xd, element, idx, 'Dc',
            os.path.join(dep_dir, f'shap_scatter_Dc_{element}.png')
        )
    
    # ========== Part 3: Optional - Generated Samples Validation ==========
    if args.generate_samples > 0:
        print("\n" + "="*80)
        print(f"Part 3: SHAP on {args.generate_samples} Generated Samples")
        print("="*80)
        
        print("\nGenerating random compositions...")
        with torch.no_grad():
            z_random = args.sigma * torch.randn(args.generate_samples, args.latent).to(device)
            X_gen = model.decode(z_random).cpu().numpy()
        
        # Apply threshold
        threshold = 0.005
        X_gen[X_gen < threshold] = 0
        
        print(f"  Generated {len(X_gen)} compositions")
        
        # Compute SHAP
        print("\nComputing SHAP for generated samples...")
        shap_bs_gen = compute_shap_values(model, 'Bs', X_gen, device)
        shap_hc_gen = compute_shap_values(model, 'Hc', X_gen, device)
        shap_dc_gen = compute_shap_values(model, 'Dc', X_gen, device)
        
        # Save summary plots
        gen_dir = os.path.join(args.outdir, 'generated_samples')
        ensure_dir(gen_dir)
        
        save_shap_summary_plot(
            shap_bs_gen, X_gen, PERIODIC_30, 'Bs',
            os.path.join(gen_dir, 'shap_summary_Bs_generated.png'),
            top_k=args.top_k
        )
        save_shap_summary_plot(
            shap_hc_gen, X_gen, PERIODIC_30, 'Hc',
            os.path.join(gen_dir, 'shap_summary_Hc_generated.png'),
            top_k=args.top_k
        )
        save_shap_summary_plot(
            shap_dc_gen, X_gen, PERIODIC_30, 'Dc',
            os.path.join(gen_dir, 'shap_summary_Dc_generated.png'),
            top_k=args.top_k
        )
        
        print(f"\n  Generated samples SHAP plots saved to {gen_dir}/")
    
    print(f"\n{'='*80}")
    print(f"SHAP analysis complete!")
    print(f"Results saved to {args.outdir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()