
"""
External Validation and High-Throughput Discovery

Part 1: External Validation
- Validates MTWAE predictions on 12 recently published Fe-based alloys
- Compares predicted vs experimental Bs, ln(Hc), Dc values
- Demonstrates model generalization capability

Part 2: High-Throughput Discovery
- Visualizes NSGA-III candidate alloys (Bs>1.75T) vs:
  * Dataset alloys (gray squares)
  * Validated alloys (blue triangles)
- Shows property space coverage

Key findings:
- MAE ≈ 0.04-0.10T for Bs prediction on external alloys
- NSMTWAE framework identifies 95 Pareto optimal compositions
- Candidates extend performance boundaries beyond experimental data
"""

import os
import argparse
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from models import MTWAE
from data import load_text_data, PERIODIC_30
from utils import set_seed, ensure_dir


def parse_composition_string(comp_str):
    """
    Parse composition string like "Fe68.2Co17.5B13Si0.5Cu0.8" into elements and fractions.
    
    Args:
        comp_str: Composition string
    
    Returns:
        Tuple of (elements, fractions)
        e.g., (['Fe','Co','B','Si','Cu'], [68.2, 17.5, 13, 0.5, 0.8])
    """
    pattern = r"([A-Z][a-z]?)([\d\.]+)"
    matches = re.findall(pattern, comp_str)
    
    elements = [m[0] for m in matches]
    fractions = [float(m[1]) for m in matches]
    
    return elements, fractions


def composition_strings_to_matrix(comp_strings, periodic_table):
    """
    Convert list of composition strings to composition matrix.
    
    Args:
        comp_strings: List of composition strings
        periodic_table: Element periodic table
    
    Returns:
        Composition matrix (n_samples, n_elements)
    """
    n_samples = len(comp_strings)
    n_elements = len(periodic_table)
    comp_matrix = np.zeros((n_samples, n_elements), dtype=np.float32)
    
    elem_to_idx = {e: i for i, e in enumerate(periodic_table)}
    
    for i, comp_str in enumerate(comp_strings):
        elements, fractions = parse_composition_string(comp_str)
        for elem, frac in zip(elements, fractions):
            if elem in elem_to_idx:
                comp_matrix[i, elem_to_idx[elem]] = frac * 0.01  # Convert to [0,1]
    
    return comp_matrix


def predict_external_compositions(model, comp_strings, scalers, periodic_table, device='cpu'):
    """
    Predict properties for external compositions using trained MTWAE.
    
    Args:
        model: Trained MTWAE model
        comp_strings: List of composition strings
        scalers: Dict of StandardScalers for inverse transform
        periodic_table: Element periodic table
        device: Device for inference
    
    Returns:
        DataFrame with predictions: [Composition, Bs_pred, ln(Hc)_pred, Dc_pred]
    """
    # Convert compositions to matrix
    comp_matrix = composition_strings_to_matrix(comp_strings, periodic_table)
    
    # Predict
    X_tensor = torch.from_numpy(comp_matrix).float().to(device)
    model.eval()
    
    with torch.no_grad():
        Z = model.encode(X_tensor) 
        Bs_pred, Hc_pred, Dc_pred = model.predict(Z)
    
    # Inverse standardization
    Bs_pred = scalers['Bs'].inverse_transform(Bs_pred.cpu().numpy()).ravel()
    lnHc_pred = scalers['Hc'].inverse_transform(Hc_pred.cpu().numpy()).ravel()
    Dc_pred = scalers['Dc'].inverse_transform(Dc_pred.cpu().numpy()).ravel()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Composition': comp_strings,
        'Bs_pred (T)': Bs_pred,
        'ln(Hc)_pred (A/m)': lnHc_pred,
        'Dc_pred (mm)': Dc_pred
    })
    
    return results


def plot_validation_bar_chart(predictions, experimental, output_path):
    """
    Plot bar chart comparing predicted vs experimental Bs values.
    
    Args:
        predictions: DataFrame with predictions
        experimental: DataFrame with experimental values
        output_path: Save path
    """
    n = len(predictions)
    x = np.arange(n)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, experimental['Bs_exp (T)'], width, 
                   label='Experimental', color='steelblue', edgecolor='k', linewidth=1)
    bars2 = ax.bar(x + width/2, predictions['Bs_pred (T)'], width,
                   label='Predicted', color='coral', edgecolor='k', linewidth=1)
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions['Bs_pred (T)'] - experimental['Bs_exp (T)']))
    
    # Formatting
    ax.set_xlabel('Alloy Compositions', fontsize=14, weight='bold')
    ax.set_ylabel('Bs (T)', fontsize=14, weight='bold')
    ax.set_title(f'External Validation: Predicted vs Experimental Bs (MAE = {mae:.3f} T)', 
                fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Alloy {i+1}' for i in range(n)], rotation=45, ha='right')
    ax.legend(fontsize=12, frameon=True)
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate MAE regions
    ax.axhline(y=1.9, color='green', linestyle='--', linewidth=1.5, alpha=0.5, 
              label='High Bs (>1.9T)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved validation bar chart: {output_path}")


def plot_property_comparison_scatter(dataset_props, validated_props, candidate_props, 
                                     output_dir, bs_thresh=1.75, lnhc_thresh=1.5, dc_thresh=1.0):
    """
    Plot property comparisons: candidates vs validated vs dataset.
    
    Args:
        dataset_props: Dict with dataset properties
        validated_props: Dict with validated alloy properties
        candidate_props: Dict with candidate alloy properties (from NSGA-III)
        output_dir: Output directory
        bs_thresh, lnhc_thresh, dc_thresh: Target thresholds
    """
    fig_configs = [
        ('Bs', 'ln(Hc)', 'Bs (T)', 'ln(Hc) (A/m)', [0, 2.5], [-3, 4], True),
        ('Bs', 'Dc', 'Bs (T)', 'Dc (mm)', [0, 2.5], [0, 8], False),
        ('Dc', 'ln(Hc)', 'Dc (mm)', 'ln(Hc) (A/m)', [0, 8], [-2, 4], True)
    ]
    
    for idx, (x_key, y_key, xlabel, ylabel, xlim, ylim, invert_y) in enumerate(fig_configs):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot dataset alloys (hollow gray squares)
        ax.scatter(dataset_props[x_key], dataset_props[y_key], 
                  facecolors='none', edgecolors='gray', marker='s',
                  s=80, alpha=0.9, linewidths=1.2, label='Dataset alloys')
        
        # Plot validated alloys (blue triangles)
        ax.scatter(validated_props[x_key], validated_props[y_key], 
                  c='blue', marker='^', s=100, alpha=0.9, 
                  edgecolors='k', linewidths=1.0, label='Validated alloys')
        
        # Plot candidate alloys (red circles)
        ax.scatter(candidate_props[x_key], candidate_props[y_key], 
                  c='red', marker='o', s=100, alpha=0.9, 
                  edgecolors='k', linewidths=1.0, label='Candidate alloys')
        
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
        
        ax.tick_params(axis='x', labelsize=20, width=2)
        ax.tick_params(axis='y', labelsize=20, width=2)
        ax.set_xlabel(xlabel, fontsize=24, weight='bold')
        ax.set_ylabel(ylabel, fontsize=24, weight='bold')
        
        ax.legend(loc='upper right', frameon=False, prop={'weight': 'bold', 'size': 14})
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'validation_comparison_{idx+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comparison plot {idx+1}: {save_path}")


def main():
    ap = argparse.ArgumentParser(
        description='External validation and high-throughput discovery')
    
    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')
    
    # Model
    ap.add_argument('--ckpt', default='checkpoints/mtwae_k8_final_inverse.pth',
                   help='Path to trained MTWAE model')
    ap.add_argument('--latent', type=int, default=8)
    
    # Candidate alloys file (from NSGA-III with Bs>1.75T)
    ap.add_argument('--candidates', default='Optimizer-Bs-1.75.xlsx',
                   help='Excel file with candidate alloys from NSGA-III')
    
    # Output
    ap.add_argument('--outdir', default='validation_results')
    ap.add_argument('--seed', type=int, default=8)
    
    args = ap.parse_args()
    
    ensure_dir(args.outdir)
    set_seed(args.seed)
    
    print("="*80)
    print("External Validation and High-Throughput Discovery")
    print("="*80)
    
    # Load dataset
    print("\nLoading training dataset...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )
    
    # Prepare scalers (from training set)
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
    print(f"Loading model from {args.ckpt}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTWAE(in_features=D, latent_size=args.latent)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()
    
    # ========== Part 1: External Validation ==========
    print("\n" + "="*80)
    print("Part 1: External Validation on 12 Published Alloys")
    print("="*80)
    
    # 12 external compositions from recent literature
    external_comps = [
        "Fe68.2Co17.5B13Si0.5Cu0.8",          # Guo et al.
        "Fe79.7Co6B13Si0.5Cu0.8",              # Guo et al.
        "Fe85B12Si2V0.5Cu0.5",                 # Li et al.
        "Fe76.5Co8.5B12Si2V0.5Cu0.5",         # Li et al.
        "Fe68Co17B12Si2V0.5Cu0.5",            # Li et al.
        "Fe68.8Co17.2B11Si2V0.5Cu0.5",        # Li et al.
        "Fe70.11Co15.39Ni1.5B9P3C1",          # Yang et al.
        "Fe69Co16Ni1Si3B11",                   # Yang et al.
        "Fe64.13Co21.38Ni1.5B8.5P3C1V0.5",    # Yang et al.
        "Fe68.4Co17.1Ni1.5B9P3C1",            # Yang et al.
        "Fe64.13Co21.38Ni1.5B8.5P3C1Mo0.5",   # Yang et al.
        "Fe64.13Co21.38Ni1.5B9P3C1",          # Yang et al.
    ]
    
    # Experimental values
    experimental = pd.DataFrame({
        'Composition': external_comps,
        'Bs_exp (T)': [1.90, 1.81, 1.70, 1.77, 1.84, 1.82, 
                       1.75, 1.72, 1.74, 1.75, 1.74, 1.75],
        'ln(Hc)_exp (A/m)': [1.19, 1.26, 2.57, 2.10, 1.80, 2.45,
                             3.33, 2.08, 2.80, 3.37, 2.82, 2.86],
        'Dc_exp (mm)': [1.24, 1.39, 1.32, 1.17, 1.38, 1.33,
                        1.38, 1.12, 1.29, 1.37, 1.29, 1.32]
    })
    
    # Predict
    print("\nPredicting properties for external compositions...")
    predictions = predict_external_compositions(
        model, external_comps, scalers, PERIODIC_30, device
    )
    
    # Calculate errors
    mae_bs = np.mean(np.abs(predictions['Bs_pred (T)'] - experimental['Bs_exp (T)']))
    mae_hc = np.mean(np.abs(predictions['ln(Hc)_pred (A/m)'] - experimental['ln(Hc)_exp (A/m)']))
    mae_dc = np.mean(np.abs(predictions['Dc_pred (mm)'] - experimental['Dc_exp (mm)']))
    
    print(f"\nValidation Results:")
    print(f"  MAE (Bs):     {mae_bs:.4f} T")
    print(f"  MAE (ln(Hc)): {mae_hc:.4f}")
    print(f"  MAE (Dc):     {mae_dc:.4f} mm")
    
    # Save predictions
    combined = predictions.copy()
    combined['Bs_exp (T)'] = experimental['Bs_exp (T)']
    combined['ln(Hc)_exp (A/m)'] = experimental['ln(Hc)_exp (A/m)']
    combined['Dc_exp (mm)'] = experimental['Dc_exp (mm)']
    
    combined.to_csv(os.path.join(args.outdir, 'external_validation.csv'), index=False)
    print(f"\n  Saved predictions to {args.outdir}/external_validation.csv")
    
    # Plot validation bar chart
    plot_validation_bar_chart(
        predictions, experimental,
        os.path.join(args.outdir, 'validation_bar_chart.png')
    )
    
    # ========== Part 2: High-Throughput Discovery Visualization ==========
    print("\n" + "="*80)
    print("Part 2: High-Throughput Discovery Visualization")
    print("="*80)
    
    # Load candidate alloys from NSGA-III (Bs>1.75T)
    if os.path.exists(args.candidates):
        print(f"\nLoading candidate alloys from {args.candidates}...")
        df_candidates = pd.read_excel(args.candidates)
        
        # Filter: ln(Hc) < 1.5 AND Dc > 1.0
        mask = (df_candidates['ln(Hc) (A/m)'] < 1.5) & (df_candidates['Dc (mm)'] > 1.0)
        df_candidates_filtered = df_candidates[mask]
        
        print(f"  Total candidates: {len(df_candidates)}")
        print(f"  Filtered (ln(Hc)<1.5, Dc>1): {len(df_candidates_filtered)}")
        
        candidate_props = {
            'Bs': df_candidates_filtered['Bs (T)'].values,
            'ln(Hc)': df_candidates_filtered['ln(Hc) (A/m)'].values,
            'Dc': df_candidates_filtered['Dc (mm)'].values
        }
    else:
        print(f"\nWarning: Candidate file {args.candidates} not found.")
        print("  Skipping high-throughput discovery visualization.")
        return
    
    # Prepare dataset properties (with valid measurements)
    Bs_array = yb.ravel()
    Hc_array = yh.ravel()  # Already ln(Hc)
    Dc_array = yd.ravel()
    
    # Get valid indices for each pair
    valid_bs_hc = ~np.isnan(Bs_array) & ~np.isnan(Hc_array)
    valid_bs_dc = ~np.isnan(Bs_array) & ~np.isnan(Dc_array)
    valid_dc_hc = ~np.isnan(Dc_array) & ~np.isnan(Hc_array)
    
    dataset_props = {
        'Bs': Bs_array[valid_bs_hc],
        'ln(Hc)': Hc_array[valid_bs_hc],
        'Dc': Dc_array[valid_bs_dc]
    }
    
    # Validated alloys properties
    validated_props = {
        'Bs': experimental['Bs_exp (T)'].values,
        'ln(Hc)': experimental['ln(Hc)_exp (A/m)'].values,
        'Dc': experimental['Dc_exp (mm)'].values
    }
    
    # Plot property comparisons
    print("\nGenerating property comparison plots...")
    plot_property_comparison_scatter(
        dataset_props, validated_props, candidate_props, 
        args.outdir, bs_thresh=1.75
    )
    
    print(f"\n{'='*80}")
    print(f"External validation complete!")
    print(f"Results saved to {args.outdir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()