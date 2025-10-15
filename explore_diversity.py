
"""
Generative Exploration & Diversity Analysis
- Use trained MTWAE (k=8) to sample from prior N(0, σ²I) and generate alloy compositions
- Predict Bs, ln(Hc), Dc; plot existing vs generated vs target subset property distributions
- Calculate "uniqueness" and "novelty" ratios (1 wt.% threshold, same element set)
- t-SNE / PCA visualization (2D), colored by Bs / ln(Hc) / Dc in latent space
- Output key statistics and figures to outputs/
"""
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

from models import MTWAE
from data import load_all_targets
from utils import (set_seed, ensure_dir, load_checkpoint, 
                   make_scalers_from_train_split, plot_kde_overlap)

# ====== Thresholds ======
BS_THRESH = 1.5       # T (Tesla)
LNHC_THRESH = 1.5     # A/m (on ln scale)
DC_THRESH = 1.0       # mm


def is_similar(comp1, comp2, frac_tol=0.01):
    """
    Check if two compositions are similar.
    Same element set & corresponding content difference < 1 wt.% → 'similar'
    
    Args:
        comp1, comp2: Composition vectors
        frac_tol: Tolerance threshold (default: 0.01 = 1 wt.%)
    
    Returns:
        True if similar, False otherwise
    """
    nz1 = set(np.where(comp1 > 0)[0])
    nz2 = set(np.where(comp2 > 0)[0])
    if nz1 != nz2:
        return False
    return np.all(np.abs(comp1 - comp2) < frac_tol)


def calc_uniqueness(recon_x, frac_tol=0.01):
    """
    Uniqueness: Proportion of compositions that are dissimilar (>1%) 
    within the generated set.
    
    Args:
        recon_x: Generated composition matrix
        frac_tol: Tolerance threshold (default: 0.01)
    
    Returns:
        Uniqueness ratio
    """
    n = recon_x.shape[0]
    unique_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not unique_mask[i]:
            continue
        for j in range(i+1, n):
            if is_similar(recon_x[i], recon_x[j], frac_tol):
                unique_mask[j] = False
    return unique_mask.mean()


def calc_novelty(recon_x, existing_x, frac_tol=0.01):
    """
    Novelty: Proportion of generated compositions that are not similar 
    to any existing samples.
    
    Args:
        recon_x: Generated composition matrix
        existing_x: Existing composition matrix
        frac_tol: Tolerance threshold (default: 0.01)
    
    Returns:
        Novelty ratio
    """
    novel = 0
    for i in range(recon_x.shape[0]):
        similar_to_any = False
        for j in range(existing_x.shape[0]):
            if is_similar(recon_x[i], existing_x[j], frac_tol):
                similar_to_any = True
                break
        if not similar_to_any:
            novel += 1
    return novel / recon_x.shape[0]


def main():
    ap = argparse.ArgumentParser(
        description='Generative exploration and diversity analysis of MTWAE')
    ap.add_argument("--latent", type=int, default=8,
                   help='Latent space dimension')
    ap.add_argument("--sigma", type=float, default=8.0,
                   help='Standard deviation of Gaussian prior')
    ap.add_argument("--n", type=int, default=3000, 
                   help="Number of generated alloys")
    ap.add_argument("--ckpt", type=str, 
                   default="checkpoints/mtwae_k8_final_inverse.pth",
                   help='Path to trained model checkpoint')
    ap.add_argument("--outdir", type=str, default="outputs",
                   help='Output directory for results')
    ap.add_argument("--seed", type=int, default=8,
                   help='Random seed (8)')
    ap.add_argument("--embed", choices=["tsne", "pca"], default="tsne",
                   help='Embedding method for visualization')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    # 1) Load data & scaler (Bs, ln(Hc), Dc; 
    #    composition as 30-dim simplex vector)
    print("Loading data...")
    comp_X, Bs_y, lnHc_y, Dc_y, periodic = load_all_targets()
    print(f"Loaded {len(comp_X)} alloy compositions")
    
    # 2) Create target standardization scalers from training set, 
    #    consistent with training procedure
    scalers = make_scalers_from_train_split(
        Bs_y, lnHc_y, Dc_y, split_seed=8, test_size=0.2
    )

    # 3) Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MTWAE(in_features=comp_X.shape[1], latent_size=args.latent)
    load_checkpoint(model, args.ckpt, device)
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.ckpt}")

    # 4) Sample from prior → Generate compositions & Predict properties
    print(f"\nGenerating {args.n} alloy compositions...")
    z = np.random.randn(args.n, args.latent) * args.sigma
    
    with torch.no_grad():
        zt = torch.tensor(z, dtype=torch.float32).to(device)
        
        # CORRECTED: Use model's decode() and predict() methods
        x_gen = model.decode(zt).cpu().numpy()  # Simplex composition
        yb_gen, yh_gen, yd_gen = model.predict(zt)  # Property predictions
        
        Bs_gen = yb_gen.cpu().numpy()
        lnHc_gen = yh_gen.cpu().numpy()
        Dc_gen = yd_gen.cpu().numpy()
    
    # Inverse standardization to physical scale (lnHc remains on ln scale)
    Bs_gen = scalers["Bs"].inverse_transform(Bs_gen).ravel()
    lnHc_gen = scalers["Hc"].inverse_transform(lnHc_gen).ravel()
    Dc_gen = scalers["Dc"].inverse_transform(Dc_gen).ravel()

    # 5) Property distribution: existing vs generated vs target subset
    print("\nPlotting property distributions...")
    target_mask = ((Bs_gen > BS_THRESH) & 
                   (lnHc_gen < LNHC_THRESH) & 
                   (Dc_gen > DC_THRESH))
    
    n_target = int(target_mask.sum())
    print(f"Target alloys (Bs>{BS_THRESH}, ln(Hc)<{LNHC_THRESH}, "
          f"Dc>{DC_THRESH}): {n_target}/{args.n}")
    
    plot_kde_overlap(
        Bs_y.ravel(), Bs_gen, Bs_gen[target_mask], 
        "Bs (T)", os.path.join(args.outdir, "dist_Bs.png")
    )
    plot_kde_overlap(
        lnHc_y.ravel(), lnHc_gen, lnHc_gen[target_mask], 
        "ln(Hc) (A/m)", os.path.join(args.outdir, "dist_lnHc.png")
    )
    plot_kde_overlap(
        Dc_y.ravel(), Dc_gen, Dc_gen[target_mask], 
        "Dc (mm)", os.path.join(args.outdir, "dist_Dc.png")
    )

    # 6) Uniqueness & Novelty 
    print("\nCalculating diversity metrics...")
    uniq_ratio = calc_uniqueness(x_gen, frac_tol=0.01)
    novel_ratio = calc_novelty(x_gen, comp_X, frac_tol=0.01)
    
    print(f"Uniqueness @1%: {uniq_ratio:.2%}")
    print(f"Novelty @1%: {novel_ratio:.2%}")
    
    with open(os.path.join(args.outdir, "diversity_stats.txt"), "w") as f:
        f.write(f"Unique ratio @1%: {uniq_ratio:.4f}\n")
        f.write(f"Novel ratio @1%: {novel_ratio:.4f}\n")

    # 7) Latent space visualization: 
    #    Dimensionality reduction and coloring of latent vectors
    #    encode existing alloy compositions to get Z, then color
    print(f"\nCreating latent space visualizations ({args.embed})...")
    with torch.no_grad():
        X_tensor = torch.tensor(comp_X, dtype=torch.float32).to(device)
        Z_exist = model.encode(X_tensor).cpu().numpy()

    # Dimensionality reduction
    if args.embed == "pca":
        reducer = PCA(n_components=2, random_state=args.seed)
        Z2 = reducer.fit_transform(Z_exist)
        print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    else:
        reducer = TSNE(n_components=2, random_state=args.seed, 
                      perplexity=30, n_iter=1000)
        Z2 = reducer.fit_transform(Z_exist)

    def scatter_color(Z2, c, title, fname):
        """Helper function to create colored scatter plot"""
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=c, s=12, cmap='viridis', alpha=0.6)
        plt.colorbar(sc, label=title.split('by ')[-1])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")

    scatter_color(Z2, Bs_y.ravel(), "Latent space colored by Bs", "latent_Bs.png")
    scatter_color(Z2, lnHc_y.ravel(), "Latent space colored by ln(Hc)", "latent_lnHc.png")
    scatter_color(Z2, Dc_y.ravel(), "Latent space colored by Dc", "latent_Dc.png")

    # 8) Summary statistics 
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"=== MTWAE Generative Exploration Summary ===\n\n")
        f.write(f"Total generated: {args.n}\n")
        f.write(f"Target alloys (Bs>{BS_THRESH}T, ln(Hc)<{LNHC_THRESH}, "
                f"Dc>{DC_THRESH}mm): {n_target} ({n_target/args.n*100:.1f}%)\n\n")
        f.write(f"Diversity Metrics:\n")
        f.write(f"  Unique ratio @1%: {uniq_ratio:.4f} ({uniq_ratio*100:.1f}%)\n")
        f.write(f"  Novel ratio @1%: {novel_ratio:.4f} ({novel_ratio*100:.1f}%)\n\n")
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Generated {args.n} alloys, {n_target} satisfy target criteria")
    print(f"Uniqueness: {uniq_ratio:.2%}, Novelty: {novel_ratio:.2%}")
    print(f"Results saved to {args.outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
