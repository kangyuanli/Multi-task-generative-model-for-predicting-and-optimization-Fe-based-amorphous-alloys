
"""
NSMTWAE Framework for Comparison Study

This version loads TEN DISTINCT pre-trained MTWAE models
and runs one NSGA-III optimization per model.

Loading options (priority order):
1) --ckpt_list "a.pt,b.pt,...,j.pt"  (comma-separated, length == n_runs)
2) --ckpt_tpl "checkpoints/mtwae_k8_run_{i}.pth"  (i starts from 1)
3) fallback to --ckpt (same ckpt for all runs; not recommended)

Counting protocol:
- We treat the FINAL POPULATION of each run as the "generated samples"
  (typically equals population size, e.g., 500 per run), then concatenate
  across 10 runs (~5,000 samples). Set --select_front 1 if you want only
  the first Pareto front instead.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import geatpy as ea

from typing import List

from models import MTWAE
from data import load_text_data, PERIODIC_30
from utils import set_seed, ensure_dir

# --------------------------
# Utilities
# --------------------------
def composition_to_string(composition, element_table, threshold=0.001):
    """Convert composition vector to chemical formula string, e.g., Fe72.15B15.03..."""
    composition_string = ''
    for i, frac in enumerate(composition):
        if frac > threshold:
            percentage = round(frac * 100, 2)
            composition_string += f'{element_table[i]}{percentage:.2f}'
    return composition_string


def parse_ckpt_list(args) -> List[str]:
    """
    Resolve a list of checkpoint paths according to user inputs.
    Priority:
      1) --ckpt_list
      2) --ckpt_tpl with {i} placeholder (i = 1..n_runs)
      3) fallback to repeating --ckpt (warn)
    """
    if args.ckpt_list:
        paths = [p.strip() for p in args.ckpt_list.split(',') if p.strip()]
        if len(paths) != args.n_runs:
            raise ValueError(
                f'--ckpt_list expects exactly {args.n_runs} paths, got {len(paths)}.'
            )
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f'Checkpoint not found: {p}')
        return paths

    if args.ckpt_tpl:
        paths = []
        for i in range(1, args.n_runs + 1):
            p = args.ckpt_tpl.format(i=i)
            if not os.path.isfile(p):
                raise FileNotFoundError(f'Checkpoint not found from --ckpt_tpl: {p}')
            paths.append(p)
        return paths

    # Fallback (not recommended, kept for backward compatibility)
    print('[WARN] Neither --ckpt_list nor --ckpt_tpl provided. '
          'All runs will reuse --ckpt.')
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Fallback checkpoint not found: {args.ckpt}')
    return [args.ckpt for _ in range(args.n_runs)]


# --------------------------
# NSGA-III Problem
# --------------------------
class MultiObjectiveProblem(ea.Problem):
    """Multi-objective optimization in MTWAE latent space."""
    def __init__(self, model, scalers, latent_dim=8, n_samples=10000, sigma=8.0, device='cpu'):
        self.model = model
        self.scalers = scalers
        self.sigma = sigma
        self.device = device

        name = 'MTWAE_Framework'
        M = 3
        # Maximize Bs, minimize ln(Hc), maximize Dc:
        maxormins = [-1, 1, -1]
        Dim = latent_dim
        varTypes = [0] * Dim

        # Latent bounds estimated from prior N(0, sigma^2 I)
        prior_samples = sigma * torch.randn(n_samples, latent_dim)
        lb = prior_samples.min(dim=0)[0].tolist()
        ub = prior_samples.max(dim=0)[0].tolist()
        lbin, ubin = [1] * Dim, [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):
        # Vars: (N, latent_dim), numpy array
        Z_tensor = torch.from_numpy(Vars).float().to(self.device)
        with torch.no_grad():
            # predictors expect latent Z -> standardized properties
            Bs_s, Hc_s, Dc_s = self.model.predict(Z_tensor)
            Bs_s = Bs_s.detach().cpu().numpy().reshape(-1, 1)
            Hc_s = Hc_s.detach().cpu().numpy().reshape(-1, 1)
            Dc_s = Dc_s.detach().cpu().numpy().reshape(-1, 1)

        # inverse standardization to physical scales
        Bs = self.scalers['Bs'].inverse_transform(Bs_s)
        Hc = self.scalers['Hc'].inverse_transform(Hc_s)   # this is ln(Hc)
        Dc = self.scalers['Dc'].inverse_transform(Dc_s)

        f = np.hstack([Bs, Hc, Dc])  # (N, 3)
        return f  # no explicit CV (latent search is unconstrained)


# --------------------------
# Single run
# --------------------------
def run_single_optimization(model, scalers, run_id, nind, maxgen, seed, output_dir,
                            sigma=8.0, latent_dim=8, device='cpu',
                            select_front=0):
    """
    Run one NSGA-III optimization with a specific (already trained) MTWAE model.

    select_front:
      0 -> save the FINAL POPULATION
      1 -> save ONLY the FIRST Pareto front (strict Pareto optimal)
    """
    print(f"\n{'='*72}")
    print(f"Run {run_id+1}: Population={nind}, Generations={maxgen}, Seed={seed}")
    print(f"{'='*72}")

    # Setup problem & algorithm
    problem = MultiObjectiveProblem(model, scalers, latent_dim=latent_dim,
                                    n_samples=10000, sigma=sigma, device=device)
    algorithm = ea.moea_NSGA3_templet(
        problem,
        ea.Population(Encoding='RI', NIND=nind),
        MAXGEN=maxgen,
        logTras=0,
        seed=seed
    )

    # Optimize
    print("  Running NSGA-III optimization...")
    res = ea.optimize(algorithm, verbose=False, drawLog=False,
                      outputMsg=False, saveFlag=False)

    # Extract final population results
    obj_values = res['ObjV']          # (P, 3)
    latent_vars = res['Vars']         # (P, latent_dim)

    # Decide what to keep: all final population or first front only
    if select_front == 1:
        NDSet = ea.ndsortDED(obj_values)[0]
        keep_idx = np.where(NDSet == 1)[0]
        kept_values = obj_values[keep_idx]
        kept_vars = latent_vars[keep_idx]
        kept_label = "first Pareto-front solutions"
    else:
        kept_values = obj_values
        kept_vars = latent_vars
        kept_label = "final population solutions"

    print(f"  Collected {len(kept_values)} {kept_label}")

    # Decode compositions from kept latent vectors
    with torch.no_grad():
        z_t = torch.from_numpy(kept_vars).float().to(device)
        x_hat = model.decode(z_t).detach().cpu().numpy()  # (N, D)

    # small threshold to clean near-zeros for printing
    threshold = 1e-3
    x_hat[x_hat < threshold] = 0.0

    comp_strings = [composition_to_string(c, PERIODIC_30, threshold) for c in x_hat]

    # Build DataFrame
    df = pd.DataFrame({
        'Run': run_id + 1,
        'Composition': comp_strings,
        'Bs (T)': kept_values[:, 0],
        'ln(Hc) (A/m)': kept_values[:, 1],
        'Dc (mm)': kept_values[:, 2],
    })

    # Criteria flags
    df['Satisfies_Criteria'] = (
        (df['Bs (T)'] > 1.5) &
        (df['ln(Hc) (A/m)'] < 1.5) &
        (df['Dc (mm)'] > 1.0)
    )

    n_satisfy = int(df['Satisfies_Criteria'].sum())
    success_rate = 100.0 * n_satisfy / max(1, len(df))
    print(f"  Success rate: {n_satisfy}/{len(df)} ({success_rate:.1f}%)")
    print(f"  Avg properties (all kept): "
          f"Bs={df['Bs (T)'].mean():.3f} T, "
          f"ln(Hc)={df['ln(Hc) (A/m)'].mean():.3f}, "
          f"Dc={df['Dc (mm)'].mean():.3f} mm")

    # Save per-run results
    df.to_csv(os.path.join(output_dir, f'run_{run_id+1}_results.csv'), index=False)
    return df


# --------------------------
# Aggregation & plotting
# --------------------------
def plot_comparison_statistics(combined_df, output_dir):
    """Aggregate stats & plot histograms for satisfied samples."""
    satisfied_df = combined_df[combined_df['Satisfies_Criteria']].copy()

    total_samples = len(combined_df)
    n_satisfied = len(satisfied_df)
    success_rate = 100.0 * n_satisfied / max(1, total_samples)

    avg_bs = satisfied_df['Bs (T)'].mean()
    avg_hc = satisfied_df['ln(Hc) (A/m)'].mean()
    avg_dc = satisfied_df['Dc (mm)'].mean()

    print(f"\n{'='*60}")
    print("NSMTWAE Framework Statistics")
    print(f"{'='*60}")
    print(f"Total samples (all kept across runs): {total_samples}")
    print(f"Satisfying criteria: {n_satisfied} ({success_rate:.1f}%)")
    print("Average properties (among satisfied):")
    print(f"  Bs:     {avg_bs:.3f} T")
    print(f"  ln(Hc): {avg_hc:.3f}")
    print(f"  Dc:     {avg_dc:.3f} mm")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    props = ['Bs (T)', 'ln(Hc) (A/m)', 'Dc (mm)']
    thresholds = [1.5, 1.5, 1.0]
    for ax, prop, thr in zip(axes, props, thresholds):
        ax.hist(satisfied_df[prop], bins=30, alpha=0.75, edgecolor='k', linewidth=0.5)
        ax.axvline(thr, color='r', linestyle='--', linewidth=2, label=f'Threshold={thr}')
        ax.axvline(satisfied_df[prop].mean(), color='k', linestyle='-', linewidth=2,
                   label=f'Mean={satisfied_df[prop].mean():.2f}')
        ax.set_xlabel(prop)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nsmtwae_property_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV stats
    stats = {
        'Framework': 'NSMTWAE',
        'Total_Samples': total_samples,
        'Satisfied_Samples': n_satisfied,
        'Success_Rate (%)': success_rate,
        'Avg_Bs (T)': avg_bs,
        'Avg_ln(Hc)': avg_hc,
        'Avg_Dc (mm)': avg_dc
    }
    pd.DataFrame([stats]).to_csv(
        os.path.join(output_dir, 'nsmtwae_statistics.csv'), index=False
    )


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description='NSMTWAE framework comparison study')

    # Data
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs',   default='Bs_target.txt')
    ap.add_argument('--hc',   default='Hc_target.txt')
    ap.add_argument('--dc',   default='Dc_target.txt')

    # Model loading
    ap.add_argument('--ckpt', default='checkpoints/mtwae_k8_final_inverse.pth',
                    help='Fallback single checkpoint (NOT recommended).')
    ap.add_argument('--ckpt_list', type=str, default='',
                    help='Comma-separated list of ckpts, length == n_runs.')
    ap.add_argument('--ckpt_tpl',  type=str, default='',
                    help='Template with {i} placeholder, e.g., "checkpoints/run_{i}.pth".')
    ap.add_argument('--latent', type=int, default=8)
    ap.add_argument('--sigma',  type=float, default=8.0,
                    help='Gaussian prior std for latent bounds (uses 8.0).')

    # Optimization
    ap.add_argument('--n_runs', type=int, default=10,
                    help='Number of models/runs (10).')
    ap.add_argument('--nind',   type=int, default=500,
                    help='Population size (500).')
    ap.add_argument('--maxgen', type=int, default=1000,
                    help='Generations (1000).')
    ap.add_argument('--base_seed', type=int, default=1,
                    help='Base seed; per-run seed = base + 10*run_id.')
    ap.add_argument('--select_front', type=int, choices=[0,1], default=0,
                    help='0: keep final population; 1: only first Pareto front.')

    # Output
    ap.add_argument('--outdir', default='nsmtwae_comparison')

    args = ap.parse_args()
    ensure_dir(args.outdir)

    print("="*84)
    print("NSMTWAE Framework Comparison Study")
    print("="*84)
    print(f"Protocol: {args.n_runs} runs × {args.nind} pop × {args.maxgen} gen")

    # Resolve device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data & scalers (fit on training splits, consistent with Methods)
    print("\nLoading data and preparing scalers...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(args.comp, args.bs, args.hc, args.dc)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    _, _, yb_tr, _ = train_test_split(Xb, yb, test_size=0.2, random_state=8)
    _, _, yh_tr, _ = train_test_split(Xh, yh, test_size=0.2, random_state=8)
    _, _, yd_tr, _ = train_test_split(Xd, yd, test_size=0.2, random_state=8)

    scalers = {
        'Bs': StandardScaler().fit(yb_tr),
        'Hc': StandardScaler().fit(yh_tr),   # this scaler is for ln(Hc)
        'Dc': StandardScaler().fit(yd_tr)
    }

    # Resolve list of checkpoints (ten distinct models)
    ckpt_paths = parse_ckpt_list(args)

    # Run
    all_results = []
    for run_id, ckpt_path in enumerate(ckpt_paths):
        # Set per-run seed (sim affects NSGA-III)
        seed = args.base_seed + run_id * 10
        set_seed(seed)

        # Load that specific trained model
        print(f"\n[Model {run_id+1}/{args.n_runs}] Loading checkpoint: {ckpt_path}")
        model = MTWAE(in_features=D, latent_size=args.latent).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # One NSGA-III optimization for this model
        df = run_single_optimization(
            model=model, scalers=scalers, run_id=run_id,
            nind=args.nind, maxgen=args.maxgen, seed=seed,
            output_dir=args.outdir, sigma=args.sigma,
            latent_dim=args.latent, device=device,
            select_front=args.select_front
        )
        all_results.append(df)

    # Combine & export
    print(f"\n{'='*60}")
    print("Combining results from all runs...")
    print(f"{'='*60}")
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(os.path.join(args.outdir, 'combined_results_all.csv'), index=False)

    satisfied_df = combined_df[combined_df['Satisfies_Criteria']].copy()
    satisfied_df.to_csv(os.path.join(args.outdir, 'combined_results_satisfied.csv'), index=False)

    # Stats & plots
    plot_comparison_statistics(combined_df, args.outdir)

    print(f"\n{'='*84}")
    print("NSMTWAE comparison study complete!")
    print(f"Results saved to: {args.outdir}/")
    print(f"{'='*84}")


if __name__ == '__main__':
    main()
