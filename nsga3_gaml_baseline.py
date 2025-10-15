
"""
GA+ML Baseline for Comparison Study

Traditional machine learning (RF + XGBoost) combined with NSGA-III
for multi-objective optimization.

Protocol:
- Composition space: 6 elements (Fe, B, Si, C, P, Co)
- 26 alloy systems (ternary to 6-element combinations)
- Each system: 500 population × 1000 generations
- Total: 13,000 samples

Models:
- XGBoost for Bs and Dc
- Random Forest for Hc (ln-transformed)
"""

import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import geatpy as ea

from data import load_text_data, PERIODIC_30
from utils import set_seed, ensure_dir


def train_ml_models(X_bs, y_bs, X_hc, y_hc, X_dc, y_dc, seed=8):
    """
    Train traditional ML models with hyperparameter tuning.

    Returns:
        Tuple of (xgb_bs, rf_hc, xgb_dc, scalers)
    """
    print("\nTraining ML models with 5-fold CV hyperparameter tuning...")

    # Split and standardize (fit scalers on training targets only)
    _, _, yb_tr, _ = train_test_split(X_bs, y_bs, test_size=0.2, random_state=seed)
    _, _, yh_tr, _ = train_test_split(X_hc, y_hc, test_size=0.2, random_state=seed)
    _, _, yd_tr, _ = train_test_split(X_dc, y_dc, test_size=0.2, random_state=seed)

    scalers = {
        'Bs': StandardScaler().fit(yb_tr),
        'Hc': StandardScaler().fit(yh_tr),  # ln(Hc)
        'Dc': StandardScaler().fit(yd_tr)
    }

    # Standardize targets
    y_bs_s = scalers['Bs'].transform(y_bs)
    y_hc_s = scalers['Hc'].transform(y_hc)
    y_dc_s = scalers['Dc'].transform(y_dc)

    # -------------------------------
    # XGBoost for Bs
    # -------------------------------
    print("  Training XGBoost for Bs...")
    xgb_bs_params = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [2, 4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'gamma': [0.001, 0.005, 0.01, 0.05],
        'reg_lambda': [0.01, 0.05, 0.1, 0.5],
        'reg_alpha': [0.0, 0.03, 0.06, 0.1]
    }
    xgb_bs = RandomizedSearchCV(
        XGBRegressor(random_state=seed, verbosity=0),
        xgb_bs_params, n_iter=30, cv=5, scoring='r2',
        random_state=seed, n_jobs=-1
    )
    xgb_bs.fit(X_bs, y_bs_s.ravel())
    print(f"    Best CV R²: {xgb_bs.best_score_:.4f}")

    # -------------------------------
    # Random Forest for ln(Hc)
    # -------------------------------
    print("  Training Random Forest for ln(Hc)...")
    rf_hc_params = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [4, 8, 12, 16, 20, 24],
        'min_samples_split': [2, 4, 8, 16],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    rf_hc = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed),
        rf_hc_params, n_iter=30, cv=5, scoring='r2',
        random_state=seed, n_jobs=-1
    )
    rf_hc.fit(X_hc, y_hc_s.ravel())
    print(f"    Best CV R²: {rf_hc.best_score_:.4f}")

    # -------------------------------
    # XGBoost for Dc
    # -------------------------------
    print("  Training XGBoost for Dc...")
    xgb_dc_params = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [2, 4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'gamma': [0.001, 0.005, 0.01, 0.05],
        'reg_lambda': [0.01, 0.05, 0.1, 0.5],
        'reg_alpha': [0.0, 0.03, 0.06, 0.1]
    }
    xgb_dc = RandomizedSearchCV(
        XGBRegressor(random_state=seed, verbosity=0),
        xgb_dc_params, n_iter=30, cv=5, scoring='r2',
        random_state=seed, n_jobs=-1
    )
    xgb_dc.fit(X_dc, y_dc_s.ravel())
    print(f"    Best CV R²: {xgb_dc.best_score_:.4f}")

    return xgb_bs.best_estimator_, rf_hc.best_estimator_, xgb_dc.best_estimator_, scalers


def composition_to_string(composition, element_table, threshold=0.001):
    """Convert composition vector to chemical formula string."""
    composition_string = ''
    for i, frac in enumerate(composition):
        if frac > threshold:
            percentage = round(frac * 100, 2)
            composition_string += f'{element_table[i]}{percentage:.2f}'
    return composition_string


class GAMLProblem(ea.Problem):
    """
    Multi-objective optimization problem for GA+ML approach.

    Optimizes in composition space with selected elements.
    """

    def __init__(self, models, scalers, element_subset, element_ranges, periodic_table):
        self.models = models  # (xgb_bs, rf_hc, xgb_dc)
        self.scalers = scalers
        self.element_subset = element_subset
        self.periodic_table = periodic_table

        name = f'GAML_{"_".join(element_subset)}'
        M = 3
        maxormins = [-1, 1, -1]  # Maximize Bs, minimize ln(Hc), maximize Dc
        Dim = len(periodic_table)
        varTypes = [0] * Dim

        # Set bounds based on element subset
        lb, ub = [], []
        for elem in periodic_table:
            if elem in element_subset:
                lb.append(element_ranges[elem][0] * 0.01)
                ub.append(element_ranges[elem][1] * 0.01)
            else:
                lb.append(0)
                ub.append(0)

        lbin, ubin = [1] * Dim, [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):
        # Constraint: composition sum ≈ 1 (tolerance 0.01)
        sum_x = np.sum(Vars, axis=1, keepdims=True)
        CV = np.maximum(0, np.abs(sum_x - 1) - 0.01)

        # Predict properties using models = (xgb_bs, rf_hc, xgb_dc)
        xgb_bs, rf_hc, xgb_dc = self.models

        Bs_pred_s = xgb_bs.predict(Vars).reshape(-1, 1)
        Hc_pred_s = rf_hc.predict(Vars).reshape(-1, 1)
        Dc_pred_s = xgb_dc.predict(Vars).reshape(-1, 1)

        # Inverse standardization
        Bs_pred = self.scalers['Bs'].inverse_transform(Bs_pred_s)
        Hc_pred = self.scalers['Hc'].inverse_transform(Hc_pred_s)
        Dc_pred = self.scalers['Dc'].inverse_transform(Dc_pred_s)

        f = np.hstack([Bs_pred, Hc_pred, Dc_pred])

        return f, CV


def run_gaml_optimization(models, scalers, element_subset, element_ranges,
                          nind, maxgen, seed, output_dir):
    """
    Run GA+ML optimization for a specific element combination.

    Args:
        models: Tuple of (xgb_bs, rf_hc, xgb_dc)
        scalers: StandardScalers
        element_subset: List of elements to optimize
        element_ranges: Dict of element ranges
        nind: Population size
        maxgen: Maximum generations
        seed: Random seed
        output_dir: Output directory

    Returns:
        DataFrame with results
    """
    print(f"\n  Optimizing system: {'-'.join(element_subset)}")

    # Setup problem
    problem = GAMLProblem(models, scalers, element_subset, element_ranges, PERIODIC_30)

    # Setup algorithm
    algorithm = ea.moea_NSGA3_templet(
        problem,
        ea.Population(Encoding='RI', NIND=nind),
        MAXGEN=maxgen,
        logTras=0,
        seed=seed
    )

    # Run optimization
    res = ea.optimize(algorithm, verbose=False, drawLog=False,
                      outputMsg=False, saveFlag=False)

    # Extract Pareto front (final population solutions as returned by geatpy)
    obj_values = res['ObjV']
    obj_vars = res['Vars']

    # Convert to strings
    comp_strings = [composition_to_string(c, PERIODIC_30) for c in obj_vars]

    # Create DataFrame
    df = pd.DataFrame({
        'System': '-'.join(element_subset),
        'N_Elements': len(element_subset),
        'Composition': comp_strings,
        'Bs (T)': obj_values[:, 0],
        'ln(Hc) (A/m)': obj_values[:, 1],
        'Dc (mm)': obj_values[:, 2]
    })

    # Check satisfaction
    df['Satisfies_Criteria'] = (
        (df['Bs (T)'] > 1.5) &
        (df['ln(Hc) (A/m)'] < 1.5) &
        (df['Dc (mm)'] > 1.0)
    )

    n_satisfy = df['Satisfies_Criteria'].sum()
    print(f"    Generated {len(df)} solutions, {n_satisfy} satisfy criteria")

    return df


def main():
    ap = argparse.ArgumentParser(
        description='GA+ML baseline comparison study')

    # Data paths
    ap.add_argument('--comp', default='Composition_feature.txt')
    ap.add_argument('--bs', default='Bs_target.txt')
    ap.add_argument('--hc', default='Hc_target.txt')
    ap.add_argument('--dc', default='Dc_target.txt')

    # Element space
    ap.add_argument('--elements', nargs='+',
                    default=['Fe', 'B', 'Si', 'P', 'C', 'Co'],
                    help='Elements to consider (Fe B Si P C Co)')
    ap.add_argument('--min_elements', type=int, default=3,
                    help='Minimum number of elements')
    ap.add_argument('--max_elements', type=int, default=6,
                    help='Maximum number of elements')

    # Optimization
    ap.add_argument('--nind', type=int, default=500)
    ap.add_argument('--maxgen', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1)

    # Output
    ap.add_argument('--outdir', default='gaml_baseline')

    args = ap.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    print("="*80)
    print("GA+ML Baseline Comparison Study")
    print("="*80)

    # Load data
    print("\nLoading data...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )

    # Define element ranges (from dataset)
    element_ranges = {
        'Fe': [30.0, 89.0], 'B': [1.0, 25.0], 'Si': [0.9, 19.2],
        'P': [1.0, 14.0], 'C': [0.15, 10.0], 'Co': [1.0, 36.0]
    }

    # Train ML models (XGB for Bs/Dc, RF for ln(Hc))
    xgb_bs, rf_hc, xgb_dc, scalers = train_ml_models(Xb, yb, Xh, yh, Xd, yd, args.seed)
    models = (xgb_bs, rf_hc, xgb_dc)

    # Generate all element combinations
    print(f"\nGenerating alloy systems ({args.min_elements}-{args.max_elements} elements)...")

    # Ensure Fe is always included
    base_elements = [e for e in args.elements if e != 'Fe']
    all_systems = []

    for n in range(args.min_elements - 1, args.max_elements):
        for combo in itertools.combinations(base_elements, n):
            all_systems.append(('Fe',) + combo)

    print(f"Total systems to optimize: {len(all_systems)}")

    # Run optimization for each system
    all_results = []

    for idx, system in enumerate(all_systems):
        print(f"\nSystem {idx+1}/{len(all_systems)}")
        df = run_gaml_optimization(
            models, scalers, system, element_ranges,
            args.nind, args.maxgen, args.seed + idx, args.outdir
        )
        all_results.append(df)

    # Combine results
    print(f"\n{'='*60}")
    print("Combining results...")
    print(f"{'='*60}")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(os.path.join(args.outdir, 'gaml_combined_results.csv'),
                       index=False)

    # Calculate statistics
    satisfied_df = combined_df[combined_df['Satisfies_Criteria']]

    total = len(combined_df)
    n_satisfied = len(satisfied_df)
    success_rate = n_satisfied / total * 100

    print(f"\nGA+ML Baseline Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Satisfying criteria: {n_satisfied} ({success_rate:.1f}%)")
    print(f"  Average properties (satisfied):")
    print(f"    Bs:     {satisfied_df['Bs (T)'].mean():.3f} T")
    print(f"    ln(Hc): {satisfied_df['ln(Hc) (A/m)'].mean():.3f}")
    print(f"    Dc:     {satisfied_df['Dc (mm)'].mean():.3f} mm")

    # Save statistics
    stats = {
        'Framework': 'GA+ML',
        'Total_Systems': len(all_systems),
        'Total_Samples': total,
        'Satisfied_Samples': n_satisfied,
        'Success_Rate (%)': success_rate,
        'Avg_Bs (T)': satisfied_df['Bs (T)'].mean(),
        'Avg_ln(Hc)': satisfied_df['ln(Hc) (A/m)'].mean(),
        'Avg_Dc (mm)': satisfied_df['Dc (mm)'].mean()
    }

    pd.DataFrame([stats]).to_csv(
        os.path.join(args.outdir, 'gaml_statistics.csv'), index=False
    )

    print(f"\n{'='*80}")
    print(f"GA+ML baseline study complete!")
    print(f"Results saved to {args.outdir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
