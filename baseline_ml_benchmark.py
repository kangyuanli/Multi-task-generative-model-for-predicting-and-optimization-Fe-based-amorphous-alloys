"""
Traditional ML baselines with 5-fold CV
Benchmark models: SVR, KNN, RF, XGBoost
Properties: Bs, ln(Hc), Dc
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from data import load_all_targets
from utils import set_seed, ensure_dir, standardize_targets


def search_space(model_name):
    """
    Define hyperparameter search space for each model.
    """
    if model_name == "SVR":
        return {
            "C": [2, 5, 10, 20, 40, 80, 120, 240],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": [0.001, 0.005, 0.01, 0.05, 0.1]
        }
    if model_name == "KNN":
        return {
            "n_neighbors": [2, 3, 4, 5, 6, 7],
            "weights": ["uniform", "distance"],
            "leaf_size": [10, 20, 30, 40, 50]
        }
    if model_name == "RF":
        return {
            "n_estimators": [50, 100, 150, 200, 300],
            "max_depth": [4, 8, 12, 16, 20, 24],
            "min_samples_split": [2, 4, 8, 16],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["auto", "sqrt", "log2", None]
        }
    if model_name == "XGB":
        return {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [2, 4, 8, 12],
            "learning_rate": [0.05, 0.1, 0.2, 0.3],
            "gamma": [0.001, 0.005, 0.01, 0.05],
            "reg_lambda": [0.01, 0.05, 0.1, 0.5],
            "reg_alpha": [0.0, 0.03, 0.06, 0.1]
        }
    raise ValueError(f"Unknown model: {model_name}")


def make_estimator(model_name):
    """Create estimator instance for given model name."""
    estimators = {
        "SVR": svm.SVR(),
        "KNN": KNeighborsRegressor(),
        "RF": RandomForestRegressor(random_state=8),
        "XGB": XGBRegressor(random_state=8, verbosity=0),
    }
    return estimators[model_name]


def run_one_property(X, y, prop_name, seed=8):
    """
    Run hyperparameter search and cross-validation for all models on one property.
    
    Args:
        X: Composition features
        y: Target property values (standardized)
        prop_name: Property name ('Bs', 'lnHc', or 'Dc')
        seed: Random seed
    
    Returns:
        List of dicts containing results for each model
    """
    results = []
    models = ["SVR", "KNN", "RF", "XGB"]
    
    print(f"\n  Property: {prop_name}")
    print(f"  {'Model':<10} {'CV R²':<10} {'Best params'}")
    print(f"  {'-'*60}")
    
    for name in models:
        est = make_estimator(name)
        params = search_space(name)
        
        # Random search with 5-fold CV
        rs = RandomizedSearchCV(
            estimator=est,
            param_distributions=params,
            n_iter=100,
            cv=5,
            scoring="r2",
            random_state=seed,
            n_jobs=-1,
            verbose=0
        )
        
        rs.fit(X, y.ravel())
        
        # Store results
        result = {
            "property": prop_name,
            "model": name,
            "cv_R2": rs.best_score_,
            "best_params": str(rs.best_params_)
        }
        results.append(result)
        
        print(f"  {name:<10} {rs.best_score_:>8.4f}  {rs.best_params_}")
    
    return results


def main():
    ap = argparse.ArgumentParser(
        description='Traditional ML baseline benchmarking')
    ap.add_argument("--outdir", type=str, default="outputs",
                   help='Output directory')
    ap.add_argument("--seed", type=int, default=8,
                   help='Random seed')
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    print("Traditional ML Baseline Benchmarking")

    # Load data
    print("\nLoading data...")
    X, Bs, lnHc, Dc, periodic = load_all_targets()
    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")

    # Standardize targets (consistent with training procedure)
    # Note: R² is invariant to affine transformations, but we maintain consistency
    Bs_s, lnHc_s, Dc_s, scalers = standardize_targets(Bs, lnHc, Dc)

    # Run benchmarking for each property
    print("\nRunning 5-fold cross-validation with hyperparameter search...")

    all_rows = []
    all_rows += run_one_property(X, Bs_s, "Bs", seed=args.seed)
    all_rows += run_one_property(X, lnHc_s, "lnHc", seed=args.seed)
    all_rows += run_one_property(X, Dc_s, "Dc", seed=args.seed)

    # Save results
    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=["property", "cv_R2"], ascending=[True, False])
    
    csv_path = os.path.join(args.outdir, "table_S4.csv")
    df.to_csv(csv_path, index=False)
    

    print("Summary of Results")
    print(df[['property', 'model', 'cv_R2']].to_string(index=False))
    print(f"\nFull results saved to: {csv_path}")



if __name__ == "__main__":
    main()
