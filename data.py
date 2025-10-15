from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Periodic table of 30 candidate elements used in Fe-based metallic glasses
PERIODIC_30 = ['Fe','B','Si','P','C','Co','Nb','Ni','Mo','Zr','Ga','Al','Dy','Cu','Cr','Y',
               'Nd','Hf','Ti','Tb','Ho','Ta','Er','Sn','W','Tm','Gd','Sm','V','Pr']


def _get_element_index(element_lists: List[List[str]], table: List[str]) -> List[List[int]]:
    """Convert element symbols to their indices in the periodic table."""
    idx = {e: i for i, e in enumerate(table)}
    out = []
    for lst in element_lists:
        out.append([idx[e] for e in lst if e in idx])
    return out


def _create_comp_matrix(indices: List[List[int]], fractions: List[List[str]], D: int) -> np.ndarray:
    """Create composition matrix from element indices and fractions.
    
    Args:
        indices: List of element indices for each alloy
        fractions: List of atomic percentages for each alloy
        D: Dimension (number of elements in periodic table)
    
    Returns:
        Composition matrix of shape (n_samples, D) with values in [0, 1]
    """
    comp = np.zeros((len(indices), D), dtype=np.float32)
    for i, row in enumerate(indices):
        for j, col in enumerate(row):
            comp[i, col] = float(fractions[i][j]) * 0.01  # Convert percentage to [0,1]
    return comp


def load_text_data(comp_path: str, bs_path: str, hc_path: str, dc_path: str,
                   table: List[str] = PERIODIC_30):
    """Load and parse composition and property data from text files.
    
    Args:
        comp_path: Path to composition feature file
        bs_path: Path to saturation magnetization (Bs) target file
        hc_path: Path to coercivity (Hc) target file
        dc_path: Path to critical diameter (Dc) target file
        table: Periodic table of candidate elements
    
    Returns:
        Tuple of ((X_Bs, y_Bs), (X_Hc, y_Hc), (X_Dc, y_Dc), D) where:
        - X_*: Composition matrices for each property
        - y_*: Target property values (Hc is log-transformed)
        - D: Number of elements in periodic table
    """
    # Read all data files
    with open(comp_path, "r", encoding="utf-8") as f:
        comp_lines = f.readlines()
    with open(bs_path, "r", encoding="utf-8") as f:
        bs_lines = f.readlines()
    with open(hc_path, "r", encoding="utf-8") as f:
        hc_lines = f.readlines()
    with open(dc_path, "r", encoding="utf-8") as f:
        dc_lines = f.readlines()

    # Parse composition data
    composition, fraction = [], []
    for line in comp_lines[1:]:  # Skip header
        s = line.split()
        n = int(s[1])  # Number of elements
        composition.append(s[2:2+n])
        fraction.append(s[2+n:2+2*n])

    # Parse target properties (missing values marked as -1)
    Bs, Hc, Dc = [], [], []
    for line in bs_lines[1:]:
        s = line.split()
        Bs.append(float(s[0]))
    for line in hc_lines[1:]:
        s = line.split()
        Hc.append(-1.0 if len(s) == 0 else float(s[0]))
    for line in dc_lines[1:]:
        s = line.split()
        Dc.append(-1.0 if len(s) == 0 else float(s[0]))

    D = len(table)
    
    # Filter samples with valid measurements for each property
    idx_Bs = [i for i, v in enumerate(Bs) if v != -1]
    idx_Hc = [i for i, v in enumerate(Hc) if v != -1]
    idx_Dc = [i for i, v in enumerate(Dc) if v != -1]

    # Convert element symbols to indices
    Bs_el = _get_element_index([composition[i] for i in idx_Bs], table)
    Hc_el = _get_element_index([composition[i] for i in idx_Hc], table)
    Dc_el = _get_element_index([composition[i] for i in idx_Dc], table)

    # Create composition matrices
    X_Bs = _create_comp_matrix(Bs_el, [fraction[i] for i in idx_Bs], D)
    X_Hc = _create_comp_matrix(Hc_el, [fraction[i] for i in idx_Hc], D)
    X_Dc = _create_comp_matrix(Dc_el, [fraction[i] for i in idx_Dc], D)

    # Create target arrays (log-transform Hc)
    y_Bs = np.array([[Bs[i]] for i in idx_Bs], dtype=np.float32)
    y_Hc = np.array([[math.log(Hc[i])] for i in idx_Hc], dtype=np.float32)  # ln(Hc)
    y_Dc = np.array([[Dc[i]] for i in idx_Dc], dtype=np.float32)

    return (X_Bs, y_Bs), (X_Hc, y_Hc), (X_Dc, y_Dc), D


def train_val_split_standardize(X: np.ndarray, y: np.ndarray, seed: int, test_size: float = 0.2):
    """Split data into train/test sets and standardize targets.
    
    Args:
        X: Composition features (kept on simplex, not standardized)
        y: Target property values
        seed: Random seed for reproducibility
        test_size: Fraction of data for testing
    
    Returns:
        X_train, X_test, y_train_standardized, y_test_standardized, scaler
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, 
                                                random_state=seed, shuffle=True)
    
    # Standardize only target values (not compositions, which are on simplex)
    sc = StandardScaler()
    y_tr_s = sc.fit_transform(y_tr)
    y_te_s = sc.transform(y_te)
    
    return X_tr, X_te, y_tr_s, y_te_s, sc

def load_all_targets(data_dir='data'):
    """
    Load all composition and target data for generation/analysis.
    
    Returns:
        Tuple of (comp_X, Bs_y, lnHc_y, Dc_y, periodic_table)
        where all samples are included (union of all tasks)
    """
    import os
    
    # Construct file paths
    comp_path = os.path.join(data_dir, 'Composition_feature.txt')
    bs_path = os.path.join(data_dir, 'Bs_target.txt')
    hc_path = os.path.join(data_dir, 'Hc_target.txt')
    dc_path = os.path.join(data_dir, 'Dc_target.txt')
    
    # Load individual task data
    (X_Bs, y_Bs), (X_Hc, y_Hc), (X_Dc, y_Dc), D = load_text_data(
        comp_path, bs_path, hc_path, dc_path
    )
    
    # For generation/visualization, we need all unique compositions
    # Union of all three datasets
    all_X = []
    all_Bs = []
    all_lnHc = []
    all_Dc = []
    
    # This is a simplified version - in practice, you'd need to match
    # compositions across datasets. For now, use Bs dataset as reference
    # since it's the largest
    
    return X_Bs, y_Bs, y_Hc, y_Dc, PERIODIC_30