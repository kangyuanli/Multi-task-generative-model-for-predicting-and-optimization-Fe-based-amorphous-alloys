"""
Multi-task Wasserstein Autoencoder (MTWAE) Training Script

This script trains the MTWAE model for predicting properties of Fe-based metallic glasses:
- Bs: Saturation magnetization
- Hc: Coercivity (log-transformed)
- Dc: Critical diameter

Model hyperparameters:
- Epochs: 800
- Batch size: 4
- Learning rate: 1e-3
- Latent dimension: 8
- λ_MMD: 1e-4

Three weighting strategies are available:
1. Inverse sample size weighting (recommended, best performance)
2. Equal weighting
3. Uncertainty weighting (Kendall)
"""

from __future__ import annotations
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import MTWAE
from data import load_text_data, train_val_split_standardize
from train import train_epoch, evaluate
from utils import set_seed, save_checkpoint, inverse_sample_size_weights


def _to_loader(X, y, batch, shuffle=True, drop_last=True):
    """Convert numpy arrays to PyTorch DataLoader."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, 
                     shuffle=shuffle, drop_last=drop_last)


def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser(description='Train MTWAE for metallic glass property prediction')
    ap.add_argument('--comp', default='Composition_feature.txt', 
                   help='Path to composition feature file')
    ap.add_argument('--bs', default='Bs_target.txt', 
                   help='Path to Bs (saturation magnetization) target file')
    ap.add_argument('--hc', default='Hc_target.txt', 
                   help='Path to Hc (coercivity) target file')
    ap.add_argument('--dc', default='Dc_target.txt', 
                   help='Path to Dc (critical diameter) target file')
    ap.add_argument('--epochs', type=int, default=800, 
                   help='Number of training epochs')
    ap.add_argument('--batch', type=int, default=4, 
                   help='Batch size (4)')
    ap.add_argument('--lr', type=float, default=1e-3, 
                   help='Learning rate (1e-3)')
    ap.add_argument('--latent', type=int, default=8, 
                   help='Latent space dimension k')
    ap.add_argument('--sigma', type=float, default=8.0, 
                   help='Standard deviation of Gaussian prior')
    ap.add_argument('--lambda_mmd', type=float, default=1e-4, 
                   help='MMD loss weight (1e-4)')
    ap.add_argument('--weighting', choices=['inverse','equal','uncertainty'], default='inverse',
                   help='Task weighting strategy (inverse recommended)')
    ap.add_argument('--seed', type=int, default=8, 
                   help='Random seed for reproducibility')
    args = ap.parse_args()

    # Set random seeds for reproducibility
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data for all three tasks
    print("Loading data...")
    (Xb, yb), (Xh, yh), (Xd, yd), D = load_text_data(
        args.comp, args.bs, args.hc, args.dc
    )
    print(f"Dataset sizes - Bs: {len(Xb)}, Hc: {len(Xh)}, Dc: {len(Xd)}")
    print(f"Input dimension: {D}")

    # Split and standardize each task separately
    Xb_tr, Xb_te, yb_tr, yb_te, sc_b = train_val_split_standardize(Xb, yb, args.seed)
    Xh_tr, Xh_te, yh_tr, yh_te, sc_h = train_val_split_standardize(Xh, yh, args.seed)
    Xd_tr, Xd_te, yd_tr, yd_te, sc_d = train_val_split_standardize(Xd, yd, args.seed)

    # Create DataLoaders with resampling strategy
    loaders_tr = {
        'Bs': _to_loader(Xb_tr, yb_tr, args.batch, shuffle=True, drop_last=True),
        'Hc': _to_loader(Xh_tr, yh_tr, args.batch, shuffle=True, drop_last=True),
        'Dc': _to_loader(Xd_tr, yd_tr, args.batch, shuffle=True, drop_last=True),
    }
    loaders_te = {
        'Bs': _to_loader(Xb_te, yb_te, args.batch, shuffle=False, drop_last=False),
        'Hc': _to_loader(Xh_te, yh_te, args.batch, shuffle=False, drop_last=False),
        'Dc': _to_loader(Xd_te, yd_te, args.batch, shuffle=False, drop_last=False),
    }

    # Initialize model and optimizer
    use_uncert = (args.weighting == 'uncertainty')
    model = MTWAE(
        in_features=D, 
        latent_size=args.latent, 
        use_uncertainty_weighting=use_uncert
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute task weights based on selected strategy
    nB, nH, nD = len(Xb_tr), len(Xh_tr), len(Xd_tr)
    if args.weighting == 'inverse':
        # Inverse sample size weighting
        w = inverse_sample_size_weights(nB, nH, nD)
        print(f"Using inverse sample size weighting: {w}")
    elif args.weighting == 'equal':
        # Equal weighting
        w = {'Bs': 1.0, 'Hc': 1.0, 'Dc': 1.0}
        print("Using equal weighting")
    else:  # uncertainty
        # Kendall uncertainty weighting (learnable)
        w = {'Bs': 1.0, 'Hc': 1.0, 'Dc': 1.0}
        print("Using uncertainty weighting (learnable σ parameters)")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        tr = train_epoch(model, loaders_tr, opt, device, args.sigma, args.lambda_mmd, w)
        
        # Evaluate on validation set
        te, mae, r2 = evaluate(model, loaders_te, device, args.sigma, args.lambda_mmd)
        
        # Print progress
        print(f"Epoch {epoch:03d} | Train Loss Bs {tr['Bs']:.4f} Hc {tr['Hc']:.4f} Dc {tr['Dc']:.4f} | MMD {tr['MMD']:.4f}")
        print(f"           | Val   Loss Bs {te['Bs']:.4f} Hc {te['Hc']:.4f} Dc {te['Dc']:.4f} | MMD {te['MMD']:.4f}")
        print(f"           | MAE   Bs {mae['Bs']:.4f} Hc {mae['Hc']:.4f} Dc {mae['Dc']:.4f}")
        print(f"           | R²    Bs {r2['Bs']:.4f} Hc {r2['Hc']:.4f} Dc {r2['Dc']:.4f}")
        
        # Save checkpoints periodically
        if epoch % 50 == 0:
            ckpt_path = f"checkpoints/mtwae_k{args.latent}_e{epoch}_{args.weighting}.pth"
            save_checkpoint(model.state_dict(), ckpt_path)
            print(f"           | Checkpoint saved: {ckpt_path}")

    # Save final model
    final_path = f"checkpoints/mtwae_k{args.latent}_final_{args.weighting}.pth"
    save_checkpoint(model.state_dict(), final_path)
    print("=" * 80)
    print(f"Training complete! Final model saved: {final_path}")


if __name__ == '__main__':
    main()