from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTWAE(nn.Module):
    """Multi-task Wasserstein Autoencoder for Fe-based Metallic Glass Property Prediction.
    
    The MTWAE consists of:
    1. Encoder q_φ(Z|X): Maps composition X to latent space Z
    2. Decoder p_θ(X̂|Z): Reconstructs composition from latent code
    3. Property Predictors f_ω(y|Z): Predict Bs, Hc, Dc from latent code
    
    Architecture details:
    - Encoder: [in_features → 90 → 48 → 30 → latent_size]
    - Decoder: [latent_size → 30 → 48 → 90 → in_features]
    - Predictors: [latent_size → 90 → 90 → 90 → 1]
    
    Each hidden layer uses LayerNorm + LeakyReLU(0.01).
    Decoder output uses Softmax to ensure valid compositions.
    """
    
    def __init__(self, in_features: int, latent_size: int, neg_slope: float = 0.01,
                 use_uncertainty_weighting: bool = False):
        """Initialize MTWAE model.
        
        Args:
            in_features: Input dimension (number of elements in periodic table)
            latent_size: Latent space dimension k 
            neg_slope: Negative slope for LeakyReLU activation (default: 0.01)
            use_uncertainty_weighting: Whether to use Kendall uncertainty weighting
        """
        super().__init__()
        act = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        
        # Encoder q_φ(Z|X): Maps composition to latent space
        # Architecture: [in_features → 90 → 48 → 30 → latent_size]
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 90), nn.LayerNorm(90), act,
            nn.Linear(90, 48), nn.LayerNorm(48), act,
            nn.Linear(48, 30), nn.LayerNorm(30), act,
            nn.Linear(30, latent_size),
        )
        
        # Decoder p_θ(X̂|Z): Reconstructs composition from latent space
        # Architecture: [latent_size → 30 → 48 → 90 → in_features]
        # Softmax applied in forward() to ensure valid composition
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 30), nn.LayerNorm(30), act,
            nn.Linear(30, 48), nn.LayerNorm(48), act,
            nn.Linear(48, 90), nn.LayerNorm(90), act,
            nn.Linear(90, in_features),
        )
        
        # Property Predictors f_ω(y|Z): Map latent code to properties
        # Each predictor: [latent_size → 90 → 90 → 90 → 1]
        def predictor():
            return nn.Sequential(
                nn.Linear(latent_size, 90), nn.LayerNorm(90), act,
                nn.Linear(90, 90), nn.LayerNorm(90), act,
                nn.Linear(90, 90), nn.LayerNorm(90), act,
                nn.Linear(90, 1),
            )
        
        self.pred_Bs = predictor()  # Saturation magnetization
        self.pred_Hc = predictor()  # Coercivity (log-transformed)
        self.pred_Dc = predictor()  # Critical diameter

        # Optional Kendall uncertainty weighting parameters
        # Each task has a learnable log(σ) parameter
        self.use_uncertainty = use_uncertainty_weighting
        if self.use_uncertainty:
            self.log_sigma_Bs = nn.Parameter(torch.zeros(1))
            self.log_sigma_Hc = nn.Parameter(torch.zeros(1))
            self.log_sigma_Dc = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        """Encode composition to latent space: Z = q_φ(Z|X)."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent code to composition: X̂ = p_θ(X̂|Z).
        
        Applies Softmax to ensure output is a valid composition (sums to 1).
        """
        return torch.softmax(self.decoder(z), dim=1)

    def predict(self, z):
        """Predict properties from latent code: y = f_ω(y|Z).
        
        Returns:
            Tuple of (y_Bs, y_Hc, y_Dc) predictions
        """
        return self.pred_Bs(z), self.pred_Hc(z), self.pred_Dc(z)

    def forward(self, x):
        """Full forward pass through MTWAE.
        
        Args:
            x: Input composition, shape (batch, in_features)
        
        Returns:
            Tuple of (x_hat, z, y_Bs, y_Hc, y_Dc) where:
            - x_hat: Reconstructed composition
            - z: Latent representation
            - y_Bs, y_Hc, y_Dc: Property predictions
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        yb, yh, yd = self.predict(z)
        return x_hat, z, yb, yh, yd