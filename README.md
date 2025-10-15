# Multi-task-Wasserstein-Autoencoder-for-inverse-design-of-Fe-based-metallic-glasses

## File Descriptions

### Core Model Files
* **models.py**: Defines the Multi-task Wasserstein Autoencoder (MTWAE) architecture with encoder, decoder, and three property predictors for simultaneous prediction of Bs, ln(Hc), and Dc.
* **losses.py**: Implements loss functions including binary cross-entropy reconstruction loss, Maximum Mean Discrepancy (MMD) with IMQ kernel, and Kendall uncertainty weighting.
* **train.py**: Contains training and evaluation functions for the MTWAE model with multi-task learning strategy.
* **data.py**: Handles data loading and preprocessing for composition features and three target properties (Bs, Hc, Dc), including train-test splitting and standardization.
* **utils.py**: Utility functions including random seed setting, checkpoint saving, inverse sample size weighting, and visualization helpers.

### Training and Validation Scripts
* **train_mtwae.py**: Main training script for MTWAE with configurable hyperparameters (epochs, batch size, learning rate, latent dimension, weighting strategy).
* **cross_validation_diff_weight.py**: Performs 5-fold cross-validation experiments to compare different weighting strategies (inverse, equal, uncertainty) and latent dimensions (k=2,4,8,16).
* **baseline_ml_benchmark.py**: Benchmarks traditional ML models (SVR, KNN, RF, XGBoost, Ridge) with 5-fold CV for comparison.

### Multi-objective Optimization Scripts
* **nsga3_inverse_design.py**: Implements NSGA-III multi-objective optimization in latent space to find Pareto optimal alloy compositions.
* **nsga3_evolution_vis.py**: Visualizes the evolutionary process of NSGA-III optimization across generations (0, 10, 200).
* **nsga3_gaml_baseline.py**: GA+ML baseline implementation combining traditional ML models with genetic algorithm for comparison.
* **nsga3_mtwae_comparison.py**: Framework comparison study running multiple MTWAE models with different initializations.

### Analysis and Validation Scripts
* **shap_analysis.py**: Performs SHAP (Shapley Additive Explanations) analysis to understand elemental contributions to each property.
* **element_frequency_analysis.py**: Analyzes element occurrence frequencies and mean contents from generated virtual library of candidates.
* **explore_diversity.py**: Evaluates generative capabilities including uniqueness, novelty, and latent space visualization using t-SNE/PCA.
* **external_validation.py**: Validates model predictions on 12 recently published Fe-based alloys and visualizes high-throughput discovery results.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

