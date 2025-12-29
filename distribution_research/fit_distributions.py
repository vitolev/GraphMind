"""
Part 2: Fit Distributions to Distribution Research Data

This script:
1. Loads evaluation results from CSV
2. Fits multiple distributions:
   - Beta distribution (Bayesian)
   - Gaussian Mixture Model (2 components)
   - Normal distribution (truncated to [0,1])
3. Visualizes all distributions together
4. Saves distribution parameters to JSON
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import json

# Optional import for scipy/sklearn (for GMM)
try:
    from sklearn.mixture import GaussianMixture
    HAS_SCIPY_SKLEARN = True
except ImportError:
    HAS_SCIPY_SKLEARN = False
    print("Scikit-learn not found. Install with 'pip install scikit-learn' for better GMM fitting.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fit_beta_bayesian(scores: np.ndarray, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> Tuple[float, float]:
    """
    Fit Beta distribution using Bayesian estimation (method of moments).
    
    Args:
        scores: Array of scores in [0, 1]
        prior_alpha: Prior alpha parameter (default: 1.0 = uniform prior)
        prior_beta: Prior beta parameter (default: 1.0 = uniform prior)
    
    Returns:
        Tuple of (posterior_alpha, posterior_beta)
    """
    scores = np.array(scores)
    scores = scores[(scores >= 0) & (scores <= 1)]
    
    if len(scores) < 2:
        return (prior_alpha, prior_beta)
    
    # Method of moments for observed data
    mean_obs = np.mean(scores)
    var_obs = np.var(scores, ddof=0)
    
    if var_obs == 0 or mean_obs == 0 or mean_obs == 1:
        return (prior_alpha, prior_beta)
    
    # Method of moments estimate
    temp = (mean_obs * (1 - mean_obs) / var_obs) - 1
    if temp <= 0:
        return (prior_alpha, prior_beta)
    
    alpha_mom = mean_obs * temp
    beta_mom = (1 - mean_obs) * temp
    
    # Bayesian update: combine prior with observed data
    n = len(scores)
    prior_weight = 1.0 / (1.0 + n)
    data_weight = 1.0 - prior_weight
    
    posterior_alpha = prior_weight * prior_alpha + data_weight * alpha_mom
    posterior_beta = prior_weight * prior_beta + data_weight * beta_mom
    
    # Ensure positive
    posterior_alpha = max(0.1, posterior_alpha)
    posterior_beta = max(0.1, posterior_beta)
    
    return (posterior_alpha, posterior_beta)


def fit_normal_distribution(scores: np.ndarray) -> Dict[str, float]:
    """
    Fit a normal distribution to scores (truncated to [0,1]).
    
    Note: Normal distribution is not ideal for bounded [0,1] data, but included for comparison.
    We fit it using method of moments and note that it may assign probability outside [0,1].
    
    Args:
        scores: Array of scores in [0, 1]
    
    Returns:
        Dictionary with mean and std
    """
    scores = np.array(scores)
    scores = scores[(scores >= 0) & (scores <= 1)]
    
    if len(scores) < 2:
        return {'mean': 0.5, 'std': 0.2}
    
    mean = np.mean(scores)
    std = np.std(scores, ddof=0)
    
    # Clamp std to reasonable range
    std = np.clip(std, 0.01, 0.5)
    
    return {
        'mean': float(mean),
        'std': float(std)
    }


def fit_gaussian_mixture(scores: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
    """
    Fit a Gaussian Mixture Model (mixture of normal distributions) to scores.
    
    Justification: Random agent systems may exhibit bimodality:
    - Component 1: Well-structured graphs (e.g., with Solver nodes, proper agent flow) → higher scores
    - Component 2: Poorly-structured graphs (missing key agents, wrong connections) → lower scores
    
    The mixture model captures this structural heterogeneity better than a single distribution.
    
    Args:
        scores: Array of scores in [0, 1]
        n_components: Number of Gaussian components (default: 2)
    
    Returns:
        Dictionary with mixture model parameters
    """
    scores = np.array(scores)
    scores = scores[(scores >= 0) & (scores <= 1)]
    
    if len(scores) < n_components * 2:
        return None
    
    # Use scikit-learn's GaussianMixture if available (more robust)
    sklearn_succeeded = False
    if HAS_SCIPY_SKLEARN:
        try:
            # Reshape scores for sklearn (needs 2D array)
            scores_2d = scores.reshape(-1, 1)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
            gmm.fit(scores_2d)
            
            # Extract parameters
            weights = gmm.weights_
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            sklearn_succeeded = True
        except Exception as e:
            # Fallback to custom EM if sklearn fails
            sklearn_succeeded = False
    
    # Fallback to custom EM algorithm if sklearn not available or failed
    if not sklearn_succeeded:
        # Initialize parameters randomly
        np.random.seed(42)
        weights = np.ones(n_components) / n_components
        means = np.random.uniform(0.2, 0.8, n_components)
        stds = np.ones(n_components) * 0.2
        
        # EM algorithm (simplified)
        max_iter = 50
        tolerance = 1e-6
        
        for iteration in range(max_iter):
            # E-step: Compute responsibilities (posterior probabilities)
            responsibilities = np.zeros((len(scores), n_components))
            for k in range(n_components):
                # Normal PDF for component k
                diff = scores - means[k]
                responsibilities[:, k] = weights[k] * np.exp(-0.5 * (diff / stds[k])**2) / (stds[k] * np.sqrt(2 * np.pi))
            
            # Normalize responsibilities
            resp_sum = responsibilities.sum(axis=1, keepdims=True)
            resp_sum[resp_sum == 0] = 1e-10  # Avoid division by zero
            responsibilities = responsibilities / resp_sum
            
            # M-step: Update parameters
            old_means = means.copy()
            old_stds = stds.copy()
            
            for k in range(n_components):
                resp_k = responsibilities[:, k]
                n_k = resp_k.sum()
                
                if n_k > 1e-6:
                    weights[k] = n_k / len(scores)
                    means[k] = np.sum(resp_k * scores) / n_k
                    var_k = np.sum(resp_k * (scores - means[k])**2) / n_k
                    stds[k] = np.sqrt(max(var_k, 1e-6))  # Ensure positive
                else:
                    # Component has no support, reinitialize
                    means[k] = np.random.uniform(0.2, 0.8)
                    stds[k] = 0.2
            
            # Check convergence
            if np.max(np.abs(means - old_means)) < tolerance and np.max(np.abs(stds - old_stds)) < tolerance:
                break
    
    # Clamp means and stds to reasonable ranges
    means = np.clip(means, 0.0, 1.0)
    stds = np.clip(stds, 0.01, 0.5)
    weights = weights / weights.sum()  # Renormalize
    
    # Calculate overall statistics
    overall_mean = np.sum(weights * means)
    overall_var = np.sum(weights * (stds**2 + means**2)) - overall_mean**2
    overall_std = np.sqrt(overall_var)
    
    return {
        'n_components': n_components,
        'weights': weights.tolist(),
        'means': means.tolist(),
        'stds': stds.tolist(),
        'overall_mean': float(overall_mean),
        'overall_std': float(overall_std),
        'overall_var': float(overall_var)
    }


def visualize_distributions(
    scores: np.ndarray,
    beta_params: Tuple[float, float],
    normal_params: Dict[str, float],
    gmm_params: Dict[str, Any] = None,
    output_path: Path = None,
    bin_width: float = 0.05
) -> None:
    """
    Visualize histogram of scores with Gaussian Mixture Model overlay.
    
    Args:
        scores: Array of scores
        beta_params: Tuple of (alpha, beta) for Beta distribution (unused, kept for compatibility)
        normal_params: Dictionary with 'mean' and 'std' for Normal distribution (unused, kept for compatibility)
        gmm_params: Dictionary with GMM parameters (required)
        output_path: Path to save figure
        bin_width: Width of histogram bins (0.02 or 0.05)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram
    bins = np.arange(0, 1.01, bin_width)
    counts, bin_edges, patches = ax.hist(scores, bins=bins, alpha=0.7, edgecolor='black', 
                                         label='Observed Scores', density=True, color='#8DA0CB')
    
    # Overlay Gaussian Mixture Model if provided
    if gmm_params and gmm_params.get('n_components') == 2:
        weights = np.array(gmm_params['weights'])
        means = np.array(gmm_params['means'])
        stds = np.array(gmm_params['stds'])
        
        x = np.linspace(0.001, 0.999, 1000)
        dx = x[1] - x[0]
        
        # Compute mixture PDF
        y_gmm = np.zeros_like(x)
        for k in range(2):
            # Normal PDF for component k (truncated to [0,1])
            diff = x - means[k]
            component_pdf = weights[k] * np.exp(-0.5 * (diff / stds[k])**2) / (stds[k] * np.sqrt(2 * np.pi))
            # Truncate and normalize
            component_pdf = np.clip(component_pdf, 0, None)
            y_gmm += component_pdf
        
        # Normalize mixture
        area_gmm = np.sum(y_gmm) * dx
        if area_gmm > 0:
            y_gmm = y_gmm / area_gmm
        
        # Plot individual components
        for k in range(2):
            diff = x - means[k]
            comp_pdf = weights[k] * np.exp(-0.5 * (diff / stds[k])**2) / (stds[k] * np.sqrt(2 * np.pi))
            comp_pdf = np.clip(comp_pdf, 0, None)
            comp_area = np.sum(comp_pdf) * dx
            if comp_area > 0:
                comp_pdf = comp_pdf / comp_area
            ax.plot(x, comp_pdf, '--', linewidth=2, alpha=0.6, 
                   label=f'Component {k+1}: N(μ={means[k]:.3f}, σ={stds[k]:.3f}), w={weights[k]:.2f}',
                   color=['#FC8D62', '#66C2A5'][k])
        
        # Plot mixture
        ax.plot(x, y_gmm, 'g-', linewidth=3, label=f'GMM: w₁N(μ₁,σ₁) + w₂N(μ₂,σ₂)', alpha=0.9, color='#1F77B4')
    
    ax.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Random Agent System Scores\nGaussian Mixture Model (2 components)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved distribution visualization to {output_path}")
    plt.close()


def main():
    """Main function to fit distributions to data."""
    # Setup paths
    output_dir = Path("distribution_research/results")
    csv_path = output_dir / "random_graph_evaluations.csv"
    
    if not csv_path.exists():
        logger.error(f"Data file not found: {csv_path}")
        logger.error("Please run 'python distribution_research/generate_distribution_data.py' first")
        return
    
    logger.info("=" * 60)
    logger.info("PART 2: FITTING DISTRIBUTIONS TO DATA")
    logger.info("=" * 60)
    logger.info(f"Loading data from: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    graph_average_scores = df['average_score'].values
    
    logger.info(f"Loaded {len(graph_average_scores)} graph average scores")
    logger.info(f"  Score range: [{min(graph_average_scores):.4f}, {max(graph_average_scores):.4f}]")
    logger.info(f"  Mean: {np.mean(graph_average_scores):.4f}")
    logger.info(f"  Std: {np.std(graph_average_scores):.4f}")
    
    if len(graph_average_scores) < 3:
        logger.warning(f"⚠️  Only {len(graph_average_scores)} graphs - Distribution fits will be unreliable.")
    
    scores_array = np.array(graph_average_scores)
    
    # Fit Gaussian Mixture Model (2 components) - only model we're using
    logger.info(f"\nFitting Gaussian Mixture Model (2 components)...")
    logger.info(f"  Justification: Random graphs may form two populations:")
    logger.info(f"    - Well-structured graphs (with Solver, proper flow) → higher scores")
    logger.info(f"    - Poorly-structured graphs (missing key agents) → lower scores")
    
    gmm_params = fit_gaussian_mixture(scores_array, n_components=2)
    
    if gmm_params:
        logger.info(f"GMM Parameters:")
        logger.info(f"  Component 1: μ={gmm_params['means'][0]:.4f}, σ={gmm_params['stds'][0]:.4f}, weight={gmm_params['weights'][0]:.4f}")
        logger.info(f"  Component 2: μ={gmm_params['means'][1]:.4f}, σ={gmm_params['stds'][1]:.4f}, weight={gmm_params['weights'][1]:.4f}")
        logger.info(f"  Overall Mean: {gmm_params['overall_mean']:.4f}")
        logger.info(f"  Overall Std: {gmm_params['overall_std']:.4f}")
        logger.info(f"\n  Distribution equation: f(x) = w₁·N(μ₁,σ₁) + w₂·N(μ₂,σ₂)")
        logger.info(f"    where w₁={gmm_params['weights'][0]:.4f}, μ₁={gmm_params['means'][0]:.4f}, σ₁={gmm_params['stds'][0]:.4f}")
        logger.info(f"    and   w₂={gmm_params['weights'][1]:.4f}, μ₂={gmm_params['means'][1]:.4f}, σ₂={gmm_params['stds'][1]:.4f}")
    else:
        logger.warning("⚠️  Could not fit GMM (insufficient data)")
        gmm_params = None
    
    # Dummy parameters for compatibility with visualize_distributions signature
    alpha, beta = 1.0, 1.0
    normal_params = {'mean': 0.5, 'std': 0.2}
    
    # Save distribution parameters (only GMM now)
    dist_params = {
        'gmm': gmm_params if gmm_params else None,
        'num_samples': len(graph_average_scores),
        'num_graphs': len(df)
    }
    
    params_path = output_dir / "distribution_parameters.json"
    with open(params_path, 'w') as f:
        json.dump(dist_params, f, indent=2)
    logger.info(f"\n✓ Saved distribution parameters to {params_path}")
    
    # Visualize
    logger.info(f"\nCreating visualization...")
    score_range = np.max(scores_array) - np.min(scores_array) if len(scores_array) > 1 else 0.0
    bin_width = 0.02 if score_range < 0.5 else 0.05
    
    viz_path = output_dir / "score_distribution.png"
    visualize_distributions(
        scores_array, 
        (alpha, beta), 
        normal_params, 
        gmm_params, 
        viz_path, 
        bin_width=bin_width
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("DISTRIBUTION FITTING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Parameters saved to: {params_path}")
    logger.info(f"Visualization saved to: {viz_path}")


if __name__ == "__main__":
    main()

