"""
Create GMM-based distribution comparisons: Random Baseline vs GNN-Guided Experiments

This script:
1. Loads random baseline scores and fits GMM
2. Loads GNN-guided scores from different experiments (GAT, GCN, SAGE)
3. Fits GMM to each GNN-guided distribution
4. Creates comparison visualizations showing GMM density functions
5. Saves high-quality images for Medium blog post
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import json
from typing import Dict, Any, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high DPI for Medium-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

# Optional import for scipy/sklearn (for GMM)
try:
    from sklearn.mixture import GaussianMixture
    HAS_SCIPY_SKLEARN = True
except ImportError:
    HAS_SCIPY_SKLEARN = False
    logger.warning("Scikit-learn not found. Install with 'pip install scikit-learn' for better GMM fitting.")


def fit_gaussian_mixture(scores: np.ndarray, n_components: int = 2) -> Optional[Dict[str, Any]]:
    """
    Fit a Gaussian Mixture Model (mixture of normal distributions) to scores.
    
    Args:
        scores: Array of scores in [0, 1]
        n_components: Number of Gaussian components (default: 2)
    
    Returns:
        Dictionary with mixture model parameters, or None if fitting fails
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
            logger.warning(f"Sklearn GMM fitting failed: {e}, using fallback")
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


def compute_gmm_pdf(x: np.ndarray, gmm_params: Dict[str, Any]) -> np.ndarray:
    """
    Compute the probability density function of a Gaussian Mixture Model.
    
    Args:
        x: Array of x values at which to evaluate PDF
        gmm_params: Dictionary with 'weights', 'means', 'stds'
    
    Returns:
        Array of PDF values
    """
    weights = np.array(gmm_params['weights'])
    means = np.array(gmm_params['means'])
    stds = np.array(gmm_params['stds'])
    
    # Compute mixture PDF
    y = np.zeros_like(x)
    for k in range(len(weights)):
        # Normal PDF for component k
        diff = x - means[k]
        component_pdf = weights[k] * np.exp(-0.5 * (diff / stds[k])**2) / (stds[k] * np.sqrt(2 * np.pi))
        # Truncate and ensure positive
        component_pdf = np.clip(component_pdf, 0, None)
        y += component_pdf
    
    # Normalize (since we're truncating to [0,1])
    dx = x[1] - x[0] if len(x) > 1 else 0.001
    area = np.sum(y) * dx
    if area > 0:
        y = y / area
    
    return y


def load_random_baseline() -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Load random baseline scores and GMM parameters.
    
    Returns:
        Tuple of (scores, gmm_params)
    """
    # Try to load from CSV
    csv_path = Path("distribution_research/results/random_graph_evaluations.csv")
    if not csv_path.exists():
        logger.error(f"Random baseline CSV not found: {csv_path}")
        return np.array([]), None
    
    logger.info(f"Loading random baseline from: {csv_path}")
    df = pd.read_csv(csv_path)
    scores = df['average_score'].values
    
    logger.info(f"  Loaded {len(scores)} random graph scores")
    logger.info(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    # Load pre-computed GMM parameters
    params_path = Path("distribution_research/results/distribution_parameters.json")
    gmm_params = None
    if params_path.exists():
        try:
            with open(params_path, 'r') as f:
                dist_params = json.load(f)
            gmm_params = dist_params.get('gmm')
            if gmm_params:
                logger.info("  Loaded pre-computed GMM parameters from distribution_research/results/distribution_parameters.json")
            else:
                logger.warning("  GMM parameters not found in distribution_parameters.json")
        except Exception as e:
            logger.warning(f"  Could not load GMM parameters: {e}")
    else:
        logger.warning(f"  GMM parameters file not found: {params_path}")
    
    # Fit GMM if not available
    if gmm_params is None:
        logger.info("  Fitting GMM to random baseline scores...")
        gmm_params = fit_gaussian_mixture(scores, n_components=2)
        if gmm_params:
            logger.info(f"  GMM fitted: μ₁={gmm_params['means'][0]:.3f}, μ₂={gmm_params['means'][1]:.3f}")
    
    return scores, gmm_params


def load_gnn_guided_scores(experiment_name: str) -> np.ndarray:
    """
    Load GNN-guided scores from an experiment's training dataset.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'gat', 'gcn', 'sage')
    
    Returns:
        Array of scores
    """
    # Map experiment names to paths
    experiment_paths = {
        'gat': Path("final-experiment-gat/training_dataset.pkl"),
        'gcn': Path("final-experiment-gcn/training_dataset.pkl"),
        'sage': Path("final-experiment-sage/training_dataset.pkl"),
    }
    
    dataset_path = experiment_paths.get(experiment_name.lower())
    if not dataset_path or not dataset_path.exists():
        logger.warning(f"Training dataset not found for {experiment_name}: {dataset_path}")
        return np.array([])
    
    logger.info(f"Loading GNN-guided scores from: {dataset_path}")
    
    try:
        # Import necessary modules for pickle to work
        # Note: This will fail if torch_geometric is not installed, which is expected
        import sys
        import os
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Import GraphSet - this may fail if torch_geometric is not available
        # but we need it for pickle to work
        try:
            from data_management.graph_storage import GraphSet
        except ImportError as e:
            logger.error(f"Failed to import GraphSet (torch_geometric may not be installed): {e}")
            logger.error("Please install torch_geometric to load training datasets")
            return np.array([])
        
        with open(dataset_path, 'rb') as f:
            training_dataset = pickle.load(f)
        
        # Extract scores from Graph objects
        # Access graphs directly via .graphs attribute to avoid method calls
        scores = []
        for graph in training_dataset.graphs:  # Direct attribute access
            llm_score = graph.llm_score  # Direct attribute access (Graph.llm_score)
            if llm_score >= 0:  # Only include evaluated graphs
                scores.append(llm_score)
        
        scores = np.array(scores)
        
        if len(scores) == 0:
            logger.warning(f"No evaluated graphs found for {experiment_name}!")
            return np.array([])
        
        logger.info(f"  Loaded {len(scores)} GNN-guided graph scores")
        logger.info(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        logger.info(f"  Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
        
        # Use only the last 150 samples (most recent evaluations)
        if len(scores) > 150:
            scores = scores[-150:]
            logger.info(f"  Using last 150 samples (most recent evaluations)")
        
        # Don't fit GMM for GNN scores - just return scores
        return scores
        
    except Exception as e:
        logger.error(f"Error loading {experiment_name} dataset: {e}")
        return np.array([])


def create_gmm_comparison(
    random_scores: np.ndarray,
    random_gmm: Optional[Dict[str, Any]],
    gnn_scores: np.ndarray,
    experiment_name: str,
    output_path: Path
) -> None:
    """
    Create comparison visualization: Random with GMM density vs GNN-guided histogram.
    
    Args:
        random_scores: Random baseline scores (for histogram background)
        random_gmm: GMM parameters for random baseline (final mixture only)
        gnn_scores: GNN-guided scores (histogram only, no GMM)
        experiment_name: Name of the experiment (for title)
        output_path: Path to save figure
    """
    # Set style for blog-post quality
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bins for histograms
    bins = np.linspace(0, 1, 40)
    
    # Plot random histogram (light background)
    ax.hist(random_scores, bins=bins, alpha=0.2, edgecolor='gray', linewidth=0.5,
            label=f'Random Sampling (n={len(random_scores)})', color='#8DA0CB', density=True)
    
    # Plot GNN-guided histogram (more prominent)
    ax.hist(gnn_scores, bins=bins, alpha=0.5, edgecolor='black', linewidth=1,
            label=f'GNN-Guided ({experiment_name.upper()}) (n={len(gnn_scores)})', 
            color='#FC8D62', density=True)
    
    # Prepare x-axis for GMM PDF
    x = np.linspace(0.001, 0.999, 1000)
    
    # Plot only the final GMM density for random (no individual components)
    if random_gmm:
        y_random = compute_gmm_pdf(x, random_gmm)
        ax.plot(x, y_random, linewidth=3, 
               label='Random GMM (fitted)', 
               alpha=0.9, color='#8DA0CB')
    
    # Add vertical lines for means
    random_mean = np.mean(random_scores)
    gnn_mean = np.mean(gnn_scores)
    
    ax.axvline(random_mean, color='#8DA0CB', linestyle=':', linewidth=2, 
               label=f'Random Mean: {random_mean:.3f}', alpha=0.7)
    ax.axvline(gnn_mean, color='#FC8D62', linestyle=':', linewidth=2,
               label=f'GNN-Guided Mean: {gnn_mean:.3f}', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Average Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title(f'Distribution Comparison: Random Baseline vs GNN-Guided ({experiment_name.upper()})', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # Set x-axis limits
    ax.set_xlim([0, 1])
    
    # Calculate improvement
    improvement = ((gnn_mean - random_mean) / random_mean) * 100 if random_mean > 0 else 0
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved comparison to: {output_path}")
    logger.info(f"  Random: mean={random_mean:.4f}, n={len(random_scores)}")
    logger.info(f"  GNN-Guided ({experiment_name.upper()}): mean={gnn_mean:.4f}, n={len(gnn_scores)}")
    logger.info(f"  Improvement: {improvement:+.1f}%")


def create_all_models_comparison(
    all_scores_dict: Dict[str, np.ndarray],
    output_path: Path
) -> None:
    """
    Create a combined comparison plot showing all GNN models together.
    
    Args:
        all_scores_dict: Dictionary mapping model names to score arrays
        output_path: Path to save figure
    """
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors for each model
    colors = {
        'GAT': '#E74C3C',  # Red
        'GCN': '#3498DB',  # Blue
        'SAGE': '#2ECC71'  # Green
    }
    
    bins = np.linspace(0, 1, 40)
    
    # Plot histograms for each model
    for model_name, scores in all_scores_dict.items():
        if len(scores) > 0:
            ax.hist(scores, bins=bins, alpha=0.5, edgecolor='black', linewidth=1,
                   label=f'{model_name} (n={len(scores)}, μ={np.mean(scores):.3f})', 
                   color=colors.get(model_name, '#95A5A6'), density=True)
    
    # Add vertical lines for means
    for model_name, scores in all_scores_dict.items():
        if len(scores) > 0:
            mean_val = np.mean(scores)
            ax.axvline(mean_val, color=colors.get(model_name, '#95A5A6'), 
                      linestyle=':', linewidth=2, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Average Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Distribution Comparison: All GNN Models', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # Set x-axis limits
    ax.set_xlim([0, 1])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved all models comparison to: {output_path}")


def create_percentiles_table(
    random_scores: np.ndarray,
    all_scores_dict: Dict[str, np.ndarray],
    output_path: Path
) -> pd.DataFrame:
    """
    Create a table with percentiles for all models.
    
    Args:
        random_scores: Random baseline scores
        all_scores_dict: Dictionary mapping model names to score arrays
        output_path: Path to save CSV file
    
    Returns:
        DataFrame with percentiles
    """
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    
    data = {}
    
    # Add random baseline
    if len(random_scores) > 0:
        data['Random Baseline'] = [
            np.percentile(random_scores, p) for p in percentiles
        ]
    
    # Add each model
    for model_name, scores in all_scores_dict.items():
        if len(scores) > 0:
            data[model_name] = [
                np.percentile(scores, p) for p in percentiles
            ]
    
    # Create DataFrame
    df = pd.DataFrame(data, index=[f'P{p}' for p in percentiles])
    
    # Add mean and std
    if len(random_scores) > 0:
        df.loc['Mean', 'Random Baseline'] = np.mean(random_scores)
        df.loc['Std', 'Random Baseline'] = np.std(random_scores)
    
    for model_name, scores in all_scores_dict.items():
        if len(scores) > 0:
            df.loc['Mean', model_name] = np.mean(scores)
            df.loc['Std', model_name] = np.std(scores)
    
    # Save to CSV
    df.to_csv(output_path)
    logger.info(f"✓ Saved percentiles table to: {output_path}")
    
    return df


def create_box_plot_comparison(
    random_scores: np.ndarray,
    all_scores_dict: Dict[str, np.ndarray],
    output_path: Path
) -> None:
    """
    Create a box plot with whiskers comparing all models.
    
    Args:
        random_scores: Random baseline scores
        all_scores_dict: Dictionary mapping model names to score arrays
        output_path: Path to save figure
    """
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    
    # Add random baseline
    if len(random_scores) > 0:
        data_to_plot.append(random_scores)
        labels.append('Random Baseline')
    
    # Add each model
    for model_name, scores in all_scores_dict.items():
        if len(scores) > 0:
            data_to_plot.append(scores)
            labels.append(model_name)
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, patch_artist=True, 
                    notch=True, showmeans=True)
    ax.set_xticklabels(labels)
    
    # Color boxes
    colors_list = ['#8DA0CB', '#E74C3C', '#3498DB', '#2ECC71']
    for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Labels and title
    ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
    ax.set_title('Score Distribution: Box Plot Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', axis='y')
    
    # Set y-axis limits
    ax.set_ylim([0, 1])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved box plot comparison to: {output_path}")


def main():
    """Main function to create all distribution comparisons."""
    logger.info("=" * 60)
    logger.info("CREATING GMM-BASED DISTRIBUTION COMPARISONS")
    logger.info("=" * 60)
    
    # Load random baseline
    random_scores, random_gmm = load_random_baseline()
    
    if len(random_scores) == 0 or random_gmm is None:
        logger.error("Failed to load random baseline! Cannot proceed.")
        return
    
    # Create output directory
    output_dir = Path("reports/distribution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiments to compare
    experiments = ['gat', 'gcn', 'sage']
    all_scores_dict = {}
    
    # Create comparison for each experiment
    for exp_name in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {exp_name.upper()} experiment")
        logger.info(f"{'='*60}")
        
        gnn_scores = load_gnn_guided_scores(exp_name)
        
        if len(gnn_scores) == 0:
            logger.warning(f"Skipping {exp_name} - no data")
            continue
        
        # Store scores for combined comparison
        all_scores_dict[exp_name.upper()] = gnn_scores
        
        # Create comparison visualization (no GMM for GNN scores)
        output_path = output_dir / f"random_vs_{exp_name}_gmm_comparison.png"
        create_gmm_comparison(
            random_scores, random_gmm,
            gnn_scores,
            exp_name, output_path
        )
    
    # Create combined comparison of all models
    if len(all_scores_dict) > 0:
        logger.info(f"\n{'='*60}")
        logger.info("Creating combined comparison of all models")
        logger.info(f"{'='*60}")
        
        combined_output_path = output_dir / "all_models_comparison.png"
        create_all_models_comparison(all_scores_dict, combined_output_path)
        
        # Create percentiles table
        percentiles_output_path = output_dir / "percentiles_table.csv"
        percentiles_df = create_percentiles_table(random_scores, all_scores_dict, percentiles_output_path)
        logger.info("\nPercentiles Table:")
        print(percentiles_df.to_string())
        
        # Create box plot comparison
        boxplot_output_path = output_dir / "boxplot_comparison.png"
        create_box_plot_comparison(random_scores, all_scores_dict, boxplot_output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL COMPARISONS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

