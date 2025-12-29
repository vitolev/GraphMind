"""
Create a Q-Q (Quantile-Quantile) plot comparing GNN models against random baseline.

This script creates a comprehensive visualization showing how each model's 
score distribution compares to the random baseline distribution.
It loads actual score distributions from CSV files and training datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high DPI for Medium-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'


def load_random_scores(csv_path: Path) -> np.ndarray:
    """Load random baseline scores from CSV."""
    logger.info(f"Loading random scores from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'average_score' not in df.columns:
        raise ValueError(f"No 'average_score' column found in {csv_path}")
    
    scores = df['average_score'].values
    logger.info(f"  Loaded {len(scores)} random graph scores")
    logger.info(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    return scores


def load_gnn_scores(experiment_name: str) -> np.ndarray:
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
        # Import GraphSet for pickle to work
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data_management.graph_storage import GraphSet
        
        with open(dataset_path, 'rb') as f:
            training_dataset = pickle.load(f)
        
        # Extract scores from Graph objects
        scores = []
        for graph in training_dataset.graphs:  # Direct attribute access
            llm_score = graph.llm_score  # Direct attribute access
            if llm_score >= 0:  # Only include evaluated graphs
                scores.append(llm_score)
        
        scores = np.array(scores)
        
        if len(scores) == 0:
            logger.warning(f"No evaluated graphs found for {experiment_name}!")
            return np.array([])
        
        logger.info(f"  Loaded {len(scores)} GNN-guided graph scores")
        
        # Use only the last 150 samples (most recent evaluations)
        if len(scores) > 150:
            scores = scores[-150:]
            logger.info(f"  Using last 150 samples (most recent evaluations)")
        
        logger.info(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        logger.info(f"  Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error loading {experiment_name} dataset: {e}")
        return np.array([])


def create_qq_plot(random_scores: np.ndarray, model_scores_dict: dict, output_path: Path):
    """
    Create a Q-Q plot comparing model quantiles against random baseline.
    
    Args:
        random_scores: Array of random baseline scores
        model_scores_dict: Dictionary mapping model names to score arrays
        output_path: Path to save the plot
    """
    # Compute quantiles from actual score distributions - use 100 percentiles for smooth curve
    percentiles = np.arange(0, 101, 1)  # 0, 1, 2, ..., 100 (101 points)
    
    # Compute quantiles for random baseline
    random_quantiles = np.percentile(random_scores, percentiles)
    
    # Ensure min is 0 and max is 1 (clip to [0, 1] range)
    random_quantiles[0] = max(0.0, np.min(random_scores))
    random_quantiles[-1] = min(1.0, np.max(random_scores))
    
    models = list(model_scores_dict.keys())
    
    # Colors for each model
    colors = {
        'GAT': '#E74C3C',  # Red
        'GCN': '#3498DB',  # Blue
        'SAGE': '#2ECC71'  # Green
    }
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot diagonal line (random baseline vs itself - perfect match)
    min_val = min(random_quantiles.min(), 0)
    max_val = max(random_quantiles.max(), 1.0)
    ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', linewidth=2, alpha=0.5, label='Random Baseline (diagonal)')
    
    # Plot Q-Q curves for each model
    for model in models:
        if model in model_scores_dict and len(model_scores_dict[model]) > 0:
            model_scores = model_scores_dict[model]
            
            # Compute quantiles for model
            model_quantiles = np.percentile(model_scores, percentiles)
            
            # Ensure min is 0 and max is 1 (clip to [0, 1] range)
            model_quantiles[0] = max(0.0, np.min(model_scores))
            model_quantiles[-1] = min(1.0, np.max(model_scores))
            
            # Plot the Q-Q curve (smooth line without markers for 100 points)
            ax.plot(random_quantiles, model_quantiles, 
                   '-', linewidth=2.5,
                   label=model, color=colors.get(model, '#95A5A6'),
                   alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Random Baseline Quantiles', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model Quantiles', fontsize=14, fontweight='bold')
    ax.set_title('Q-Q Plot: Model Score Distributions vs Random Baseline\n(Higher values indicate better performance)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # Add legend
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved Q-Q plot to: {output_path}")
    
    # Print summary statistics
    logger.info("\nQuantile Comparison Summary:")
    logger.info("=" * 60)
    # P50 is at index 50 (0-indexed, since percentiles are 0, 1, 2, ..., 100)
    random_p50 = random_quantiles[50]
    for model in models:
        if model in model_scores_dict and len(model_scores_dict[model]) > 0:
            model_scores = model_scores_dict[model]
            model_quantiles = np.percentile(model_scores, percentiles)
            model_p50 = model_quantiles[50]
            mean_val = np.mean(model_scores)
            
            logger.info(f"\n{model}:")
            logger.info(f"  Mean: {mean_val:.4f}")
            logger.info(f"  All quantiles above diagonal: {np.all(model_quantiles > random_quantiles)}")
            logger.info(f"  Median (P50): {model_p50:.4f} vs Random {random_p50:.4f}")


def create_percentile_comparison_plot(random_scores: np.ndarray, model_scores_dict: dict, output_path: Path):
    """
    Create an alternative visualization showing percentile bars for comparison.
    
    Args:
        random_scores: Array of random baseline scores
        model_scores_dict: Dictionary mapping model names to score arrays
        output_path: Path to save the plot
    """
    # Extract quantile values - use subset for bar plot display
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_labels = [f'P{p}' for p in percentiles]
    
    # Compute quantiles
    random_quantiles = np.percentile(random_scores, percentiles)
    
    models = ['Random Baseline'] + list(model_scores_dict.keys())
    colors_list = ['#8DA0CB', '#E74C3C', '#3498DB', '#2ECC71']
    
    # Prepare data
    data_dict = {'Random Baseline': random_quantiles}
    for model, scores in model_scores_dict.items():
        if len(scores) > 0:
            data_dict[model] = np.percentile(scores, percentiles)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(percentile_labels))
    width = 0.2
    
    # Create bars for each model
    for i, model in enumerate(models):
        if model in data_dict:
            values = data_dict[model]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, 
                  color=colors_list[i % len(colors_list)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Percentile', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Percentile Comparison: Random Baseline vs GNN Models', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(percentile_labels)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', axis='y')
    ax.set_ylim([0, 1.0])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved percentile comparison plot to: {output_path}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CREATING Q-Q PLOTS FROM ACTUAL DISTRIBUTIONS")
    logger.info("=" * 60)
    
    # Input and output paths
    random_csv_path = Path("distribution_research/results/random_graph_evaluations.csv")
    output_dir = Path("reports/distribution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load random baseline scores
    random_scores = load_random_scores(random_csv_path)
    
    if len(random_scores) == 0:
        logger.error("Failed to load random baseline scores!")
        exit(1)
    
    # Load GNN model scores
    experiments = ['gat', 'gcn', 'sage']
    model_scores_dict = {}
    
    for exp_name in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading {exp_name.upper()} scores")
        logger.info(f"{'='*60}")
        scores = load_gnn_scores(exp_name)
        if len(scores) > 0:
            model_scores_dict[exp_name.upper()] = scores
    
    if len(model_scores_dict) == 0:
        logger.error("No GNN model scores loaded!")
        exit(1)
    
    # Create Q-Q plot
    logger.info(f"\n{'='*60}")
    logger.info("Creating Q-Q plot")
    logger.info(f"{'='*60}")
    qq_output_path = output_dir / "qq_plot_comparison.png"
    create_qq_plot(random_scores, model_scores_dict, qq_output_path)
    
    # Create alternative percentile comparison plot
    logger.info(f"\n{'='*60}")
    logger.info("Creating percentile bar comparison plot")
    logger.info(f"{'='*60}")
    percentile_output_path = output_dir / "percentile_bar_comparison.png"
    create_percentile_comparison_plot(random_scores, model_scores_dict, percentile_output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL PLOTS COMPLETE")
    logger.info("=" * 60)

