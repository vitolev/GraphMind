"""
Create comparison histogram: Random Sampling vs GNN-Guided Algorithm

This script:
1. Loads random graph scores from distribution_research
2. Loads GNN-guided scores from training dataset
3. Creates a comparison histogram showing both distributions
4. Saves high-quality image for Medium blog post
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
from pathlib import Path
from config.settings import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high DPI for Medium-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

def load_random_scores(csv_path: Path) -> np.ndarray:
    """Load random graph scores from CSV."""
    logger.info(f"Loading random scores from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'average_score' not in df.columns:
        raise ValueError(f"No 'average_score' column found in {csv_path}")
    
    scores = df['average_score'].values
    logger.info(f"  Loaded {len(scores)} random graph scores")
    logger.info(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    logger.info(f"  Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
    
    return scores


def load_gnn_guided_scores(config: Config) -> np.ndarray:
    """
    Load GNN-guided graph scores from training dataset.
    
    The training dataset contains all graphs that were:
    1. Selected by GNN predictions (Step 3)
    2. Evaluated with LLMs (Step 4) 
    3. Added to training dataset (Step 5)
    
    This represents all graphs evaluated in the GNN-guided loop.
    """
    logger.info("Loading GNN-guided scores from training dataset...")
    logger.info("  Source: data/training_dataset.pkl")
    logger.info("  This contains all graphs evaluated during the GNN-guided loop")
    
    dataset_path = config.data_dir / "training_dataset.pkl"
    
    if not dataset_path.exists():
        logger.warning(f"Training dataset not found at {dataset_path}!")
        logger.info("  Alternative: Try extracting from all_iterations_data.csv")
        return np.array([])
    
    try:
        with open(dataset_path, 'rb') as f:
            training_dataset = pickle.load(f)
        
        # Extract scores from Graph objects
        scores = []
        for graph in training_dataset.get_all():
            llm_score = graph.get_llm_score()
            if llm_score >= 0:  # Only include evaluated graphs (exclude unevaluated)
                scores.append(llm_score)
        
        scores = np.array(scores)
        
        if len(scores) == 0:
            logger.warning("No evaluated graphs found in training dataset!")
            return np.array([])
        
        logger.info(f"  Loaded {len(scores)} GNN-guided graph scores")
        
        # Take only the last 150 samples (most recent evaluations)
        if len(scores) > 150:
            scores = scores[-150:]
            logger.info(f"  Using last 150 samples (most recent evaluations)")
        
        logger.info(f"  Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        logger.info(f"  Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
        logger.info(f"  These are scores from graphs selected by GNN and evaluated with LLMs")
        
        return scores
    except Exception as e:
        logger.error(f"Error loading training dataset: {e}")
        return np.array([])


def create_comparison_histogram(
    random_scores: np.ndarray,
    gnn_scores: np.ndarray,
    output_path: Path
) -> None:
    """Create comparison histogram of random vs GNN-guided scores."""
    
    # Set style for blog-post quality
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Determine bin edges (use same bins for both distributions)
    all_scores = np.concatenate([random_scores, gnn_scores])
    min_score = min(np.min(random_scores), np.min(gnn_scores))
    max_score = max(np.max(random_scores), np.max(gnn_scores))
    
    # Use 30 bins, but ensure they cover the full range
    bins = np.linspace(min_score, max_score, 31)
    
    # Plot histograms (overlaid, semi-transparent)
    n1, bins1, patches1 = ax.hist(
        random_scores, 
        bins=bins, 
        alpha=0.6, 
        edgecolor='black', 
        linewidth=1.5,
        label=f'Random Sampling (n={len(random_scores)})',
        color='#E74C3C',  # Red
        density=False  # Show counts, not density
    )
    
    n2, bins2, patches2 = ax.hist(
        gnn_scores, 
        bins=bins, 
        alpha=0.6, 
        edgecolor='black', 
        linewidth=1.5,
        label=f'GNN-Guided Algorithm (n={len(gnn_scores)})',
        color='#3498DB',  # Blue
        density=False  # Show counts, not density
    )
    
    # Add vertical lines for means
    random_mean = np.mean(random_scores)
    gnn_mean = np.mean(gnn_scores)
    
    ax.axvline(random_mean, color='#E74C3C', linestyle='--', linewidth=2.5, 
               label=f'Random Mean: {random_mean:.3f}', alpha=0.8)
    ax.axvline(gnn_mean, color='#3498DB', linestyle='--', linewidth=2.5, 
               label=f'GNN-Guided Mean: {gnn_mean:.3f}', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Average Score', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Graphs', fontsize=16, fontweight='bold')
    ax.set_title('Score Distribution Comparison: Random Sampling vs GNN-Guided Algorithm', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis limits
    ax.set_xlim([min_score - 0.05, max_score + 0.05])
    
    # Calculate improvement for logging
    improvement = ((gnn_mean - random_mean) / random_mean) * 100
    logger.info(f"  Improvement: {improvement:+.1f}%")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved comparison histogram to: {output_path}")
    logger.info(f"  Random: mean={random_mean:.4f}, n={len(random_scores)}")
    logger.info(f"  GNN-Guided: mean={gnn_mean:.4f}, n={len(gnn_scores)}")
    logger.info(f"  Improvement: {improvement:+.1f}%")


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("CREATING DISTRIBUTION COMPARISON")
    logger.info("=" * 60)
    
    # Load config
    config_path = Path("config/experiment_config.yaml")
    config = Config.from_yaml(config_path)
    
    # Find random scores file (use the most recent one)
    random_csv_paths = [
        Path("distribution_research/results/random_graph_evaluations.csv"),
        Path("distribution_research/results1/random_graph_evaluations.csv"),
        Path("distribution_research/results0/random_graph_evaluations.csv"),
    ]
    
    random_csv_path = None
    for path in random_csv_paths:
        if path.exists():
            random_csv_path = path
            break
    
    if random_csv_path is None:
        logger.error("No random graph evaluations CSV found!")
        logger.error("Please run distribution_research/generate_distribution_data.py first")
        return
    
    # Load scores
    random_scores = load_random_scores(random_csv_path)
    gnn_scores = load_gnn_guided_scores(config)
    
    if len(gnn_scores) == 0:
        logger.error("No GNN-guided scores found in training dataset!")
        return
    
    # Create output directory
    output_dir = Path("reports/blog_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison histogram
    output_path = output_dir / "random_vs_gnn_comparison.png"
    create_comparison_histogram(random_scores, gnn_scores, output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

