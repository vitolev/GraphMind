"""
Script to generate all images needed for the Medium blog post.

Run this script to create all visualizations in high quality (300 DPI) for Medium.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import logging
from config.settings import Config
from post_processing.diagnostics import (
    create_diagnostic_report,
    visualize_rmse_trends,
    visualize_predictions_vs_actual,
    visualize_best_graphs
)
from data_management.graph_storage import load_training_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high DPI for Medium-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

def main():
    """Generate all blog post images."""
    
    # Load config
    config_path = Path("config/experiment_config.yaml")
    config = Config.from_yaml(config_path)
    
    # Output directory for blog images
    blog_images_dir = Path("reports/blog_images")
    blog_images_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GENERATING BLOG POST IMAGES")
    logger.info("=" * 60)
    
    # 1. RMSE Trends (already blog-post ready)
    logger.info("\n1. Creating RMSE trends visualization...")
    try:
        rmse_path = visualize_rmse_trends(config, blog_images_dir, logger)
        if rmse_path:
            logger.info(f"   ✓ Saved: {rmse_path}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # 2. Predictions vs Actual scatter plot
    logger.info("\n2. Creating predictions vs actual scatter plot...")
    try:
        training_dataset = load_training_dataset(config.data_dir, logger)
        if training_dataset.size() > 0:
            pred_path = visualize_predictions_vs_actual(
                training_dataset, blog_images_dir, logger, top_n=20
            )
            logger.info(f"   ✓ Saved: {pred_path}")
        else:
            logger.warning("   ⚠ Training dataset is empty")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # 3. Best graphs visualization
    logger.info("\n3. Creating best graphs visualization...")
    try:
        if training_dataset.size() > 0:
            best_graphs_path = visualize_best_graphs(
                training_dataset, blog_images_dir, logger, top_n=10
            )
            logger.info(f"   ✓ Saved: {best_graphs_path}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    # 4. Distribution comparison (if distribution research results exist)
    logger.info("\n4. Creating distribution comparison...")
    try:
        dist_research_path = Path("distribution_research/results/random_graph_evaluations.csv")
        if dist_research_path.exists():
            create_distribution_comparison(dist_research_path, blog_images_dir, logger)
        else:
            logger.warning(f"   ⚠ Distribution research data not found at {dist_research_path}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("BLOG IMAGE GENERATION COMPLETE")
    logger.info(f"All images saved to: {blog_images_dir}")
    logger.info("=" * 60)


def create_distribution_comparison(csv_path: Path, output_dir: Path, logger: logging.Logger):
    """Create distribution comparison visualization."""
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(csv_path)
    
    if 'average_score' not in df.columns:
        logger.warning("   ⚠ No 'average_score' column in distribution data")
        return
    
    scores = df['average_score'].values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Histogram
    ax.hist(scores, bins=30, alpha=0.7, edgecolor='black', density=True, label='Random Graph Scores')
    
    # Add distribution fits if available
    beta_params_path = Path("distribution_research/results/beta_distribution_parameters.json")
    if beta_params_path.exists():
        import json
        with open(beta_params_path) as f:
            params = json.load(f)
            if 'beta' in params:
                alpha = params['beta']['alpha']
                beta = params['beta']['beta']
                
                # Overlay Beta distribution
                x = np.linspace(0, 1, 1000)
                from scipy.stats import beta as beta_dist
                y = beta_dist.pdf(x, alpha, beta)
                ax.plot(x, y, 'r-', linewidth=3, label=f'Beta(α={alpha:.2f}, β={beta:.2f})', alpha=0.7)
    
    ax.set_xlabel('Score', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Random Graph Score Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / "distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   ✓ Saved: {output_path}")


if __name__ == "__main__":
    main()

