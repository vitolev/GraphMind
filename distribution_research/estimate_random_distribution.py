"""
Distribution Research: Estimate the true distribution of random agent system scores.

This script:
1. Generates a large number of graphs (1,000,000)
2. Randomly samples N graphs (100 for full run, 2 for testing)
3. Evaluates each graph on 10 random problems
4. Saves results to CSV with graph structure
5. Fits Beta distribution using Bayesian estimation
6. Visualizes histogram with Beta distribution overlay
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import logging
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from config.settings import Config
from data_management.graph_storage import Graph, GraphSet
from graph_generation.graph_generation import _random_graph
from evaluation.math_solver import load_math_problems
from evaluation.llm_callers import set_llm_provider
from evaluation.llm_evaluator import evaluate_selected_graphs
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_large_graph_pool(num_graphs: int, max_depth: int, max_nodes: int, logger: logging.Logger) -> GraphSet:
    """
    Generate a large pool of random graphs.
    
    Args:
        num_graphs: Number of graphs to generate
        max_depth: Maximum depth for graph generation
        max_nodes: Maximum nodes for graph generation
        logger: Logger instance
    
    Returns:
        GraphSet containing generated graphs
    """
    logger.info(f"Generating {num_graphs:,} random graphs...")
    graph_pool = GraphSet()
    
    generated_count = 0
    attempts = 0
    max_attempts = num_graphs * 2  # Allow some duplicates
    
    start_time = time.time()
    
    while generated_count < num_graphs and attempts < max_attempts:
        attempts += 1
        try:
            graph = _random_graph(max_depth=max_depth, max_nodes=max_nodes)
            
            # Check if graph exceeds limits
            if len(graph.get_nodes()) > max_nodes:
                continue
            
            # Check for duplicates
            if graph_pool.contains(graph):
                continue
            
            graph_pool.add_graph(graph)
            generated_count += 1
            
            if generated_count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = generated_count / elapsed if elapsed > 0 else 0
                logger.info(f"  Generated {generated_count:,}/{num_graphs:,} graphs ({rate:.0f} graphs/sec)")
        
        except Exception as e:
            logger.warning(f"Error generating graph: {e}")
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Generated {graph_pool.size():,} unique graphs in {elapsed:.2f}s")
    
    return graph_pool


# Removed evaluate_graph_on_problems - now using evaluate_selected_graphs from main loop

def fit_beta_bayesian(scores: np.ndarray, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> Tuple[float, float]:
    """
    Fit Beta distribution using Bayesian estimation (conjugate prior).
    
    For Beta distribution with Beta(α, β) prior and observed data,
    the posterior is Beta(α + sum(log(x)), β + sum(log(1-x))).
    But for simplicity, we use method of moments with Bayesian smoothing.
    
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
    # Using weighted average (simplified approach)
    n = len(scores)
    # Weight prior less as we have more data
    prior_weight = 1.0 / (1.0 + n)
    data_weight = 1.0 - prior_weight
    
    posterior_alpha = prior_weight * prior_alpha + data_weight * alpha_mom
    posterior_beta = prior_weight * prior_beta + data_weight * beta_mom
    
    # Ensure positive
    posterior_alpha = max(0.1, posterior_alpha)
    posterior_beta = max(0.1, posterior_beta)
    
    return (posterior_alpha, posterior_beta)


def visualize_distribution(
    scores: np.ndarray,
    alpha: float,
    beta: float,
    output_path: Path,
    bin_width: float = 0.05
) -> None:
    """
    Visualize histogram of scores with Beta distribution overlay.
    
    Args:
        scores: Array of scores
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        output_path: Path to save figure
        bin_width: Width of histogram bins (0.02 or 0.05)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram
    bins = np.arange(0, 1.01, bin_width)
    counts, bin_edges, patches = ax.hist(scores, bins=bins, alpha=0.7, edgecolor='black', 
                                         label='Observed Scores', density=True)
    
    # Overlay Beta distribution
    # Beta PDF: x^(α-1) * (1-x)^(β-1) / B(α,β)
    # We'll compute it manually since scipy might not be available
    x = np.linspace(0.001, 0.999, 1000)
    
    # Compute log PDF
    log_pdf = (alpha - 1) * np.log(x + 1e-10) + (beta - 1) * np.log(1 - x + 1e-10)
    
    # Normalize (approximate Beta function using Stirling's approximation)
    # B(α,β) ≈ Γ(α)Γ(β)/Γ(α+β)
    # For normalization, we can use the fact that the integral should be 1
    # So we normalize by the area under the curve
    y = np.exp(log_pdf - np.max(log_pdf))  # Prevent overflow
    # Normalize to integrate to 1
    dx = x[1] - x[0]
    area = np.sum(y) * dx
    if area > 0:
        y = y / area
    
    ax.plot(x, y, 'r-', linewidth=3, label=f'Beta(α={alpha:.2f}, β={beta:.2f})')
    
    ax.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Random Agent System Scores\nwith Beta Distribution Fit', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    # Add statistics text
    mean_beta = alpha / (alpha + beta)
    var_beta = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    stats_text = f'Beta Distribution:\nMean: {mean_beta:.4f}\nStd: {np.sqrt(var_beta):.4f}\nα: {alpha:.4f}\nβ: {beta:.4f}'
    ax.text(0.7, 0.7, stats_text, transform=ax.transAxes, fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distribution visualization to {output_path}")


def main():
    """Main function to run distribution research."""
    # Load config
    config_path = Path("config/experiment_config.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    config = Config.from_yaml(config_path)
    
    # Override config for this research
    config.num_eval_problems = 10  # Set to 10 as requested
    config.max_nodes = 8
    config.max_depth = 3
    
    # Research parameters
    NUM_GRAPHS_TO_GENERATE = 10000  # Large pool
    NUM_GRAPHS_TO_SAMPLE = 2  # Start with 2 for testing, then 100
    NUM_PROBLEMS_PER_GRAPH = 10
    
    logger.info("=" * 60)
    logger.info("DISTRIBUTION RESEARCH: Random Agent System Score Distribution")
    logger.info("=" * 60)
    logger.info(f"Generating {NUM_GRAPHS_TO_GENERATE:,} graphs...")
    logger.info(f"Sampling {NUM_GRAPHS_TO_SAMPLE} random graphs for evaluation")
    logger.info(f"Evaluating each graph on {NUM_PROBLEMS_PER_GRAPH} random problems")
    logger.info("=" * 60)
    
    # Setup output directory
    output_dir = Path("distribution_research/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set LLM provider
    set_llm_provider(
        provider=getattr(config, 'llm_provider', 'groq'),
        local_model=getattr(config, 'local_llm_model', 'microsoft/Phi-3-mini-4k-instruct'),
        local_device=getattr(config, 'local_llm_device', 'auto'),
        ollama_model=getattr(config, 'ollama_model', 'llama3.2'),
        ollama_base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434')
    )
    
    # Load math problems
    logger.info("Loading math problems...")
    from evaluation.math_solver import load_math_problems
    math_problems = load_math_problems(config, logger)
    logger.info(f"Loaded {len(math_problems)} problems")
    
    # Step 1: Generate large graph pool
    graph_pool = generate_large_graph_pool(
        num_graphs=NUM_GRAPHS_TO_GENERATE,
        max_depth=config.max_depth,
        max_nodes=config.max_nodes,
        logger=logger
    )
    
    # Step 2: Randomly sample graphs
    logger.info(f"\nRandomly sampling {NUM_GRAPHS_TO_SAMPLE} graphs from pool...")
    all_graphs = graph_pool.get_all()
    sampled_graphs = random.sample(all_graphs, min(NUM_GRAPHS_TO_SAMPLE, len(all_graphs)))
    logger.info(f"✓ Sampled {len(sampled_graphs)} graphs")
    
    # Step 3: Evaluate sampled graphs using the same function as main loop
    logger.info(f"\nEvaluating {len(sampled_graphs)} graphs using main loop evaluation function...")
    
    # Convert sampled graphs to GraphSet
    sampled_graphset = GraphSet()
    for graph in sampled_graphs:
        sampled_graphset.add_graph(graph)
    
    # Use the same evaluation function as main loop
    evaluation_metrics, evaluated_graphs = evaluate_selected_graphs(
        config=config,
        logger=logger,
        selected_graphs=sampled_graphset,
        math_problems=math_problems
    )
    
    # Extract results from evaluated graphs
    results = []
    graph_average_scores = []  # Average score per graph (this is what we fit the distribution to)
    
    # Get per-graph metrics from evaluation_metrics
    per_graph_metrics = evaluation_metrics.get('per_graph_metrics', [])
    
    for graph_idx, graph in enumerate(evaluated_graphs.get_all()):
        graph_llm_score = graph.get_llm_score()  # This is already the average score
        graph_time = graph.get_time_evaluating()
        
        # Get problem scores from per_graph_metrics (for detailed logging/storage)
        problem_scores = []
        if graph_idx < len(per_graph_metrics):
            pg_metrics = per_graph_metrics[graph_idx]
            problem_scores = pg_metrics.get('problem_scores', [])
        else:
            # Fallback: use average score (shouldn't happen, but safety check)
            logger.warning(f"No per_graph_metrics for graph {graph_idx}, using average score")
            problem_scores = [graph_llm_score]
        
        # Store the average score per graph (this is what we'll fit distribution to)
        graph_average_scores.append(graph_llm_score)
        
        # Store results
        result = {
            'graph_id': graph_idx,
            'graph_nodes': json.dumps(graph.get_nodes()),
            'graph_edges': json.dumps(graph.get_edges()),
            'num_nodes': len(graph.get_nodes()),
            'num_edges': len(graph.get_edges()),
            'average_score': graph_llm_score,
            'problem_scores': json.dumps(problem_scores),
            'total_evaluation_time': graph_time,
            'num_problems_evaluated': len(problem_scores)
        }
        results.append(result)
        
        logger.info(f"[Graph {graph_idx + 1}/{len(sampled_graphs)}] "
                   f"Average score: {graph_llm_score:.4f}, "
                   f"Problems: {len(problem_scores)}, "
                   f"Scores: {[f'{s:.3f}' for s in problem_scores[:5]]}{'...' if len(problem_scores) > 5 else ''}")
    
    # Step 4: Save results to CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "random_graph_evaluations.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved results to {csv_path}")
    
    # Step 5: Fit Beta distribution (Bayesian) to average scores per graph
    # We fit ONE Beta distribution to ALL graph average scores together
    # This models the distribution of graph performance (not individual problem scores)
    logger.info(f"\nFitting Beta distribution to {len(graph_average_scores)} graph average scores...")
    logger.info(f"  Note: Fitting ONE distribution to all {len(graph_average_scores)} graph averages")
    logger.info(f"  Graph average scores: {[f'{s:.4f}' for s in graph_average_scores]}")
    
    if len(graph_average_scores) < 3:
        logger.warning(f"⚠️  Only {len(graph_average_scores)} graphs - Beta fit will be unreliable. "
                      f"Need at least 3-5 graphs for meaningful distribution estimation.")
    
    scores_array = np.array(graph_average_scores)
    alpha, beta = fit_beta_bayesian(scores_array, prior_alpha=1.0, prior_beta=1.0)
    
    mean_beta = alpha / (alpha + beta)
    var_beta = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    
    logger.info(f"Beta Distribution Parameters:")
    logger.info(f"  α (alpha): {alpha:.4f}")
    logger.info(f"  β (beta): {beta:.4f}")
    logger.info(f"  Mean: {mean_beta:.4f}")
    logger.info(f"  Std: {np.sqrt(var_beta):.4f}")
    logger.info(f"  Variance: {var_beta:.4f}")
    
    # Save distribution parameters
    dist_params = {
        'alpha': float(alpha),
        'beta': float(beta),
        'mean': float(mean_beta),
        'std': float(np.sqrt(var_beta)),
        'variance': float(var_beta),
        'num_samples': len(graph_average_scores),
        'num_graphs': len(sampled_graphs)
    }
    
    params_path = output_dir / "beta_distribution_parameters.json"
    with open(params_path, 'w') as f:
        json.dump(dist_params, f, indent=2)
    logger.info(f"✓ Saved distribution parameters to {params_path}")
    
    # Step 6: Visualize
    logger.info(f"\nCreating visualization...")
    # Determine bin width based on score range
    score_range = np.max(scores_array) - np.min(scores_array)
    bin_width = 0.02 if score_range < 0.5 else 0.05
    
    viz_path = output_dir / "score_distribution.png"
    visualize_distribution(scores_array, alpha, beta, viz_path, bin_width=bin_width)
    
    logger.info("\n" + "=" * 60)
    logger.info("DISTRIBUTION RESEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Evaluations: {csv_path}")
    logger.info(f"  - Parameters: {params_path}")
    logger.info(f"  - Visualization: {viz_path}")


if __name__ == "__main__":
    main()

