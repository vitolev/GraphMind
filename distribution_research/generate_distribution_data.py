"""
Part 1: Generate Distribution Research Data

This script:
1. Generates a large number of random graphs (configurable)
2. Randomly samples N graphs (configurable)
3. Evaluates each graph on M random problems (configurable)
4. Saves results to CSV with graph structure and scores
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
        max_nodes: Maximum nodes per graph
        logger: Logger instance
    
    Returns:
        GraphSet containing unique graphs
    """
    graph_pool = GraphSet()
    start_time = time.time()
    generated_count = 0
    
    logger.info(f"Generating {num_graphs:,} random graphs...")
    
    while graph_pool.size() < num_graphs:
        try:
            graph = _random_graph(max_depth=max_depth, max_nodes=max_nodes)
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


def main():
    """Main function to generate distribution research data."""
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
    NUM_GRAPHS_TO_GENERATE = 5000000  # Large pool
    NUM_GRAPHS_TO_SAMPLE = 150  # Number of graphs to evaluate
    NUM_PROBLEMS_PER_GRAPH = 5  # Problems per graph
    
    logger.info("=" * 60)
    logger.info("PART 1: GENERATING DISTRIBUTION RESEARCH DATA")
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
    
    # Step 4: Extract results and save to CSV
    logger.info(f"\nExtracting results and saving to CSV...")
    results = []
    graph_average_scores = []
    
    for graph in evaluated_graphs.get_all():
        graph_llm_score = graph.get_llm_score()
        graph_time = graph.get_time_evaluating()
        
        # Extract problem scores from the per_graph_metrics if available
        problem_scores = []
        for pg_metric in evaluation_metrics.get('per_graph_metrics', []):
            if abs(pg_metric.get('llm_average_score', 0) - graph_llm_score) < 1e-6:
                problem_scores = pg_metric.get('problem_scores', [])
                break
        
        graph_average_scores.append(graph_llm_score)
        
        # Count nodes and edges
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        num_nodes = len(nodes)
        num_edges = len(edges)
        
        result = {
            'graph_id': len(results),
            'graph_nodes': json.dumps(nodes),
            'graph_edges': json.dumps(edges),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'average_score': graph_llm_score,
            'problem_scores': json.dumps(problem_scores),
            'total_evaluation_time': graph_time,
            'num_problems_evaluated': len(problem_scores)
        }
        results.append(result)
        
        logger.info(f"[Graph {len(results)}/{len(sampled_graphs)}] "
                    f"Average score: {graph_llm_score:.4f}, "
                    f"Problems: {len(problem_scores)}, "
                    f"Scores: {[f'{s:.3f}' for s in problem_scores[:5]]}{'...' if len(problem_scores) > 5 else ''}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "random_graph_evaluations.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved results to {csv_path}")
    logger.info(f"  Total graphs evaluated: {len(results)}")
    logger.info(f"  Average score range: [{min(graph_average_scores):.4f}, {max(graph_average_scores):.4f}]")
    logger.info(f"  Mean score: {np.mean(graph_average_scores):.4f}")
    logger.info(f"  Std score: {np.std(graph_average_scores):.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {csv_path}")
    logger.info(f"Next step: Run 'python distribution_research/fit_distributions.py' to fit distributions")


if __name__ == "__main__":
    main()



