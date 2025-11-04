import logging
import random
from typing import List, Dict, Any
from config.settings import Config


def evaluate_selected_graphs(
    config: Config,
    logger: logging.Logger,
    selected_graphs: List[Dict[str, Any]],
    math_problems: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: evaluate_selected_graphs
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Evaluate selected graphs and return actual performance scores.
        Uses simple formula based on graph structure as placeholder.
    
    RESPONSIBILITY MATRIX:
        - ownership: alice
        - critical: yes
    
    INPUTS:
        config (Config): Configuration
        logger (Logger): Logger
        selected_graphs (List[Dict]): Graphs to evaluate, each:
                                     {
                                         'graph': {
                                             'num_nodes': int,
                                             'edge_index': List,
                                             ...
                                         },
                                         'score': float (predicted)
                                     }
        math_problems (List[Dict]): Math problems (used for count)
    
    OUTPUTS:
        List[Dict]: Evaluation results ready for training
                   Each:
                   {
                       'graph': {...},
                       'actual_score': float,
                       'num_correct': int,
                       'num_attempted': int,
                       'average_steps': float,
                       'execution_time': float,
                   }
    
    FORMULA (PLACEHOLDER):
        score = (edges / nodes^2) + random_noise
        - Better connected graphs → higher score
        - Random noise simulates variability
    
    LAST UPDATED: 2025-11-04
    ════════════════════════════════════════════════════════════════════════════
    """
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION (Simple Formula)")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  - Graphs to evaluate: {len(selected_graphs)}")
    logger.info(f"  - Math problems available: {len(math_problems)}")
    logger.info(f"  - Formula: score = edges/(nodes^2) + noise")
    
    results = []
    
    for graph_dict in selected_graphs:
        graph = graph_dict.get('graph', {})
        result = _evaluate_single_graph(graph, len(math_problems), logger)
        results.append(result)
    
    logger.info(f"\n  ✓ Evaluated {len(results)} graphs")
    
    if results:
        scores = [r['actual_score'] for r in results]
        logger.info(f"\nScore Summary:")
        logger.info(f"  - Min: {min(scores):.4f}")
        logger.info(f"  - Max: {max(scores):.4f}")
        logger.info(f"  - Mean: {sum(scores)/len(scores):.4f}")
    
    return results


def _evaluate_single_graph(
    graph: Dict[str, Any],
    num_problems: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: _evaluate_single_graph
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Evaluate single graph using simple formula
    
    FORMULA:
        1. Extract: num_nodes, num_edges from graph
        2. Calculate base score: edges / (nodes^2)
        3. Add random noise: +/- 10%
        4. Clamp to [0, 1]
        5. Simulate num_correct based on score
    
    EXAMPLE:
        num_nodes = 5
        num_edges = 8
        base = 8 / 25 = 0.32
        score = 0.32 ± 10% random = ~0.35
        num_correct = round(0.35 * 10) = 3-4 correct problems
    
    LAST UPDATED: 2025-11-04
    ════════════════════════════════════════════════════════════════════════════
    """
    
    print(graph)
    graph_id = graph.get('original_id')
    
    # Extract graph structure
    num_nodes = graph.get('num_nodes', 1)
    edge_index = graph.get('edge_index', [])
    num_edges = len(edge_index)
    
    # FORMULA: edges / nodes^2
    base_score = num_edges / (num_nodes ** 2) if num_nodes > 0 else 0.0
    
    # Add random noise (~10%)
    noise = random.uniform(-0.1, 0.1)
    score = base_score + noise
    
    # Clamp to [0, 1]
    actual_score = max(0.0, min(1.0, score))
    
    # Simulate performance on problems
    num_correct = round(actual_score * num_problems)
    
    logger.debug(
        f"  Evaluated {graph_id}: "
        f"nodes={num_nodes}, edges={num_edges}, "
        f"base={base_score:.4f}, noise={noise:+.4f}, "
        f"score={actual_score:.4f}, "
        f"correct={num_correct}/{num_problems}"
    )
    
    return {
        'graph': graph,
        'actual_score': actual_score,
        'num_correct': num_correct,
        'num_attempted': num_problems,
        'average_steps': random.uniform(5, 50),
        'execution_time': random.uniform(1, 10),
    }
