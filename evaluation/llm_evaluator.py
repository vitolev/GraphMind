import logging
import random
import time
from typing import List, Dict, Any, Tuple
from config.settings import Config

import numpy as np
from data_management.graph_storage import GraphSet

def evaluate_selected_graphs(
    config: Config,
    logger: logging.Logger,
    selected_graphs: GraphSet,
    math_problems: Any,
) -> Tuple[Dict[str, Any], GraphSet]:
    
    step_start = time.time()
    num_graphs = selected_graphs.size()
    
    logger.debug(f"Starting evaluation of {num_graphs} selected graphs")
    
    scores = []
    
    for graph in selected_graphs.get_all():
        try:
            num_nodes = len(graph.get_nodes())
            num_edges = len(graph.get_edges())
            
            if num_nodes > 0:
                structure_score = num_edges / (num_nodes ** 2)
            else:
                structure_score = 0.0
            
            random_component = np.random.uniform(0.0, 0.3)
            score = structure_score + random_component
            
            node_types = [node_type for _, node_type in graph.nodes]
            if 'type_a' in node_types:
                score += 0.2
            
            score = min(1.0, max(0.0, score))
            
            graph.set_llm_score(score, time=0.1)
            scores.append(score)
            
        except Exception as e:
            logger.warning(f"Error evaluating graph: {e}")
            graph.set_llm_score(0.0, time=0.1)
            scores.append(0.0)
            continue
    
    evaluation_time = time.time() - step_start
    scores_array = np.array(scores)
    
    metrics = {
        'step_name': 'evaluation',
        'duration_seconds': round(evaluation_time, 4),
        'num_samples': num_graphs,
        'best_evaluated': float(scores_array.max()) if len(scores_array) > 0 else None,
        'worst_evaluated': float(scores_array.min()) if len(scores_array) > 0 else None,
        'mean_evaluated': float(scores_array.mean()) if len(scores_array) > 0 else None,
        'std_evaluated': float(scores_array.std()) if len(scores_array) > 0 else None,
        'metadata': {
            'evaluation_time_per_graph': round(evaluation_time / num_graphs, 4) if num_graphs > 0 else 0,
            'evaluation_method': 'synthetic_heuristic',  #TODO
        }
    }
    
    logger.debug(
        f"Evaluation complete - "
        f"Best: {metrics['best_evaluated']:.4f}, "
        f"Mean: {metrics['mean_evaluated']:.4f}, "
        f"Time: {evaluation_time:.4f}s"
    )
    
    return metrics, selected_graphs
