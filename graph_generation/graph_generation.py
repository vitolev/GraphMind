import logging
import random
from typing import Dict, Any, Tuple, Optional
from config.settings import Config
from data_management.graph_storage import Graph, GraphSet
import networkx as nx
import time

def generate_graph_batch(
    config: Config,
    logger: logging.Logger,
    training_dataset: Optional[GraphSet] = None,
) -> Tuple[Dict[str, Any], GraphSet]:
    
    step_start = time.time()

    strategy = config.generation_strategy
    
    if strategy == 'random':
        graphset, duration = _random_strategy(config, logger, training_dataset)
    # elif strategy == 'similar_to_training':
    #     graphset = _similar_to_training_strategy(config, logger, training_dataset)
    # elif strategy == 'custom':
    #     graphset = _custom_strategy(config, logger, training_dataset)
    else:
        raise ValueError(f"Unknown generation strategy: {strategy}")
    
    num_generated = graphset.size()

    duration = time.time() - step_start
    
    metrics = {
        'step_name': 'generation',
        'duration_seconds': round(duration, 4),
        'num_samples': num_generated,
        'strategy': strategy,
        'metadata': {
            'node_types': config.node_types,
            'num_graphs_per_iteration': config.num_graphs_per_iteration,
        }
    }
    
    return metrics, graphset

def _random_strategy(
    config: Config,
    logger: logging.Logger,
    training_dataset: Optional[GraphSet] = None,
) -> Tuple[GraphSet, float]:

    generated_graphs = GraphSet()
    
    for graph_idx in range(config.num_graphs_per_iteration):
        if (graph_idx + 1) % 500 == 0:
            logger.debug(f"Generated {graph_idx + 1} / {config.num_graphs_per_iteration} graphs")
        try:
            num_nodes = random.randint(config.min_nodes, config.max_nodes)
            
            edge_probability = random.uniform(0.1, 0.5)
            nx_graph = nx.gnp_random_graph(num_nodes, edge_probability)
            
            nodes_with_types = [
                (node_id, random.choice(config.agent_types))
                for node_id in nx_graph.nodes()
            ]
            
            edges = list(nx_graph.edges())
            
            graph = Graph(nodes=nodes_with_types, edges=edges)
            generated_graphs.add_graph(graph)
            
        except Exception as e:
            logger.error(f"Error generating graph {graph_idx}: {e}")
            continue
    
    return generated_graphs