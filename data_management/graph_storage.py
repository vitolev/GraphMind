
"""
Storage for good_graphs_set

The good_graphs_set accumulates high-performing graphs across iterations.
We need to:
- Save it periodically
- Load it on startup
- Manage its size
"""

import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any
from config.settings import Config

@dataclass
class Graph:
    """Graph data structure"""

    def __init__(self, nodes: List[tuple[int, str]], edges: List[tuple[int, int]], gnn_score: float = 0.0, llm_score: float = 0.0, time_evaluating: float = 0.0):
        self.nodes: List[tuple[int, str]] = nodes # List of (node_id, node_type)
        self.edges: List[tuple[int, int]] = edges  # List of (from_node_id, to_node_id): directed edges
        self.gnn_score: float = gnn_score   # Score predicted by GNN
        self.llm_score: float = llm_score   # Score evaluated by LLM
        self.time_evaluating: float = time_evaluating   # Time taken for evaluation

    def set_gnn_score(self, score: float) -> None:
        self.gnn_score = score

    def set_llm_score(self, score: float, time: float) -> None:
        self.llm_score = score
        self.time_evaluating = time

    def get_nodes(self) -> List[tuple[int, str]]:
        return self.nodes
    
    def get_edges(self) -> List[tuple[int, int]]:
        return self.edges
    
    def get_gnn_score(self) -> float:
        return self.gnn_score
    
    def get_llm_score(self) -> float:
        return self.llm_score
    
    def get_time_evaluating(self) -> float:
        return self.time_evaluating
    
    def to_pyg(self):
        """Convert to PyG Data object"""
        pass

@dataclass
class GraphSet:
    """Container for graphs sorted """
    
    def __init__(self):
        self.graphs: List[Graph] = []

    def get(self, index: int) -> Graph:
        return self.graphs[index]
    
    def sort_by_scores(self) -> None:
        """Sort graphs by llm_score and gnn_score descending"""
        self.graphs.sort(key=lambda g: (g.llm_score, g.gnn_score), reverse=True)
    
    def add_graph(self, graph: Graph, sort=False) -> None:
        """Add a single graph to the set"""
        self.graphs.append(graph)
        if sort:
            self.sort_by_scores()

    def add_graphs(self, new_graphs: List[Graph], sort=False) -> None:
        """Add multiple graphs to the set"""
        for graph in new_graphs:
            self.add_graph(graph)
        if sort:
            self.sort_by_scores()
    
    def get_all(self) -> List[Graph]:
        """Get all graphs"""
        return self.graphs
    
    def get_best_k_and_remove(self, k: int) -> List[Graph]:
        """Get top-K graphs and remove them from the set"""
        selected = self.graphs[:k]
        self.graphs = self.graphs[k:]
        return selected
    
    def size(self) -> int:
        """Get number of graphs"""
        return len(self.graphs)
    
    def enforce_max_size(self, max_size) -> None:
        """Trim to max size if exceeded"""
        if len(self.graphs) > max_size:
            self.graphs = self.graphs[:max_size]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'graphs': [  # Serialize each graph as a dict
                {
                    'nodes': graph.get_nodes(),
                    'edges': graph.get_edges(),
                    'gnn_score': graph.get_gnn_score(),
                    'llm_score': graph.get_llm_score(),
                    'time_evaluating': graph.get_time_evaluating()
                }
                for graph in self.graphs
            ],
        }

def load_good_graphs_set(
    config_data_dir: Path,
    logger: logging.Logger
) -> GraphSet:
    """
    Load good_graphs_set from disk
    
    Args:
        config_data_dir: Path to data directory
        logger: Logger
    
    Returns:
        GraphSet object
    """
    graph_set = GraphSet()
    set_path = config_data_dir / "good_graphs_set.pkl"
    
    if set_path.exists():
        try:
            with open(set_path, 'rb') as f:
                loaded = pickle.load(f)
            graph_set.graphs = loaded.get('graphs', [])
            logger.info(f"Loaded good graphs set with {graph_set.size()} graphs from {set_path}")
        except Exception as e:
            logger.warning(f"Could not load good graphs set: {e}, starting fresh")
    else:
        logger.info(f"No existing good graphs set found at {set_path}, starting fresh")
    
    return graph_set

def save_good_graphs_set(
    graph_set: GraphSet,
    config_data_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save good_graphs_set to disk
    
    Args:
        graph_set: GraphSet to save
        config_data_dir: Path to data directory
        logger: Logger
    """
    config_data_dir.mkdir(parents=True, exist_ok=True)
    set_path = config_data_dir / "good_graphs_set.pkl"
    
    try:
        with open(set_path, 'wb') as f:
            pickle.dump(graph_set.to_dict(), f)
        logger.debug(f"Saved good graphs set ({graph_set.size()} graphs) to {set_path}")
    except Exception as e:
        logger.error(f"Failed to save good graphs set: {e}")
        raise

def select_top_graphs(
    config: Config,
    logger: logging.Logger,
    good_graphs_set: GraphSet,
    batch: GraphSet
) -> GraphSet:
    """
    Merge top_k_to_keep graphs from batch into good_graphs_set,
    select eval_k_best top graphs for evaluation, remove them from good_graphs_set, limit the size of good_graphs_set to top_k_to_keep,
    and return the selected graphs for evaluation.
    Args:
        config: Configuration. It contains top_k_to_keep, a number of graphs from new_graphs_set to add to good_graphs_set
                and eval_k_best, a number of graphs to output.
        logger: Logger
        good_graphs_set: Existing good graphs set
        batch: New graphs to consider
    Returns:
        New GraphSet with eval_k_best top graphs for evaluation.
    """
    # Merge top_k_to_keep from batch into good_graphs_set
    num_to_add = min(config.top_k_to_keep, batch.size())   # To avoid index error
    batch.sort_by_scores()  # Ensure batch is sorted
    graphs_to_add = [batch.get(i) for i in range(num_to_add)]
    good_graphs_set.add_graphs(graphs_to_add, sort=True)    # Maintain sorted order
    logger.debug(f"Added {num_to_add} graphs to good graphs set, now has {good_graphs_set.size()} graphs.")
    
    # Select eval_k_best top graphs for evaluation
    num_to_select = min(config.eval_k_best, good_graphs_set.size())
    selected_graphs = good_graphs_set.get_best_k_and_remove(num_to_select)
    logger.debug(f"Selected {num_to_select} graphs for evaluation, good graphs set now has {good_graphs_set.size()} graphs.")
    
    # Enforce max size of good_graphs_set
    good_graphs_set.enforce_max_size(config.top_k_to_keep)
    logger.debug(f"Enforced max size of good graphs set to {config.top_k_to_keep}, now has {good_graphs_set.size()} graphs.")

    # Save updated good_graphs_set
    save_good_graphs_set(good_graphs_set, config.data_dir, logger)
    
    # Return selected graphs as a new GraphSet
    eval_graphs_set = GraphSet()
    eval_graphs_set.add_graphs(selected_graphs)
    
    return eval_graphs_set