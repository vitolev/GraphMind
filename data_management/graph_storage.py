
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
from bisect import bisect_left

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
    
    def add_graph(self, graph: Graph) -> None:
        """Add a single graph to the set and maintain sorted order"""
        graph_llm_score = graph.get_llm_score()
        graph_gnn_score = graph.get_gnn_score()

        # Create the key for comparison
        new_key = (graph_llm_score, graph_gnn_score)

        # Find insertion index using bisection on the same key structure
        keys = [(g.get_llm_score(), g.get_gnn_score()) for g in self.graphs]
        insert_idx = bisect_left(keys, new_key)

        # Insert while maintaining order
        self.graphs.insert(insert_idx, graph)

    def add_graphs(self, new_graphs: List[Graph]) -> None:
        """Add multiple graphs to the set and maintain sorted order"""
        for graph in new_graphs:
            self.add_graph(graph)
    
    def get_all(self) -> List[Graph]:
        """Get all graphs"""
        return self.graphs
    
    def get_best_k_and_remove(self, k: int) -> List[Dict[str, Any]]:
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


def load_good_graphs_set(
    config_data_dir: Path,
    logger: logging.Logger
) -> GraphSet:
    """
    Load good_graphs_set from disk
    
    Args:
        config_data_dir: Path to data directory
        max_size: Maximum size of set
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
            loaded_max_size = loaded.get('max_size', None)
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

def add_to_graphs_set(
    graph_set: GraphSet,
    new_graphs: List[Dict[str, Any]],
    logger: logging.Logger
) -> None:
    
    graph_set.add_graphs(new_graphs)
    logger.debug(f"Good graphs set updated, now has {graph_set.size()} graphs.")

def select_for_evaluation(
    graph_set: GraphSet,
    k: int,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Select top-K graphs for evaluation from good_graphs_set and remove them from the set.
    
    Args:
        graph_set: GraphSet to select from
        k: Number of graphs to select
        logger: Logger
    Returns:
        List of selected graph dictionaries
    """
    selected_graphs = graph_set.get_best_k_and_remove(k)
    logger.debug(f"Selected best {len(selected_graphs)} graphs for evaluation and removed them from the good graphs set.")
    return selected_graphs

def select_top_graphs(
    config: Config,
    logger: logging.Logger,
    predictions: list,
    good_graphs_set: GraphSet
) -> list:
    """
    Function adds the new predictions, adds them to the good graph set
    """
    top_predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:config.top_k_to_keep]

    add_to_graphs_set(
        good_graphs_set,
        top_predictions,
        logger
    )

    selected = select_for_evaluation(good_graphs_set, config.top_k_to_keep, logger)
    
    good_graphs_set.enforce_max_size(max_size=config.top_k_to_keep)

    save_good_graphs_set(
        good_graphs_set,
        config.data_dir,
        logger
    )
    #print(selected)
    return selected

