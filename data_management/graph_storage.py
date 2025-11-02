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
from typing import List, Dict, Any

class GraphSet:
    """Container for good graphs"""
    
    def __init__(self, max_size: int = 1000):
        self.graphs: List[Dict[str, Any]] = []
        self.max_size = max_size
    
    def add_graphs(self, new_graphs: List[Dict[str, Any]]) -> None:
        """Add graphs to set
        TODO: avoid sorting on every addition if possible, as the existing graph list is already sorted
        """
        self.graphs.extend(new_graphs)
        # Sort by score (descending). For equal scores, newer graphs (later in list) come last.
        self.graphs.sort(key=lambda g: g.get('score', float('-inf')), reverse=True)
    
    def get_all(self) -> List[Dict[str, Any]]:
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
    
    def is_full(self) -> bool:
        """Check if at max capacity"""
        return len(self.graphs) >= self.max_size
    
    def enforce_max_size(self) -> None:
        """Trim to max size if exceeded"""
        if len(self.graphs) > self.max_size:
            self.graphs = self.graphs[:self.max_size]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'graphs': self.graphs,
            'count': len(self.graphs),
            'max_size': self.max_size,
        }

def load_good_graphs_set(
    config_data_dir: Path,
    max_size: int,
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
    graph_set = GraphSet(max_size=max_size)
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

def add_to_graphs_set(
    graph_set: GraphSet,
    new_graphs: List[Dict[str, Any]],
    logger: logging.Logger
) -> None:
    """
    Add new graphs and enforce max size
    
    Args:
        graph_set: GraphSet to update
        new_graphs: Graphs to add
        logger: Logger
    """
    graph_set.add_graphs(new_graphs)
    graph_set.enforce_max_size()
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
