
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
from config.settings import Config

class GraphSet:
    """Container for good graphs"""
    
    def __init__(self):
        self.graphs: List[Dict[str, Any]] = []

    def get(self, index: int) -> Dict[str, Any]:
        return self.graphs[index]
    
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
    
    def enforce_max_size(self, max_size) -> None:
        """Trim to max size if exceeded"""
        if len(self.graphs) > max_size:
            self.graphs = self.graphs[:max_size]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'graphs': self.graphs,
            'count': len(self.graphs),
        }

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



# """
# ================================================================================
# FILE: data_management/graph_set.py
# ================================================================================

# PURPOSE:
#     Container for good Graph objects that accumulates high-performing
#     multiagent systems across iterations.

# COLLECTION:
#     data_management

# DEPENDENCIES:
#     - logging (stdlib)
#     - pickle (stdlib)
#     - typing (stdlib)

# KEY COMPONENTS:
#     - GraphSet: Container for Graph objects
#     - load_good_graphs_set(): Load from disk
#     - save_good_graphs_set(): Save to disk
#     - add_to_graphs_set(): Add new graphs
#     - select_for_evaluation(): Select top graphs

# RESPONSIBILITY MATRIX:
#     - alice: Owns GraphSet implementation

# DATA STORAGE:
#     - Stores Graph objects (from data_management.graph)
#     - Sorted by gnn_predicted_score (descending)
#     - Manages accumulation across iterations
#     - Persists to disk via pickle

# LAST UPDATED: 2025-11-04
# STATUS: active
# ================================================================================
# """

# import logging
# import pickle
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from config.settings import Config
# from data_management.graph import Graph


# class GraphSet:
#     """
#     ════════════════════════════════════════════════════════════════════════════
#     CLASS: GraphSet
#     ════════════════════════════════════════════════════════════════════════════
    
#     PURPOSE:
#         Container for Graph objects representing high-performing multiagent
#         systems. Accumulates across iterations, sorted by GNN predicted score.
    
#     ATTRIBUTES:
#         - graphs: List of Graph objects
#         - max_size: Maximum number of graphs to keep
    
#     METHODS:
#         - add_graphs(): Add new Graph objects
#         - get_all(): Get all graphs
#         - get_best_k(): Get top-K graphs without removing
#         - size(): Get number of graphs
#         - enforce_max_size(): Trim to max size
#         - to_dict(): Serialize all graphs
    
#     ════════════════════════════════════════════════════════════════════════════
#     """
    
#     def __init__(self, max_size: int = 1000):
#         """
#         Initialize GraphSet
        
#         Args:
#             max_size: Maximum number of graphs to keep
#         """
#         self.graphs: List[Graph] = []
#         self.max_size = max_size
    
#     def add_graphs(self, new_graphs: List[Graph]) -> None:
#         """
#         Add new Graph objects to set and re-sort by predicted score
        
#         Args:
#             new_graphs: List of Graph objects to add
#         """
#         self.graphs.extend(new_graphs)
        
#         # Sort by gnn_predicted_score (descending): best predictions first
#         self.graphs.sort(
#             key=lambda g: g.gnn_predicted_score if g.gnn_predicted_score is not None else float('-inf'),
#             reverse=True
#         )
    
#     def get_all(self) -> List[Graph]:
#         """Get all graphs in the set"""
#         return self.graphs
    
#     def get_best_k(self, k: int) -> List[Graph]:
#         """
#         Get top-K graphs WITHOUT removing them from set
        
#         Args:
#             k: Number of graphs to select
        
#         Returns:
#             Top-K Graph objects
#         """
#         return self.graphs[:k]
    
#     def get(self, index: int) -> Graph:
#         """Get graph at specific index"""
#         return self.graphs[index]
    
#     def size(self) -> int:
#         """Get number of graphs in set"""
#         return len(self.graphs)
    
#     def enforce_max_size(self) -> None:
#         """
#         Trim set to max_size if exceeded
        
#         Keeps the FIRST max_size graphs (best scores, since sorted descending)
#         Removes worst-scoring graphs
#         """
#         if len(self.graphs) > self.max_size:
#             self.graphs = self.graphs[:self.max_size]
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for serialization"""
#         return {
#             'graphs': [g.to_dict() for g in self.graphs],
#             'count': len(self.graphs),
#             'max_size': self.max_size,
#         }


# def load_good_graphs_set(
#     config_data_dir: Path,
#     max_size: int,
#     logger: logging.Logger
# ) -> GraphSet:
#     """
#     ════════════════════════════════════════════════════════════════════════════
#     FUNCTION: load_good_graphs_set
#     ════════════════════════════════════════════════════════════════════════════
    
#     PURPOSE:
#         Load good_graphs_set from disk or create new if doesn't exist
    
#     INPUTS:
#         config_data_dir: Path to data directory
#         max_size: Maximum size of set (from config)
#         logger: Logger
    
#     OUTPUTS:
#         GraphSet: Loaded or new empty GraphSet
    
#     ════════════════════════════════════════════════════════════════════════════
#     """
#     graph_set = GraphSet(max_size=max_size)
#     set_path = config_data_dir / "good_graphs_set.pkl"
    
#     if set_path.exists():
#         try:
#             with open(set_path, 'rb') as f:
#                 loaded = pickle.load(f)
            
#             # Deserialize Graph objects from saved dicts
#             graph_dicts = loaded.get('graphs', [])
#             graph_set.graphs = [Graph.from_dict(gd, logger) for gd in graph_dicts]
            
#             logger.info(
#                 f"Loaded good_graphs_set with {graph_set.size()} graphs from {set_path}"
#             )
#         except Exception as e:
#             logger.warning(
#                 f"Could not load good_graphs_set: {e}, starting fresh"
#             )
#     else:
#         logger.info(
#             f"No existing good_graphs_set found at {set_path}, starting fresh"
#         )
    
#     return graph_set


# def save_good_graphs_set(
#     graph_set: GraphSet,
#     config_data_dir: Path,
#     logger: logging.Logger
# ) -> None:
#     """
#     ════════════════════════════════════════════════════════════════════════════
#     FUNCTION: save_good_graphs_set
#     ════════════════════════════════════════════════════════════════════════════
    
#     PURPOSE:
#         Save good_graphs_set to disk for persistence
    
#     INPUTS:
#         graph_set: GraphSet to save
#         config_data_dir: Path to data directory
#         logger: Logger
    
#     ════════════════════════════════════════════════════════════════════════════
#     """
#     config_data_dir.mkdir(parents=True, exist_ok=True)
#     set_path = config_data_dir / "good_graphs_set.pkl"
    
#     try:
#         with open(set_path, 'wb') as f:
#             pickle.dump(graph_set.to_dict(), f)
#         logger.debug(
#             f"Saved good_graphs_set ({graph_set.size()} graphs) to {set_path}"
#         )
#     except Exception as e:
#         logger.error(f"Failed to save good_graphs_set: {e}")
#         raise


# def add_to_graphs_set(
#     graph_set: GraphSet,
#     new_graphs: List[Graph],
#     logger: logging.Logger
# ) -> None:
#     """
#     ════════════════════════════════════════════════════════════════════════════
#     FUNCTION: add_to_graphs_set
#     ════════════════════════════════════════════════════════════════════════════
    
#     PURPOSE:
#         Add new graphs to set and enforce max size
    
#     INPUTS:
#         graph_set: GraphSet to update
#         new_graphs: List of Graph objects to add
#         logger: Logger
    
#     ════════════════════════════════════════════════════════════════════════════
#     """
#     initial_size = graph_set.size()
    
#     # Add and re-sort
#     graph_set.add_graphs(new_graphs)
    
#     # Enforce max size
#     graph_set.enforce_max_size()
    
#     final_size = graph_set.size()
    
#     logger.debug(
#         f"Added {len(new_graphs)} graphs to good_graphs_set "
#         f"({initial_size} → {final_size})"
#     )


# def select_for_evaluation(
#     graph_set: GraphSet,
#     k: int,
#     logger: logging.Logger
# ) -> List[Graph]:
#     """
#     ════════════════════════════════════════════════════════════════════════════
#     FUNCTION: select_for_evaluation
#     ════════════════════════════════════════════════════════════════════════════
    
#     PURPOSE:
#         Select top-K Graph objects for evaluation WITHOUT removing them
    
#     INPUTS:
#         graph_set: GraphSet to select from
#         k: Number of graphs to select
#         logger: Logger
    
#     OUTPUTS:
#         Top-K Graph objects
    
#     NOTES:
#         - Does NOT modify graph_set (non-destructive)
#         - Graphs remain in set for next iteration
    
#     ════════════════════════════════════════════════════════════════════════════
#     """
#     selected_graphs = graph_set.get_best_k(k)
    
#     logger.debug(
#         f"Selected top {len(selected_graphs)} graphs for evaluation "
#         f"(from {graph_set.size()} in good_graphs_set)"
#     )
    
#     return selected_graphs


# def select_top_graphs(
#     config: Config,
#     logger: logging.Logger,
#     graphs: List[Graph],
#     good_graphs_set: GraphSet
# ) -> List[Graph]:
#     """
#     ════════════════════════════════════════════════════════════════════════════
#     FUNCTION: select_top_graphs
#     ════════════════════════════════════════════════════════════════════════════
    
#     PURPOSE:
#         Manage good_graphs_set and select best graphs for evaluation:
#         1. Extract top_k_to_keep graphs from predictions (by gnn_predicted_score)
#         2. Add to accumulating good_graphs_set
#         3. Enforce max_size
#         4. Select eval_k_best for evaluation
#         5. Save updated set
    
#     INPUTS:
#         config: Configuration with top_k_to_keep, eval_k_best, good_graphs_max_size
#         logger: Logger
#         graphs: List of Graph objects with gnn_predicted_score set
#         good_graphs_set: Accumulating GraphSet
    
#     OUTPUTS:
#         List[Graph]: Selected graphs ready for evaluation
    
#     ALGORITHM:
#         Step 1: Sort graphs by gnn_predicted_score (already done in add_to_graphs_set)
#         Step 2: Extract top_k_to_keep
#         Step 3: Add to good_graphs_set (triggers re-sort)
#         Step 4: Enforce max_size
#         Step 5: Select eval_k_best for evaluation
#         Step 6: Save set
    
#     ════════════════════════════════════════════════════════════════════════════
#     """
    
#     logger.info(
#         f"Step 3.1: Extracting top {config.top_k_to_keep} graphs "
#         f"from {len(graphs)} predictions..."
#     )
    
#     # Sort by gnn_predicted_score (best first)
#     sorted_graphs = sorted(
#         graphs,
#         key=lambda g: g.gnn_predicted_score if g.gnn_predicted_score is not None else float('-inf'),
#         reverse=True
#     )
    
#     # Extract top_k_to_keep
#     top_k_graphs = sorted_graphs[:config.top_k_to_keep]
#     logger.info(f"  ✓ Extracted {len(top_k_graphs)} graphs")
    
#     # Add to good_graphs_set
#     logger.info(f"Step 3.2: Adding {len(top_k_graphs)} to good_graphs_set...")
#     size_before = good_graphs_set.size()
#     add_to_graphs_set(good_graphs_set, top_k_graphs, logger)
#     logger.info(f"  ✓ Size: {size_before} → {good_graphs_set.size()}")
    
#     # Enforce max size
#     if good_graphs_set.size() > config.good_graphs_max_size:
#         logger.info(
#             f"Step 3.3: Trimming good_graphs_set to max size "
#             f"({good_graphs_set.size()} → {config.good_graphs_max_size})..."
#         )
#         good_graphs_set.enforce_max_size()
#         logger.info(f"  ✓ Trimmed to {good_graphs_set.size()}")
#     else:
#         logger.info(
#             f"Step 3.3: Good graphs set size OK "
#             f"({good_graphs_set.size()} ≤ {config.good_graphs_max_size})"
#         )
    
#     # Select for evaluation
#     logger.info(
#         f"Step 3.4: Selecting {config.eval_k_best} best graphs "
#         f"for evaluation..."
#     )
#     selected_graphs = select_for_evaluation(good_graphs_set, config.eval_k_best, logger)
#     logger.info(f"  ✓ Selected {len(selected_graphs)} graphs")
    
#     # Save
#     logger.info(f"Step 3.5: Saving updated good_graphs_set...")
#     save_good_graphs_set(good_graphs_set, config.data_dir, logger)
#     logger.info(f"  ✓ Saved {good_graphs_set.size()} graphs")
    
#     return selected_graphs
