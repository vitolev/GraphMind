
"""
Storage for good_graphs_set

The good_graphs_set accumulates high-performing graphs across iterations.
We need to:
- Save it periodically
- Load it on startup
- Manage its size
"""
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
from config.settings import Config
import torch
from torch_geometric.data import Data, HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from config.nodes import NODE_TYPES

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
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Graph):
            return False
        return (
            self.nodes == other.nodes and
            self.edges == other.edges
        )
    
    def to_pyg(self, config: Config):
        """
        Convert to a PyG Data or HeteroData object.

        Args:
            config: Config object with `config.data_format` with value
                    "Data" returns torch_geometric.data.Data (homogeneous),
                    "HeteroData" returns torch_geometric.data.HeteroData (heterogeneous),
                    otherwise raises ValueError.

        Returns:
            torch_geometric.data.Data or torch_geometric.data.HeteroData
        """

        agent_types = NODE_TYPES
        type = config.data_format

        # --- Case 1: Homogeneous Data ---
        if type == "Data":
            # One-hot encode node types based on config.agent_types
            num_types = len(agent_types)
            type_to_idx = {t: i for i, t in enumerate(agent_types)}
            edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()

            # Node features
            num_nodes = len(self.nodes)
            x = torch.zeros((num_nodes, num_types), dtype=torch.float)

            for i, (_, node_type) in enumerate(self.nodes):
                if node_type not in type_to_idx:
                    raise ValueError(f"Unknown node type '{node_type}' not in config.agent_types")
                x[i, type_to_idx[node_type]] = 1.0

            # ------------------------------
            # Add super node (virtual node)
            # ------------------------------

            # Super-node feature vector: zeros (same dimension)
            super_node_feat = torch.zeros((1, num_types), dtype=torch.float)

            # Append super node to x
            x = torch.cat([x, super_node_feat], dim=0)

            # Index of the super node (last node)
            super_idx = num_nodes

            # Edges from super node to all nodes
            row_to = torch.full((num_nodes,), super_idx, dtype=torch.long)
            col_to = torch.arange(num_nodes, dtype=torch.long)

            # Edges from all nodes to super node
            row_from = torch.arange(num_nodes, dtype=torch.long)
            col_from = torch.full((num_nodes,), super_idx, dtype=torch.long)

            # Stack new edges (bidirectional)
            new_edges = torch.stack(
                [
                    torch.cat([row_to, row_from]),
                    torch.cat([col_to, col_from])
                ],
                dim=0
            )

            # Append edges
            edge_index = torch.cat([edge_index, new_edges], dim=1)

            # Final Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([self.llm_score], dtype=torch.float)
            )

            return data

        # --- Case 2: Heterogeneous Data ---
        elif type == "HeteroData":
            hetero_data = HeteroData()

            # ---- FAST: Build node groups & local index maps in one pass ----
            nodes_by_type = {t: [] for t in agent_types}
            node_id_to_local = {}

            for node_id, node_type in self.nodes:
                local_idx = len(nodes_by_type[node_type])
                nodes_by_type[node_type].append(node_id)
                node_id_to_local[node_id] = (node_type, local_idx)

            # Assign node features (1-dimensional)
            for node_type, node_ids in nodes_by_type.items():
                hetero_data[node_type].x = torch.ones((len(node_ids), 1), dtype=torch.float)

            # ---- FAST: Collect edges by type without preinitializing everything ----
            edges_by_type = {}

            for src_id, dst_id in self.edges:
                src_type, src_idx = node_id_to_local[src_id]
                dst_type, dst_idx = node_id_to_local[dst_id]

                key = (src_type, "to", dst_type)
                if key not in edges_by_type:
                    edges_by_type[key] = []
                edges_by_type[key].append((src_idx, dst_idx))

            # ---- Only add edge types that actually exist ----
            for key, edge_list in edges_by_type.items():
                hetero_data[key].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # ---- Super node ----
            super_type = "super"
            hetero_data[super_type].x = torch.zeros((1, 1), dtype=torch.float)

            # ---- Connect super node to all node types (bidirectional) ----
            for node_type, node_ids in nodes_by_type.items():
                n = len(node_ids)
                if n == 0:
                    continue

                # node_type to super (only in one direction)
                hetero_data[(node_type, "to", super_type)].edge_index = torch.stack([
                    torch.arange(n, dtype=torch.long),
                    torch.zeros(n, dtype=torch.long)
                ], dim=0)

            # Target
            hetero_data.y = torch.tensor([self.llm_score], dtype=torch.float)
            
            return hetero_data
        
        #-- Invalid type specified ---
        else:
            raise ValueError(f"Invalid type '{type}' specified. Use 'Data' or 'HeteroData'.")
        
    def visualize(self) -> None:
        """Visualize the graph using networkx and matplotlib"""
        G = nx.DiGraph()
        for node_id, node_type in self.nodes:
            G.add_node(node_id, label=node_type)
        G.add_edges_from(self.edges)

        pos = nx.planar_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, font_size=8, arrows=True)
        plt.show()

class GraphSet:
    """Container for graphs sorted """
    
    def __init__(self):
        self.graphs: List[Graph] = []

    def get(self, index: int) -> Graph:
        return self.graphs[index]
    
    def contains(self, graph: Graph) -> bool:
        """Check if a graph is already in the set"""
        return any(g == graph for g in self.graphs)
    
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

    def to_pyg(self, config: Config):
        """Convert all graphs to PyG Data or HeteroData objects"""
        pyg_graphs = []
        for graph in self.graphs:
            pyg_graph = graph.to_pyg(config)
            pyg_graphs.append(pyg_graph)
        return pyg_graphs


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
            graph_set = loaded
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
            pickle.dump(graph_set, f)
        logger.debug(f"Saved good graphs set ({graph_set.size()} graphs) to {set_path}")
    except Exception as e:
        logger.error(f"Failed to save good graphs set: {e}")
        raise

def load_training_dataset(
    config_data_dir: Path,
    logger: logging.Logger
) -> GraphSet:
    """
    Load training dataset from disk
    
    If file doesn't exist, returns empty dataset
    
    Args:
        config_data_dir: Path to data directory
        logger: Logger
    
    Returns:
        GraphSet object
    """
    dataset = GraphSet()
    dataset_path = config_data_dir / "training_dataset.pkl"
    
    if dataset_path.exists():
        try:
            with open(dataset_path, 'rb') as f:
                loaded = pickle.load(f)
            dataset = loaded
            logger.info(f"Loaded training dataset with {dataset.size()} graphs from {dataset_path}")
        except Exception as e:
            logger.warning(f"Could not load training dataset: {e}, starting fresh")
    else:
        logger.info(f"No existing training dataset found at {dataset_path}, starting fresh")
    
    return dataset

def save_training_dataset(
    dataset: GraphSet,
    config_data_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save training dataset to disk
    
    Args:
        dataset: GraphSet to save
        config_data_dir: Path to data directory
        logger: Logger
    """
    config_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config_data_dir / "training_dataset.pkl"
    
    try:
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.debug(f"Saved training dataset ({dataset.size()} graphs) to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to save training dataset: {e}")
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

    metrics = {
        'step_name': 'selection',
        'num_graphs_selected': eval_graphs_set.size(),
        'good_graphs_set_size': good_graphs_set.size(),
        # Next metric is a metric about the graphs, so we take the first graph and we count how many of which nodes there are
        'good_graphs_description': {
            'node_type_counts': {
                node_type: sum(1 for graph in good_graphs_set.get_all() for _, t in graph.get_nodes() if t == node_type)
                for node_type in NODE_TYPES
            }
        }
    }
    return metrics, eval_graphs_set

def update_training_data(
    config: Config,
    logger: logging.Logger,
    evaluation_results: GraphSet,
    training_dataset: GraphSet
) -> int:
    """
    Add evaluated graphs (Graph with llm_score) to training dataset and save it.

    Args:
        config: Configuration
        logger: Logger
        evaluation_results: GraphSet with evaluated graphs
        training_dataset: GraphSet representing the training dataset

    Returns:
        Number of samples actually added
    """
    # Add graphs to training dataset
    training_dataset.add_graphs(evaluation_results.get_all())
    num_added = evaluation_results.size()

    # Save updated training dataset
    save_training_dataset(training_dataset, config.data_dir, logger)

    metrics = { 
        'step_name': 'training_data_update',
        'num_training_samples_added': num_added,
        'total_training_dataset_size': training_dataset.size()
    }

    return metrics


if __name__ == "__main__":
    sample_graph = Graph(
        nodes=[(0, 'START'), (1, 'Solver'), (2, 'END')],
        edges=[(0, 1), (1, 2)],
        gnn_score=0.8,
        llm_score=0.9,
        time_evaluating=1.2
    )

    hetData = sample_graph.to_pyg(Config(data_format="HeteroData"))
    print(hetData)

