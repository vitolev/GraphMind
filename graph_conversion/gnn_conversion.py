"""

#TODO preveri vse tu, veliko bo drugače

Convert serialized graphs to GNN-compatible format

This converts graph dictionaries to the format expected by GNN models
(node features, edge indices, etc.)
"""

import logging
from typing import List, Dict, Any
import numpy as np

def convert_to_gnn_format_batch(
    serialized_graphs: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Convert serialized graphs to GNN-compatible format
    
    Each graph becomes:
    {
        'node_features': np.array,
        'edge_index': List[tuple],
        'graph_features': Dict,
        'original_id': str,
    }
    
    Args:
        serialized_graphs: List of serialized graph dicts
        logger: Logger
    
    Returns:
        List of GNN-format graphs
    """
    
    logger.debug(f"Converting {len(serialized_graphs)} graphs to GNN format")
    
    gnn_graphs = []
    for graph in serialized_graphs:
        try:
            gnn_graph = _convert_single_to_gnn_format(graph)
            gnn_graphs.append(gnn_graph)
        except Exception as e:
            logger.warning(f"Failed to convert graph {graph.get('id', '?')}: {e}")
            continue
    
    logger.debug(f"Converted {len(gnn_graphs)} graphs to GNN format")
    return gnn_graphs

def _convert_single_to_gnn_format(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert single graph to GNN format
    
    Args:
        graph: Serialized graph dictionary
    
    Returns:
        GNN-format graph
    """
    
    num_agents = graph.get('num_agents', 1)
    agent_roles = graph.get('agent_roles', [])
    edges = graph.get('communication_edges', [])
    
    # Create node features: one-hot encode agent roles
    node_features = _encode_agent_roles(agent_roles)
    
    # Edge indices
    edge_index = edges
    
    # Graph-level features
    graph_features = {
        'num_agents': num_agents,
        'num_edges': len(edges),
        'density': len(edges) / (num_agents * (num_agents - 1)) if num_agents > 1 else 0,
    }
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'graph_features': graph_features,
        'original_id': graph.get('id'),
        'num_nodes': num_agents,
    }

def _encode_agent_roles(agent_roles: List[str]) -> np.ndarray:
    """
    Encode agent roles as features
    
    Args:
        agent_roles: List of role strings
    
    Returns:
        Node feature matrix (num_agents x feature_dim)
    """
    
    # Simple encoding: one-hot for each role type
    role_types = ['solver', 'verifier', 'coordinator']
    
    features = []
    for role in agent_roles:
        # One-hot encoding
        one_hot = [1 if role == r else 0 for r in role_types]
        # Add degree placeholder (to be computed later)
        one_hot.append(0)
        features.append(one_hot)
    
    return np.array(features, dtype=np.float32)

def convert_from_gnn_format(
    gnn_graphs: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Convert GNN-format graphs back to serialized format
    
    Args:
        gnn_graphs: List of GNN-format graphs
        logger: Logger
    
    Returns:
        List of serialized graphs
    """
    
    logger.debug(f"Converting {len(gnn_graphs)} graphs from GNN format")
    
    serialized = []
    for gnn_graph in gnn_graphs:
        try:
            graph = _convert_single_from_gnn_format(gnn_graph)
            serialized.append(graph)
        except Exception as e:
            logger.warning(f"Failed to convert from GNN format: {e}")
            continue
    
    logger.debug(f"Converted {len(serialized)} graphs from GNN format")
    return serialized

def _convert_single_from_gnn_format(gnn_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert single GNN-format graph back to serialized
    
    Args:
        gnn_graph: GNN-format graph
    
    Returns:
        Serialized graph
    """
    
    return {
        'id': gnn_graph.get('original_id'),
        'num_agents': gnn_graph.get('num_nodes'),
        'agent_roles': _decode_agent_roles(gnn_graph.get('node_features')),
        'communication_edges': gnn_graph.get('edge_index'),
        'graph_data': None,
    }

def _decode_agent_roles(node_features: np.ndarray) -> List[str]:
    """
    Decode agent roles from one-hot features
    
    Args:
        node_features: Node feature matrix
    
    Returns:
        List of role strings
    """
    
    role_types = ['solver', 'verifier', 'coordinator']
    roles = []
    
    for features in node_features:
        # Get argmax of first 3 features (one-hot encoding)
        role_idx = int(np.argmax(features[:3]))
        roles.append(role_types[role_idx])
    
    return roles
