"""
================================================================================
FILE: graph_generation/graph_generation.py
================================================================================

PURPOSE:
    Generate heterogeneous multiagent system architectures as torch_geometric
    HeteroData objects. Creates random node and edge types with features.

COLLECTION:
    graph_generation

DEPENDENCIES:
    - logging (stdlib)
    - random (stdlib)
    - torch (external)
    - torch_geometric (external)

KEY COMPONENTS:
    - generate_langgraph_variants(): Generate Graph objects with HeteroData

RESPONSIBILITY MATRIX:
    - alice: Owns graph generation

DATA FORMAT:
    Each Graph contains HeteroData with:
    - Node types: 'solver', 'verifier', 'coordinator'
    - Edge types: ('agent_type', 'communication', 'agent_type')
    - Features: Random tensors (will be replaced with real features later)

NOTES:
    - Currently generates random features (placeholder)
    - Later: Can add learned features, attention weights, etc.
    - HeteroData format compatible with PyTorch Geometric models

LAST UPDATED: 2025-11-04
STATUS: active
================================================================================
"""

import logging
import random
from typing import List
import torch
from torch_geometric.data import HeteroData
from config.settings import Config
from data_management.graph import Graph


def generate_langgraph_variants(
    num_graphs: int,
    config: Config,
    logger: logging.Logger
) -> List[Graph]:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: generate_langgraph_variants
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Generate N heterogeneous multiagent system architectures as Graph objects
        with HeteroData structure.
    
    INPUTS:
        num_graphs: Number of graphs to generate (e.g., 100,000)
        config: Configuration with:
               - num_agents_min, num_agents_max: Agent count range
               - agent_roles: Available roles
               - node_feature_dim: Feature dimension per node
               - seed: Random seed (optional)
        
        logger: Logger
    
    OUTPUTS:
        List[Graph]: Graph objects, each containing:
                    - agent_ids, agent_roles, communication_edges
                    - hetero_data: torch_geometric.data.HeteroData
    
    ALGORITHM:
        For each graph:
        1. Generate random number of agents
        2. Assign random roles to agents
        3. Create heterogeneous data:
           - Node types: 'solver', 'verifier', 'coordinator'
           - Random nodes per type
           - Random edges between types
           - Random features per node
        4. Create Graph object with all data
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
    
    logger.debug(f"Generating {num_graphs} heterogeneous graphs as Graph objects")
    
    graphs = []
    for i in range(num_graphs):
        graph = _create_single_graph(config, i, logger)
        graphs.append(graph)
    
    logger.debug(f"Generated {len(graphs)} heterogeneous graphs")
    return graphs


def _create_single_graph(
    config: Config,
    graph_id: int,
    logger: logging.Logger
) -> Graph:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: _create_single_graph
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Create a single heterogeneous graph as a Graph object
    
    INPUTS:
        config: Configuration
        graph_id: Unique ID for this graph
        logger: Logger
    
    OUTPUTS:
        Graph: Graph object with:
              - agent_ids, agent_roles, communication_edges
              - hetero_data: HeteroData with node types and edges
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    # Step 1: Randomly choose number of agents
    num_agents = random.randint(config.num_agents_min, config.num_agents_max)
    
    # Step 2: Create agent IDs and assign random roles
    agent_ids = [f'agent_{i}' for i in range(num_agents)]
    agent_roles = [
        random.choice(config.agent_roles)
        for _ in range(num_agents)
    ]
    
    # Step 3: Create communication edges between agents
    communication_edges = _generate_communication_edges(agent_ids, config)
    
    # Step 4: Create heterogeneous data structure
    hetero_data = _create_hetero_data(agent_ids, agent_roles, communication_edges, config, logger)
    
    # Step 5: Create and return Graph object
    graph = Graph(
        graph_id=f'graph_{graph_id}',
        agent_ids=agent_ids,
        agent_roles=agent_roles,
        communication_edges=communication_edges,
        _logger=logger
    )
    
    # Store HeteroData in graph for later use
    graph.hetero_data = hetero_data
    
    return graph


def _generate_communication_edges(agent_ids: List[str], config: Config) -> List[tuple]:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: _generate_communication_edges
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Generate communication pattern between agents
    
    PATTERNS:
        - hub (star): One central agent
        - mesh: Randomly connected
        - chain: Linear sequence
        - ring: Circular pattern
    
    INPUTS:
        agent_ids: List of agent IDs
        config: Configuration
    
    OUTPUTS:
        List of (agent_id, agent_id) edges
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    pattern = random.choice(['hub', 'mesh', 'chain', 'ring'])
    edges = []
    num_agents = len(agent_ids)
    
    if pattern == 'hub':
        # Star pattern: agent_0 is central hub
        for i in range(1, num_agents):
            edges.append((agent_ids, agent_ids[i]))
            edges.append((agent_ids[i], agent_ids))
    
    elif pattern == 'mesh':
        # Randomly connected (density ~0.3)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if random.random() < 0.3:
                    edges.append((agent_ids[i], agent_ids[j]))
                    edges.append((agent_ids[j], agent_ids[i]))
    
    elif pattern == 'chain':
        # Linear chain
        for i in range(num_agents - 1):
            edges.append((agent_ids[i], agent_ids[i + 1]))
            edges.append((agent_ids[i + 1], agent_ids[i]))
    
    elif pattern == 'ring':
        # Circular pattern
        for i in range(num_agents):
            next_i = (i + 1) % num_agents
            edges.append((agent_ids[i], agent_ids[next_i]))
    
    return edges


def _create_hetero_data(
    agent_ids: List[str],
    agent_roles: List[str],
    communication_edges: List[tuple],
    config: Config,
    logger: logging.Logger
) -> HeteroData:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: _create_hetero_data
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Create torch_geometric HeteroData structure with node types and edges
    
    STRUCTURE:
        Node types: 'solver', 'verifier', 'coordinator'
        Edge type: ('agent_type_a', 'communication', 'agent_type_b')
    
    INPUTS:
        agent_ids: List of agent IDs
        agent_roles: Role per agent (defines node type)
        communication_edges: Edges between agents
        config: Configuration with feature dimension
        logger: Logger
    
    OUTPUTS:
        HeteroData: torch_geometric heterogeneous graph
    
    EXAMPLE OUTPUT:
        HeteroData(
            solver={'x': tensor(...), ...},
            verifier={'x': tensor(...), ...},
            coordinator={'x': tensor(...), ...},
            ('solver', 'communication', 'verifier'): {
                'edge_index': tensor(...),
                ...
            },
            ...
        )
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    hetero_data = HeteroData()
    
    # Step 1: Count nodes per type
    node_types = sorted(set(agent_roles))
    nodes_per_type = {node_type: agent_roles.count(node_type) for node_type in node_types}
    
    logger.debug(f"Node types: {nodes_per_type}")
    
    # Step 2: Create node features for each type (random for now)
    feature_dim = config.node_feature_dim if hasattr(config, 'node_feature_dim') else 16
    
    for node_type in node_types:
        num_nodes = nodes_per_type[node_type]
        # Create random node features: (num_nodes, feature_dim)
        hetero_data[node_type].x = torch.randn(num_nodes, feature_dim, dtype=torch.float32)
    
    # Step 3: Create mapping from agent_id to (node_type, node_index_in_type)
    agent_id_to_node_idx = {}
    type_counters = {node_type: 0 for node_type in node_types}
    
    for agent_id, role in zip(agent_ids, agent_roles):
        node_idx = type_counters[role]
        agent_id_to_node_idx[agent_id] = (role, node_idx)
        type_counters[role] += 1
    
    # Step 4: Group edges by type and create edge_index tensors
    edges_by_type = {}
    
    for from_id, to_id in communication_edges:
        from_type, from_idx = agent_id_to_node_idx[from_id]
        to_type, to_idx = agent_id_to_node_idx[to_id]
        
        edge_key = (from_type, 'communication', to_type)
        
        if edge_key not in edges_by_type:
            edges_by_type[edge_key] = []
        
        edges_by_type[edge_key].append((from_idx, to_idx))
    
    # Step 5: Create edge tensors
    for edge_key, edges in edges_by_type.items():
        if edges:
            # Convert to (2, num_edges) format
            edge_from = [e for e in edges]
            edge_to = [e for e in edges]
            
            edge_index = torch.LongTensor([edge_from, edge_to])
            hetero_data[edge_key].edge_index = edge_index
            
            logger.debug(f"Edge type {edge_key}: {len(edges)} edges")
    
    return hetero_data


def hetero_data_to_dict(hetero_data: HeteroData) -> dict:
    """
    ════════════════════════════════════════════════════════════════════════════
    FUNCTION: hetero_data_to_dict
    ════════════════════════════════════════════════════════════════════════════
    
    PURPOSE:
        Convert HeteroData to dictionary for serialization
    
    INPUTS:
        hetero_data: torch_geometric HeteroData object
    
    OUTPUTS:
        Dictionary representation
    
    ════════════════════════════════════════════════════════════════════════════
    """
    
    result = {
        'node_types': hetero_data.node_types,
        'edge_types': hetero_data.edge_types,
        'node_features': {},
        'edges': {},
    }
    
    # Save node features
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x'):
            result['node_features'][node_type] = hetero_data[node_type].x.tolist()
    
    # Save edges
    for edge_type in hetero_data.edge_types:
        if hasattr(hetero_data[edge_type], 'edge_index'):
            result['edges'][str(edge_type)] = hetero_data[edge_type].edge_index.tolist()
    
    return result
