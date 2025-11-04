"""
================================================================================
FILE: graph_generation/langgraph_generator.py
================================================================================

PURPOSE:
    Generate multiagent system architectures as Graph objects

LAST UPDATED: 2025-11-04
STATUS: active
================================================================================
"""

import logging
import random
from typing import List
from config.settings import Config
from data_management.graph import Graph


def generate_langgraph_variants(
    num_graphs: int,
    config: Config,
    logger: logging.Logger
) -> List[Graph]:
    """
    Generate N LangGraph variants as Graph objects
    
    Args:
        num_graphs: Number of graphs to generate (e.g., 100,000)
        config: Configuration object
        logger: Logger
    
    Returns:
        List of Graph objects (not dicts!)
    """
    
    if config.seed is not None:
        random.seed(config.seed)
    
    logger.debug(f"Generating {num_graphs} LangGraph variants as Graph objects")
    
    graphs = []
    for i in range(num_graphs):
        graph = _create_single_graph(config, i, logger)
        graphs.append(graph)
    
    logger.debug(f"Generated {len(graphs)} graphs")
    return graphs

def _create_single_graph(
    config: Config,
    graph_id: int,
    logger: logging.Logger
) -> Graph:
    """
    Create a single LangGraph variant as a Graph object
    
    Args:
        config: Configuration
        graph_id: Unique ID for this graph
        logger: Logger
    
    Returns:
        Graph object (not dict!)
    """
    
    # Randomly choose number of agents
    num_agents = random.randint(config.num_agents_min, config.num_agents_max)
    
    # Create agent IDs and roles
    agent_ids = [f'agent_{i}' for i in range(num_agents)]
    agent_roles = [
        random.choice(config.agent_roles)
        for _ in range(num_agents)
    ]
    
    # Create communication edges (between agent IDs, not indices!)
    communication_edges = _generate_communication_edges(agent_ids, config)
    
    # Create Graph object directly
    graph = Graph(
        graph_id=f'graph_{graph_id}',
        agent_ids=agent_ids,
        agent_roles=agent_roles,
        communication_edges=communication_edges,
        _logger=logger
    )
    
    return graph


def _generate_communication_edges(agent_ids: List[str], config: Config) -> List[tuple]:
    """
    Generate communication pattern between agents
    
    Args:
        agent_ids: List of agent IDs (e.g., ['agent_0', 'agent_1', ...])
        config: Configuration
    
    Returns:
        List of (agent_id, agent_id) edges
    """
    
    pattern = random.choice(['hub', 'mesh', 'chain'])
    edges = []
    num_agents = len(agent_ids)
    
    if pattern == 'hub':
        # Star pattern: agent_0 is hub
        for i in range(1, num_agents):
            edges.append((agent_ids, agent_ids[i]))
            edges.append((agent_ids[i], agent_ids))
    
    elif pattern == 'mesh':
        # Fully connected with random probability
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and random.random() < 0.5:
                    edges.append((agent_ids[i], agent_ids[j]))
    
    elif pattern == 'chain':
        # Linear chain
        for i in range(num_agents - 1):
            edges.append((agent_ids[i], agent_ids[i + 1]))
            edges.append((agent_ids[i + 1], agent_ids[i]))
    
    return edges
