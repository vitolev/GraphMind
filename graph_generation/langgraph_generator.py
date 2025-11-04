import logging
import random
from typing import List, Dict, Any
from config.settings import Config

def generate_langgraph_variants(
    num_graphs: int,
    config: Config,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Generate N LangGraph variants
    
    Args:
        num_graphs: Number of graphs to generate (e.g., 100,000)
        config: Configuration object
        logger: Logger
    
    Returns:
        List of graph dictionaries
        Each graph has:
        {
            'id': str,
            'num_agents': int,
            'agent_roles': List[str],
            'communication_edges': List[Tuple],
            'graph_data': Any,  # Raw graph structure
        }
    """
    
    if config.seed is not None:
        random.seed(config.seed)
    
    logger.debug(f"Generating {num_graphs} LangGraph variants")
    
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
) -> Dict[str, Any]:
    """
    Create a single LangGraph variant
    
    Args:
        config: Configuration
        graph_id: Unique ID for this graph
        logger: Logger
    
    Returns:
        Graph dictionary
    """
    
    # Randomly choose number of agents
    num_agents = random.randint(config.num_agents_min, config.num_agents_max)
    
    # Randomly assign roles to agents
    agent_roles = [
        random.choice(config.agent_roles)
        for _ in range(num_agents)
    ]
    
    # Create communication edges
    communication_edges = _generate_communication_edges(num_agents, config)
    
    # Create graph structure
    graph_data = {
        'num_agents': num_agents,
        'agent_roles': agent_roles,
        'communication_edges': communication_edges,
        'coordination_strategy': random.choice(['hub', 'mesh', 'chain', 'mixed']),
    }
    
    graph = {
        'id': f'graph_{graph_id}',
        'num_agents': num_agents,
        'agent_roles': agent_roles,
        'communication_edges': communication_edges,
        'graph_data': graph_data,
    }
    
    return graph

def _generate_communication_edges(num_agents: int, config: Config) -> List[tuple]:
    """
    Generate communication pattern between agents
    
    Different patterns: hub (star), mesh (fully connected), chain, etc.
    
    Args:
        num_agents: Number of agents
        config: Configuration
    
    Returns:
        List of (source, target) edges
    """
    
    pattern = random.choice(['hub', 'mesh', 'chain'])
    edges = []
    
    if pattern == 'hub':
        # Star pattern: agent 0 is hub
        for i in range(1, num_agents):
            edges.append((0, i))
            edges.append((i, 0))
    
    elif pattern == 'mesh':
        # Fully connected
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and random.random() < 0.5:  # 50% chance
                    edges.append((i, j))
    
    elif pattern == 'chain':
        # Linear chain
        for i in range(num_agents - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
    
    return edges
