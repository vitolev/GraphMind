"""
Serialize LangGraph objects to JSON-compatible format

The goal is to convert LangGraph objects (which contain functions, etc.)
into serializable dictionaries that can be stored and processed.
"""

import logging
import json
from typing import List, Dict, Any

def serialize_langgraph_batch(
    langgraph_list: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Serialize batch of LangGraph objects to JSON-compatible format
    
    Args:
        langgraph_list: List of LangGraph objects
        logger: Logger
    
    Returns:
        List of serialized graph dictionaries
    """
    
    logger.debug(f"Serializing {len(langgraph_list)} graphs")
    
    serialized = []
    for graph in langgraph_list:
        try:
            ser_graph = _serialize_single_graph(graph)
            serialized.append(ser_graph)
        except Exception as e:
            logger.warning(f"Failed to serialize graph {graph.get('id', '?')}: {e}")
            continue
    
    logger.debug(f"Serialized {len(serialized)} graphs")
    return serialized

def _serialize_single_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize a single LangGraph object
    
    Args:
        graph: LangGraph object with structure
    
    Returns:
        Serialized dictionary
    """
    
    return {
        'id': graph.get('id'),
        'num_agents': graph.get('num_agents'),
        'agent_roles': graph.get('agent_roles'),
        'communication_edges': graph.get('communication_edges'),
        'graph_data': graph.get('graph_data'),
    }

def deserialize_langgraph_batch(
    serialized_list: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Deserialize batch of graphs back to LangGraph format
    
    Args:
        serialized_list: List of serialized graphs
        logger: Logger
    
    Returns:
        List of LangGraph objects
    """
    
    logger.debug(f"Deserializing {len(serialized_list)} graphs")
    
    deserialized = []
    for ser_graph in serialized_list:
        try:
            graph = _deserialize_single_graph(ser_graph)
            deserialized.append(graph)
        except Exception as e:
            logger.warning(f"Failed to deserialize graph: {e}")
            continue
    
    logger.debug(f"Deserialized {len(deserialized)} graphs")
    return deserialized

def _deserialize_single_graph(ser_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize a single graph
    
    Args:
        ser_graph: Serialized graph dictionary
    
    Returns:
        LangGraph object
    """
    return {
        'id': ser_graph.get('id'),
        'num_agents': ser_graph.get('num_agents'),
        'agent_roles': ser_graph.get('agent_roles'),
        'communication_edges': ser_graph.get('communication_edges'),
        'graph_data': ser_graph.get('graph_data'),
    }
