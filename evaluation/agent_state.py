"""Agent state definitions for LangGraph multi-agent system."""

import operator
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TypedDict, Annotated, List, Tuple


@dataclass
class GraphStructure:
    """Graph topology stored in state for knowledge flow."""
    nodes: List[Tuple[int, str]]  # (node_id, node_type)
    edges: List[Tuple[int, int]]  # (src, dst)
    graph_in: Dict[int, List[int]]  # node_id -> [incoming_node_ids]
    graph_out: Dict[int, List[int]]  # node_id -> [outgoing_node_ids]
    
    def get_incoming_nodes(self, node_id: int) -> List[int]:
        """Get list of node IDs that have edges into this node."""
        return self.graph_in.get(node_id, [])
    
    def get_outgoing_nodes(self, node_id: int) -> List[int]:
        """Get list of node IDs that this node has edges to."""
        return self.graph_out.get(node_id, [])
    
    def get_node_type(self, node_id: int) -> Optional[str]:
        """Get the type of a node by its ID."""
        for nid, ntype in self.nodes:
            if nid == node_id:
                return ntype
        return None


@dataclass
class ScopedKnowledge:
    """Knowledge isolated within a specific scope."""
    scope_id: str
    node_data: Dict[int, Any] = field(default_factory=dict)  # node_id -> output data
    
    def add(self, node_id: int, data: Any):
        """Add knowledge from a node in this scope."""
        self.node_data[node_id] = data
    
    def get(self, node_id: int) -> Optional[Any]:
        """Get knowledge from a node in this scope."""
        return self.node_data.get(node_id)


@dataclass
class GlobalKnowledge:
    """Global knowledge store shared across all agents (legacy - kept for compatibility)."""
    entries: Dict[int, Any] = field(default_factory=dict)
    
    def add(self, node_id: int, data: Any):
        """Add an entry to global knowledge."""
        self.entries[node_id] = data
    
    def get(self, node_id: int) -> Optional[Any]:
        """Get an entry from global knowledge."""
        return self.entries.get(node_id)


class AgentState(TypedDict):
    """State structure for LangGraph agents."""
    problem: Annotated[list, operator.add]
    global_knowledge: GlobalKnowledge  # Legacy - kept for backward compatibility
    graph_structure: Optional[GraphStructure]  # NEW: Graph topology
    scoped_knowledge: Dict[str, ScopedKnowledge]  # NEW: scope_id -> ScopedKnowledge
    scope_mapping: Dict[int, str]  # NEW: node_id -> scope_id (set by split nodes)
    current_scope: str  # NEW: Default scope (usually "root")
    result: Annotated[list, operator.add]
    node_type: Optional[str]
    node_id: Optional[int]
    solution: Optional[str]

