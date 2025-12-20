"""Agent state definitions for LangGraph multi-agent system."""

import operator
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TypedDict, Annotated


@dataclass
class GlobalKnowledge:
    """Global knowledge store shared across all agents."""
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
    global_knowledge: GlobalKnowledge
    result: Annotated[list, operator.add]
    node_type: Optional[str]
    node_id: Optional[int]
    solution: Optional[str]

