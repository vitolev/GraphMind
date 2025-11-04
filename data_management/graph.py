import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np


@dataclass
class Graph:
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: IDENTIFICATION & TIMESTAMPS
    # ═══════════════════════════════════════════════════════════════════════════
    
    graph_id: str
    iteration: Optional[int] = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: AGENT STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    agent_types: List[str] = field(default_factory=list)
    communication_edges: List[Tuple[str, str]] = field(default_factory=list)
    

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: FEATURES (For GNN)
    # ═══════════════════════════════════════════════════════════════════════════
    
    node_features: Optional[np.ndarray] = None
    graph_features: Dict[str, Any] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: PREDICTIONS (From GNN)
    # ═══════════════════════════════════════════════════════════════════════════
    
    gnn_predicted_score: Optional[float] = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: EVALUATIONS (From LLM)
    # ═══════════════════════════════════════════════════════════════════════════
    
    llm_actual_score: Optional[float] = None
    llm_evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    _logger: Optional[logging.Logger] = None
    
    def __post_init__(self):
        """Validate graph after initialization"""
        if len(self.agent_ids) == 0:
            raise ValueError("Graph must have at least one agent")
        
        if len(self.agent_roles) != len(self.agent_ids):
            raise ValueError(
                f"agent_roles length ({len(self.agent_roles)}) "
                f"!= agent_ids length ({len(self.agent_ids)})"
            )
        
        if self.node_features is not None and self.node_features.shape != len(self.agent_ids):
            raise ValueError(
                f"node_features rows ({self.node_features.shape}) "
                f"!= num_agents ({len(self.agent_ids)})"
            )
        
        if not self.created_at:
            self.created_at = datetime.now().timestamp()
        
        if not self.graph_features:
            self._compute_graph_features()
    
    def _compute_graph_features(self) -> None:
        """Compute graph-level statistics"""
        num_agents = len(self.agent_ids)
        num_edges = len(self.communication_edges)
        density = num_edges / (num_agents * (num_agents - 1)) if num_agents > 1 else 0
        avg_degree = 2 * num_edges / num_agents if num_agents > 0 else 0
        
        unique_roles = set(self.agent_roles)
        unique_types = set(self.agent_types) if self.agent_types else set()
        
        self.graph_features = {
            'num_agents': num_agents,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': avg_degree,
            'num_unique_roles': len(unique_roles),
            'num_unique_types': len(unique_types),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def set_gnn_prediction(self, score: float) -> None:
        """
        Set GNN predicted score
        
        Args:
            score: Predicted score (0-1)
        """
        if not 0.0 <= score <= 1.0:
            if self._logger:
                self._logger.warning(f"GNN score {score} outside [0, 1], clipping")
            score = max(0.0, min(1.0, score))
        
        self.gnn_predicted_score = float(score)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def set_llm_evaluation(
        self,
        actual_score: float,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Set LLM evaluation result
        
        Args:
            actual_score: Actual performance (0-1)
            metadata: Evaluation details (num_correct, execution_time, etc.)
        """
        if not 0.0 <= actual_score <= 1.0:
            if self._logger:
                self._logger.warning(f"LLM score {actual_score} outside [0, 1], clipping")
            actual_score = max(0.0, min(1.0, actual_score))
        
        self.llm_actual_score = float(actual_score)
        
        if metadata:
            self.llm_evaluation_metadata = metadata
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVERSIONS TO GNN FORMATS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_homogeneous(self) -> Dict[str, Any]:
        """
        ════════════════════════════════════════════════════════════════
        FUNCTION: to_homogeneous
        ════════════════════════════════════════════════════════════════
        
        PURPOSE:
            Convert to simplest possible graph format:
            - All nodes are uniform (no node types)
            - All edges are uniform (one edge type)
            - Ready for standard GNN models
        
        OUTPUT:
            Dict with:
            {
                'node_features': np.ndarray (num_agents, feature_dim),
                'edge_index': List[(from_idx, to_idx)],
                'num_nodes': int,
                'graph_features': Dict,
                'graph_id': str,
            }
        
        NOTES:
            - Edge indices are agent indices (0, 1, 2, ...)
            - No node/edge type information
            - Best for simple GNN architectures
        
        ════════════════════════════════════════════════════════════════
        """
        
        # Create mapping from agent_ids to indices
        agent_id_to_idx = {aid: idx for idx, aid in enumerate(self.agent_ids)}
        
        # Convert communication edges to index pairs
        edge_index = []
        for from_id, to_id in self.communication_edges:
            from_idx = agent_id_to_idx.get(from_id)
            to_idx = agent_id_to_idx.get(to_id)
            if from_idx is not None and to_idx is not None:
                edge_index.append((from_idx, to_idx))
        
        # Use provided node_features or create simple ones (one-hot roles)
        if self.node_features is not None:
            node_features = self.node_features
        else:
            # Simple: one-hot encode roles
            unique_roles = sorted(set(self.agent_roles))
            role_to_idx = {role: idx for idx, role in enumerate(unique_roles)}
            
            node_features = np.zeros((len(self.agent_ids), len(unique_roles)), dtype=np.float32)
            for agent_idx, role in enumerate(self.agent_roles):
                role_idx = role_to_idx[role]
                node_features[agent_idx, role_idx] = 1.0
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'num_nodes': len(self.agent_ids),
            'graph_features': self.graph_features,
            'graph_id': self.graph_id,
        }
    
    def to_heterogeneous(self) -> Dict[str, Any]:
        """
        ════════════════════════════════════════════════════════════════
        FUNCTION: to_heterogeneous
        ════════════════════════════════════════════════════════════════
        
        PURPOSE:
            Convert to heterogeneous graph format:
            - Nodes have types (from agent_types)
            - Edges can have types
            - Ready for HetGAT, HAN models
        
        OUTPUT:
            Dict with:
            {
                'node_data': {
                    'node_types': List[str],
                    'node_features': np.ndarray,
                    'unique_types': List[str],
                },
                'edge_data': {
                    'edge_index': List[(from_idx, to_idx)],
                    'edge_types': List[str] (all same for now),
                },
                'num_nodes': int,
                'graph_features': Dict,
                'graph_id': str,
            }
        
        NOTES:
            - For now, edge types are all 'communication'
            - Can extend to multiple edge types later
            - Better for advanced GNN architectures
        
        ════════════════════════════════════════════════════════════════
        """
        
        # Use agent_types if available, otherwise use roles
        if self.agent_types and len(self.agent_types) == len(self.agent_ids):
            node_types = self.agent_types
        else:
            node_types = self.agent_roles
        
        unique_types = sorted(set(node_types))
        
        # Create mapping from agent_ids to indices
        agent_id_to_idx = {aid: idx for idx, aid in enumerate(self.agent_ids)}
        
        # Convert communication edges to index pairs
        edge_index = []
        edge_types = []
        for from_id, to_id in self.communication_edges:
            from_idx = agent_id_to_idx.get(from_id)
            to_idx = agent_id_to_idx.get(to_id)
            if from_idx is not None and to_idx is not None:
                edge_index.append((from_idx, to_idx))
                edge_types.append('communication')  # All same type for now
        
        # Use provided node_features or create simple ones
        if self.node_features is not None:
            node_features = self.node_features
        else:
            # Simple: one-hot encode types
            type_to_idx = {t: idx for idx, t in enumerate(unique_types)}
            
            node_features = np.zeros((len(self.agent_ids), len(unique_types)), dtype=np.float32)
            for agent_idx, node_type in enumerate(node_types):
                type_idx = type_to_idx[node_type]
                node_features[agent_idx, type_idx] = 1.0
        
        return {
            'node_data': {
                'node_types': node_types,
                'node_features': node_features,
                'unique_types': unique_types,
            },
            'edge_data': {
                'edge_index': edge_index,
                'edge_types': edge_types,
            },
            'num_nodes': len(self.agent_ids),
            'graph_features': self.graph_features,
            'graph_id': self.graph_id,
        }
    
    def to_langgraph(self) -> Dict[str, Any]:
        """
        Convert to LangGraph format for execution
        
        (Placeholder - will implement when needed)
        """
        return {
            'agents': self.agent_ids,
            'roles': self.agent_roles,
            'edges': self.communication_edges,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING FORMAT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_training_sample(self) -> Dict[str, Any]:
        """
        Convert to training dataset format
        
        Returns:
            {
                'graph': self.to_homogeneous(),
                'actual_score': float,
                'metadata': {...}
            }
        """
        if self.llm_actual_score is None:
            raise ValueError(f"Graph {self.graph_id} not evaluated - no llm_actual_score")
        
        return {
            'graph': self.to_homogeneous(),
            'actual_score': self.llm_actual_score,
            'metadata': {
                'graph_id': self.graph_id,
                'gnn_predicted_score': self.gnn_predicted_score,
                'evaluation_metadata': self.llm_evaluation_metadata,
            }
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'graph_id': self.graph_id,
            'created_at': self.created_at,
            'iteration': self.iteration,
            'agent_ids': self.agent_ids,
            'agent_roles': self.agent_roles,
            'agent_types': self.agent_types,
            'communication_edges': self.communication_edges,
            'node_features': self.node_features.tolist() if self.node_features is not None else None,
            'graph_features': self.graph_features,
            'gnn_predicted_score': self.gnn_predicted_score,
            'llm_actual_score': self.llm_actual_score,
            'llm_evaluation_metadata': self.llm_evaluation_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], logger: logging.Logger = None) -> 'Graph':
        """Deserialize from dictionary"""
        node_features = None
        if data.get('node_features') is not None:
            node_features = np.array(data['node_features'])
        
        return cls(
            graph_id=data['graph_id'],
            created_at=data.get('created_at'),
            iteration=data.get('iteration'),
            agent_ids=data['agent_ids'],
            agent_roles=data['agent_roles'],
            agent_types=data.get('agent_types', []),
            communication_edges=data['communication_edges'],
            node_features=node_features,
            graph_features=data.get('graph_features', {}),
            gnn_predicted_score=data.get('gnn_predicted_score'),
            llm_actual_score=data.get('llm_actual_score'),
            llm_evaluation_metadata=data.get('llm_evaluation_metadata', {}),
            _logger=logger,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Graph({self.graph_id}): "
            f"agents={len(self.agent_ids)}, edges={len(self.communication_edges)}, "
            f"gnn_pred={self.gnn_predicted_score}, llm_actual={self.llm_actual_score}"
        )
