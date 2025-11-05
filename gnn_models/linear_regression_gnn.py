import logging
import numpy as np
from typing import List, Dict, Any
from config.settings import Config
from torch_geometric.data import Data, HeteroData

class LinearRegressionGNN:
    """
    The model uses a simple linear regression for predicting graph scores.
    We will first extract basic features from the GNN graphs, then fit the linear regression
    model using the closed-form solution. You can look at _extract_features to see what 
    features are used, this will also be used as a baseline.
    """
    def __init__(self, config: Config, logger: logging.Logger):
        self.model_name = "LinearRegressionGNN"
        self.config = config
        self.logger = logger
        
        self.weights = None
        self.bias = None
        self.is_trained = False
        self.loss_history = []
        
        logger.info(f"Initialized {self.model_name}")
    
    def _extract_features(self, gnn_graphs: List[Any]) -> np.ndarray:
        features_list = []

        if isinstance(gnn_graphs[0], Data):
            # -- For PyG Data objects ---
            for graph in gnn_graphs:
                num_nodes = graph.num_nodes
                edge_index = graph.edge_index
                num_edges = len(edge_index)
                
                density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
                avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
                longest_length = 5 #TODO: compute the longest path lenght, that should be a useful feature
                
                features = [
                    float(num_nodes),
                    float(num_edges),
                    density,
                    avg_degree,
                    longest_length,
                ]
                
                features_list.append(features)
            
            return np.array(features_list, dtype=np.float32)

        elif isinstance(gnn_graphs[0], HeteroData):
            # -- For PyG HeteroData objects ---
            for graph in gnn_graphs:
                total_num_nodes = 0
                total_num_edges = 0
                
                for node_type in graph.node_types:
                    num_nodes = graph[node_type].num_nodes
                    total_num_nodes += num_nodes
                
                for edge_type in graph.edge_types:
                    edge_index = graph[edge_type].edge_index
                    num_edges = edge_index.size(1)
                    total_num_edges += num_edges
                
                density = total_num_edges / (total_num_nodes * (total_num_nodes - 1)) if total_num_nodes > 1 else 0
                avg_degree = 2 * total_num_edges / total_num_nodes if total_num_nodes > 0 else 0
                longest_length = 5 #TODO: compute the longest path lenght, that should be a useful feature. Should not be constant as it is now, because then we have colinearity with bias term and it messes up the regression
                
                features = [
                    float(total_num_nodes),
                    float(total_num_edges),
                    density,
                    avg_degree,
                    #longest_length,
                ]
                
                features_list.append(features)
            
            return np.array(features_list, dtype=np.float32)
        
        else:
            raise TypeError("Input graphs must be either PyG Data or HeteroData objects")
    
    def fit(self, training_data: List[Any]) -> float:
        if not training_data:
            raise ValueError("Training data is empty")
        
        targets = np.array([d.y for d in training_data])

        X = self._extract_features(training_data)
        y = targets
        
        X_with_bias = np.column_stack([X, np.ones(len(X))])
        
        # Using closed form solution for linear regression
        try:
            XtX_inv = np.linalg.inv(X_with_bias.T @ X_with_bias)
            weights_with_bias = XtX_inv @ X_with_bias.T @ y
            
            self.weights = weights_with_bias[:-1]
            self.bias = weights_with_bias[-1]
            
            predictions = X @ self.weights + self.bias
            mse_loss = np.mean((predictions - y) ** 2)
            
            self.is_trained = True
            self.loss_history.append(mse_loss)
            
            self.logger.debug(
                f"LinearRegressionGNN training complete "
                f"(samples: {len(training_data)}, loss: {mse_loss:.4f})"
            )
            
            return mse_loss
            
        except np.linalg.LinAlgError:
            self.logger.error("Singular matrix during linear regression fit")
            return 0.0
    
    def predict(self, gnn_graphs: List[Any]) -> List[float]:
        if not self.is_trained:
            self.logger.warning("Model not trained yet, returning random predictions")
            import random
            return [random.random() for _ in gnn_graphs]
        
        X = self._extract_features(gnn_graphs)
        
        predictions = X @ self.weights + self.bias
        
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions.flatten().tolist()
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'name': self.model_name,
            'weights': self.weights,
            'bias': self.bias,
            'is_trained': self.is_trained,
            'loss_history': self.loss_history,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        self.weights = state.get('weights')
        self.bias = state.get('bias')
        self.is_trained = state.get('is_trained', False)
        self.loss_history = state.get('loss_history', [])
    
    def get_latest_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else None


def get_model(config: Config, logger: logging.Logger) -> LinearRegressionGNN:
    return LinearRegressionGNN(config, logger)
