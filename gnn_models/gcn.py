import logging
from config.settings import Config
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Dict, Any
from config.nodes import NODE_TYPES


class GCNNet(torch.nn.Module):
    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.is_trained = False
        self.logger.info("Initializing GCNNet model...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_dim = len(NODE_TYPES)  # one-hot feature size
        hidden_dim = self.config.gnn_hidden_dim
        num_layers = self.config.gnn_num_layers
        dropout = self.config.gnn_dropout

        # Build layers
        self.gcn_layers = torch.nn.ModuleList()

        # First layer
        self.gcn_layers.append(GCNConv(in_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Last GCN layer (produces final per-node embedding)
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # MLP head for graph-level prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        # GCN layers
        for gcn in self.gcn_layers:
            x = F.relu(gcn(x, edge_index))
            x = self.dropout(x)

        # Graph-level representation via super node
        super_indices = data.super_node_idx.to(self.device)
        super_embeddings = x[super_indices]  

        # Final prediction
        score = self.mlp(super_embeddings)
        return score.squeeze() 

    @torch.no_grad()
    def predict(self, data_list: List[Data]) -> List[float]:
        self.eval()
        if not self.is_trained:
            self.logger.warning("Model not trained yet, returning random predictions.")
            import random
            return [random.random() for _ in data_list]

        preds = []
        for data in data_list:
            data = data.to(self.device)
            pred = self(data).item()
            preds.append(pred)

        return preds

    def fit(self, dataset: List[Data]):
        """Train the GCN on a dataset of graphs"""
        self.train()
        epochs = self.config.gnn_epochs
        batch_size = self.config.gnn_batch_size
        lr = self.config.gnn_learning_rate

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0
            for data in loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                pred = self(data).squeeze()
                loss = loss_fn(pred, data.y.to(self.device))

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            avg_loss = total_loss / len(dataset)
            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True
        return avg_loss


def get_model(config: Config, logger: logging.Logger) -> GCNNet:
    return GCNNet(config, logger)
