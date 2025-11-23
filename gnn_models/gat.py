import logging
from config.settings import Config
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Dict, Any
from config.nodes import NODE_TYPES

class GATNet(torch.nn.Module):
    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.logger.info("Initializing GATNet model...")


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_dim = len(NODE_TYPES)  # one-hot encoding size
        hidden_dim = self.config.gnn_hidden_dim
        heads = self.config.gnn_num_heads 
        num_layers = self.config.gnn_num_layers

        self.gat_layers = torch.nn.ModuleList()

        # First layer
        self.gat_layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))

        # Last GAT layer (output per node, concat=False)
        self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))

        # MLP to map last-node embedding → scalar [0–1]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()  # output in [0,1]
        )

        self.to(self.device)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        # GAT layers
        for gat in self.gat_layers:
            x = F.elu(gat(x, edge_index))

        # extract embedding of final node
        final_node_idx = data.final_node.item()   # scalar index
        graph_embedding = x[final_node_idx]

        # predict scalar
        score = self.mlp(graph_embedding)
        return score

    @torch.no_grad()
    def predict(self, data_list):
        self.eval()
        preds = []
        for data in data_list:
            data = data.to(self.device)
            pred = self(data).item()
            preds.append(pred)
        return preds

    def fit(self, dataset: List[Data]):
        """Train the GAT on a dataset of graphs"""
        self.train()
        epochs = self.config.gnn_epochs
        batch_size = self.config.gnn_batch_size
        lr = self.config.gnn_learning_rate

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()  # assuming regression

        for epoch in range(1, epochs + 1):
            total_loss = 0
            for data in loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                pred = self(data).squeeze()  # [batch_size]
                loss = loss_fn(pred, data.y.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            avg_loss = total_loss / len(dataset)
            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Return final loss
        return avg_loss
    
def get_model(config: Config, logger: logging.Logger) -> GATNet:
    return GATNet(config, logger)