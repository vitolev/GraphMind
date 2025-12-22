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
        self.is_trained = False
        self.logger.info("Initializing GATNet model...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_dim = len(NODE_TYPES)  # one-hot encoding size
        hidden_dim = self.config.gnn_hidden_dim
        heads = self.config.gnn_num_heads 
        num_layers = self.config.gnn_num_layers
        dropout = self.config.gnn_dropout

        self.gat_layers = torch.nn.ModuleList()

        # First layer
        self.gat_layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))
        
        # Last GAT layer (output per node, concat=False)
        self.gat_layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        # MLP to map last-node embedding to scalar [0–1]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()  # output in [0,1]
        )

        self.to(self.device)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        # GAT layers
        for gat in self.gat_layers:
            x = F.elu(gat(x, edge_index))

        # Graph-level representation via super node
        super_indices = data.ptr[1:] - 1
        super_embeddings = x[super_indices]  

        # Final prediction
        score = self.mlp(super_embeddings)
        return score.squeeze(-1) 

    @torch.no_grad()
    def predict(self, data_list: List[Data]) -> List[float]:
        self.eval()
        if not self.is_trained:
            self.logger.warning("Model not trained yet, returning random predictions.")
            import random
            return [random.random() * 0.5 for _ in data_list]

        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False) 
        preds = [] 
        for _, batch in enumerate(loader): 
            batch = batch.to(self.device) 
            batch_pred = self(batch) 
            preds.extend(batch_pred.cpu().tolist())
            
        return preds

    def fit(self, dataset: List[Data]):
        """Train the GAT on a dataset of graphs"""
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
                pred = self(data).squeeze()  # [batch_size]
                loss = loss_fn(pred, data.y.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            avg_loss = total_loss / len(dataset)
            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True

        # Return final loss
        return avg_loss
    
def get_model(config: Config, logger: logging.Logger) -> GATNet:
    return GATNet(config, logger)