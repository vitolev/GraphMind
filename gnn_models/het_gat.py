import logging
from config.settings import Config
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from typing import List
from config.nodes import NODE_TYPES, EDGES

class HetGATNet(torch.nn.Module):
    def __init__(self, config: Config, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.is_trained = False
        self.logger.info("Initializing HetGATNet model...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_dim = self.config.gnn_hidden_dim
        self.heads = self.config.gnn_num_heads
        self.num_layers = self.config.gnn_num_layers
        self.dropout = self.config.gnn_dropout

        # Heterogeneous GAT layers
        self.gat_layers = torch.nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = self.hidden_dim * self.heads if layer_idx > 0 else 1

            conv_dict = {}
            for (src_type, dst_type) in EDGES:
                # Use GATConv on existing node features
                conv_dict[(src_type, "to", dst_type)] = GATConv(
                    in_channels=in_dim,
                        out_channels=self.hidden_dim,
                        heads=self.heads,
                        concat=True,
                        dropout=self.dropout,
                        add_self_loops=False
                    )
            
            # Add additional edges to 'super' node
            for ntype in NODE_TYPES:    
                conv_dict[(ntype, "to", "super")] = GATConv(
                    in_channels=in_dim,
                    out_channels=self.hidden_dim,
                    heads=self.heads,
                    concat=True,
                    dropout=self.dropout,
                    add_self_loops=False
                )

            self.gat_layers.append(HeteroConv(conv_dict, aggr='mean'))

        # Final MLP to scalar per graph
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * self.heads, 1),
            torch.nn.Sigmoid()
        )

        self.to(self.device)

    def forward(self, data: HeteroData):
        # ============================================================
        # 1) Build x_dict for all node types (None if not in graph)
        # ============================================================
        x_dict = {}
        for ntype in NODE_TYPES:
            x = data[ntype].x
            x_dict[ntype] = x if x.numel() > 0 else None

        # Manually add 'super' node
        x_dict['super'] = data['super'].x

        # ============================================================
        # 2) Pass through hetero GAT layers
        # ============================================================
        for i, conv in enumerate(self.gat_layers):

            # Filter edge types that exist AND whose src/dst embeddings exist
            filtered_edges = {}
            for etype in conv.convs.keys():
                if etype not in data.edge_index_dict:
                    continue

                src, _, dst = etype
                if x_dict[src] is None or x_dict[dst] is None:
                    continue

                filtered_edges[etype] = data.edge_index_dict[etype]

            # Apply GAT
            out_dict = conv(x_dict, filtered_edges)

            # Update only the node types that were computed
            for ntype in out_dict:
                out_dict[ntype] = F.elu(out_dict[ntype])

            # Merge with existing features
            x_dict = {ntype: out_dict.get(ntype, x_dict[ntype]) for ntype in x_dict}

            if i == 0:  # after first layer
                H = self.hidden_dim * self.heads
                x_dict['START'] = torch.ones((x_dict['START'].shape[0], H), device=self.device)     # Manually convert START node embedding to correct size

        # ============================================================
        # 3) Use super node as graph-level embedding
        # ============================================================
        super_emb = x_dict["super"]
        
        return self.mlp(super_emb).view(-1)

    @torch.no_grad()
    def predict(self, data_list: List[HeteroData]) -> List[float]:
        self.eval()
        if not self.is_trained:
            self.logger.warning("Model not trained yet, returning random predictions.")
            import random
            return [random.random() * 0.5 for _ in data_list]

        loader = DataLoader(data_list, batch_size=1024, shuffle=False)
        preds = []

        for i, batch in enumerate(loader):
            self.logger.debug(f"Predicting batch {i}/{len(loader)}")
            
            batch = batch.to(self.device)
            batch_pred = self(batch)
            preds.extend(batch_pred.view(-1).cpu().tolist())

        return preds

    def fit(self, dataset: List[HeteroData]):
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
                optimizer.zero_grad()
                data = data.to(self.device)
                pred = self(data).view(-1)
                loss = loss_fn(pred, data.y.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(data)

            avg_loss = total_loss / len(dataset)
            if epoch % 10 == 0 or epoch == 1:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True
        return avg_loss


def get_model(config: Config, logger: logging.Logger) -> HetGATNet:
    return HetGATNet(config, logger)
