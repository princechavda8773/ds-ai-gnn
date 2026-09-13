import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """(b) Graph Encoding / Network Embedding. Swap gnn_type='gcn' or 'gat'."""

    def __init__(self, in_dim, hidden_dim, gnn_type="gcn", heads=2):
        super().__init__()
        self.gnn_type = gnn_type
        if gnn_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == "gat":
            self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
            self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        else:
            raise ValueError("gnn_type must be 'gcn' or 'gat'")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        return h  # one embedding vector per node in the (batched) subgraph


class SocialInfluenceModel(nn.Module):
    """
    Full model = (b)+(c) GNNEncoder -> pick ego-node embedding
                 -> (d) concat handcrafted features
                 -> (e) MLP prediction head -> sigmoid
    """

    def __init__(self, node_feat_dim, handcrafted_dim, hidden_dim=32, gnn_type="gcn"):
        super().__init__()
        self.encoder = GNNEncoder(node_feat_dim, hidden_dim, gnn_type=gnn_type)

        # (e) Prediction head. Input = GNN embedding of ego node + handcrafted feats.
        fused_dim = hidden_dim + handcrafted_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        node_embeddings = self.encoder(batch.x, batch.edge_index)  
        ego_embeddings = node_embeddings[batch.ptr[:-1] + batch.v_local]

        fused = torch.cat([ego_embeddings, batch.handcrafted], dim=1)  # (d)
        logits = self.head(fused).squeeze(-1)  # (e)
        return logits  # raw logits; apply sigmoid outside (BCEWithLogitsLoss does it for us)


if __name__ == "__main__":
    # Quick sanity check with fake data
    from torch_geometric.data import Data, Batch

    fake = [
        Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            y=torch.tensor([1.0]),
            v_local=torch.tensor([0]),
            handcrafted=torch.randn(1, 7),
        )
        for _ in range(4)
    ]
    batch = Batch.from_data_list(fake)
    model = SocialInfluenceModel(node_feat_dim=3, handcrafted_dim=7)
    out = model(batch)
    print("Model architecture:\n", model)
    print("\nOutput logits shape (should be [4]):", out.shape)
