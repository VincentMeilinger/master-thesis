import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, embedding_dim):
        super(GAT, self).__init__()

        # First layer:
        #   - use 8 attention heads
        #   - output 8 features per head
        self.conv1 = GATv2Conv(node_feature_dim, 8, heads=8, dropout=0.6, edge_dim=edge_feature_dim)

        # Second layer:
        #   - use a single attention head
        #   - output num_classes features per node
        self.conv2 = GATv2Conv(8 * 8, embedding_dim, heads=1, concat=False, dropout=0.6, edge_dim=edge_feature_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        # Directly use the output of the second layer as node embeddings
        return x
