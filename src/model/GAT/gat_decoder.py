import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATDecoder(nn.Module):
    def __init__(self, num_features, num_edge_features, embedding_dim, num_node_types, num_edge_types):
        super(GATDecoder, self).__init__()
        self.gat1 = GATv2Conv(num_features, 8, heads=8, dropout=0.6, edge_dim=num_edge_features)
        self.gat2 = GATv2Conv(8 * 8, embedding_dim, heads=1, concat=False, dropout=0.6, edge_dim=num_edge_features)

        # Decoders
        self.node_type_decoder = nn.Linear(embedding_dim, num_node_types)
        self.edge_type_decoder = nn.Linear(2 * embedding_dim, num_edge_types)  # Concatenation of two node embeddings

    def forward(self, x, edge_index, edge_attr):
        # Node embeddings
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)

        # Predict node types
        node_types = self.node_type_decoder(x)

        # Predict edge types
        edge_types = self.edge_type_decoder(torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1))

        return x, node_types, edge_types
