import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=5, concat=True, negative_slope=0.2, dropout=0.0, add_self_loops=True, edge_dim=None):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=concat, negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(heads * hidden_channels if concat else hidden_channels, hidden_channels, heads=heads, concat=True, negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
        self.linear_output = torch.nn.Sequential(
            torch.nn.Linear(heads * hidden_channels, hidden_channels),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)

        x = self.linear_output(x)
        return x
