import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.0, add_self_loops=True, edge_dim=None):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat, negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(out_channels*heads if concat else out_channels, out_channels, heads=heads, concat=False, negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x
