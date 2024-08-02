import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear


class GATv2Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATv2Decoder, self).__init__()
        self.linear1 = Linear(in_channels, in_channels // 2)
        self.linear2 = Linear(in_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear1(x)
        x = torch.nn.functional.elu(x)
        x = self.linear2(x)
        return x
