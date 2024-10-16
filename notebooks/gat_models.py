from torch_geometric.nn import GATv2Conv, HeteroConv

from src.shared.graph_schema import *


class HeteroGATEncoderLinear(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads=8):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: GATv2Conv(
                (-1, -1), hidden_channels, heads=num_heads, concat=True)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: GATv2Conv(
                (-1, -1), out_channels, heads=1, concat=False)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.lin = torch.nn.Linear(out_channels, out_channels)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Conv 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # Conv 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # Final linear layer
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}

        return x_dict


class HeteroGATEncoderLinearDropout(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads=8, dropout_p=0.5):
        super().__init__()

        # First GAT layer for each edge type with dropout
        self.conv1 = HeteroConv({
            edge_type: GATv2Conv(
                (-1, -1),
                hidden_channels,
                heads=num_heads,
                concat=True,
                dropout=dropout_p  # Apply dropout to attention coefficients
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        # Second GAT layer for each edge type with dropout
        self.conv2 = HeteroConv({
            edge_type: GATv2Conv(
                (-1, -1),
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout_p  # Apply dropout to attention coefficients
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        self.lin = torch.nn.Linear(out_channels, out_channels)

        # Dropout layer for node features
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Conv 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout after activation

        # Conv 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout after activation

        # Final linear layer
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Optional: Apply dropout after linear layer

        return x_dict