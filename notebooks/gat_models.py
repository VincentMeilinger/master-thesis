import torch.nn
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
    def __init__(self, metadata, hidden_channels, out_channels, num_heads=8, dropout_p=0.3):
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
        return x_dict


class HeteroGATEncoder1Conv2LinearDropout(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_heads=8, dropout_p=0.3):
        super().__init__()

        # GAT layer for each edge type with dropout
        self.conv = HeteroConv({
            edge_type: GATv2Conv(
                (-1, -1),
                hidden_channels,
                heads=num_heads,
                concat=True,
                dropout=dropout_p  # Apply dropout to attention coefficients
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        self.lin1 = torch.nn.Linear(hidden_channels * num_heads, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        # Dropout layer for node features
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Conv 1
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout after activation

        # Conv 2
        x_dict = {key: self.lin1(x) for key, x in x_dict.items()}
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout after activation

        # Final linear layer
        x_dict = {key: self.lin2(x) for key, x in x_dict.items()}

        return x_dict


class HomoGATEncoderLinearDropout(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads=8, dropout_p=0.3):
        super().__init__()

        # First GAT layer for each edge type with dropout
        self.conv_1 = GATv2Conv(
            (-1, -1),
            hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout_p  # Apply dropout to attention coefficients
        )

        # Second GAT layer for each edge type with dropout
        self.conv_2 = GATv2Conv(
            (-1, -1),
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout_p  # Apply dropout to attention coefficients
        )

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # First GAT layer
        x = self.conv_1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Second GAT layer
        x = self.conv_2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Final linear layer
        x = self.lin(x)
        #x = F.elu(x)

        return x

class HomoGATEncoderLinearDropout2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads=8, dropout_p=0.3):
        super().__init__()

        # First GAT layer for each edge type with dropout
        self.conv_1 = GATv2Conv(
            (-1, -1),
            hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout_p  # Apply dropout to attention coefficients
        )

        # Second GAT layer for each edge type with dropout
        self.conv_2 = GATv2Conv(
            (-1, -1),
            64,
            heads=1,
            concat=False,
            dropout=dropout_p  # Apply dropout to attention coefficients
        )

        self.lin = torch.nn.Linear(64, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # First GAT layer
        x = self.conv_1(x, edge_index)
        x = F.elu(x)

        # Second GAT layer
        x = self.conv_2(x, edge_index)
        x = F.elu(x)

        # Final linear layer
        x = self.lin(x)
        x = F.elu(x)

        return x


class HomoGATEncoderLinearDropoutHiddenChannels(torch.nn.Module):
    def __init__(self, hidden_channels_gat, hidden_channels_linear, out_channels, num_heads=8, dropout_p=0.3):
        super().__init__()

        # First GAT layer for each edge type with dropout
        self.conv_1 = GATv2Conv(
            (-1, -1),
            hidden_channels_gat,
            heads=num_heads,
            concat=True,
            dropout=dropout_p  # Apply dropout to attention coefficients
        )

        # Second GAT layer for each edge type with dropout
        self.conv_2 = GATv2Conv(
            (-1, -1),
            hidden_channels_linear,
            heads=1,
            concat=False,
            dropout=dropout_p  # Apply dropout to attention coefficients
        )

        self.lin = torch.nn.Linear(hidden_channels_linear, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # First GAT layer
        x = self.conv_1(x, edge_index)
        x = F.elu(x)

        # Second GAT layer
        x = self.conv_2(x, edge_index)
        x = F.elu(x)

        # Final linear layer
        x = self.lin(x)
        x = F.elu(x)

        return x


class HomoGATv2Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads=8, dropout_p=0.3):
        super().__init__()

        self.conv1 = GATv2Conv(
            (-1, -1),
            hidden_channels // num_heads,  # Output dim per head
            heads=num_heads,
            concat=True,
            dropout=dropout_p
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout_p
        )
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)  # Normalize embeddings for triplet loss

        return x