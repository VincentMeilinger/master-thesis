import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.nn import Linear


class GATv2Encoder(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            edge_feature_dim,
            edge_types,
            node_types,
            heads=5,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True
    ):
        super(GATv2Encoder, self).__init__()

        self.conv_1 = HeteroConv({
            edge_type: GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_feature_dim
            )
            for edge_type in edge_types
        }, aggr='mean')

        self.conv_2 = HeteroConv({
            edge_type: GATv2Conv(
                in_channels=heads * hidden_channels if concat else hidden_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_feature_dim
            )
            for edge_type in edge_types
        }, aggr='mean')

        self.lin_out = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_out[node_type] = torch.nn.Sequential(
                Linear(heads * hidden_channels, hidden_channels),
                torch.nn.Dropout(dropout),
                Linear(hidden_channels, out_channels)
            )

    def forward(self, x_dict, edge_index_dict, edge_feature_dict):
        """
        :param x_dict: dict of torch.Tensor
            Node feature vectors for each node type.
        :param edge_index_dict: dict of torch.Tensor
            Edge indices for each edge type.
        :param edge_feature_dict: dict of torch.Tensor
            Edge attribute vectors for each edge type.
        """

        x_dict = self.conv_1(x_dict, edge_index_dict, edge_feature_dict)
        for node_type in x_dict.keys():
            x_dict[node_type] = F.dropout(F.relu(x_dict[node_type]), p=0.5, training=self.training)

        x_dict = self.conv_2(x_dict, edge_index_dict, edge_feature_dict)
        for node_type in x_dict.keys():
            x_dict[node_type] = F.dropout(F.relu(x_dict[node_type]), p=0.5, training=self.training)

        out_dict = {}
        for node_type in x_dict.keys():
            out_dict[node_type] = self.lin_out[node_type](x_dict[node_type])

        return out_dict
