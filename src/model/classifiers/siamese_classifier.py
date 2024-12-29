import random
import numpy as np

import torch.nn as nn
from src.shared.graph_schema import *

random.seed(40)
np.random.seed(40)
torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

class EmbeddingNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, embedding_size=16, dropout=0.2):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_net: EmbeddingNet, edge_spec: [EdgeType], gat_encoders: dict[EdgeType, nn.Module]):
        super(TripletNet, self).__init__()
        self.edge_spec = edge_spec
        self.gat_encoders = gat_encoders
        self.embedding_net = embedding_net

        for gat in self.gat_encoders.values():
            gat.eval()
            for param in gat.parameters():
                param.requires_grad = False

    def forward(self, data_dict: dict):
        anchor = []
        positive = []
        negative = []

        for edge_type in self.edge_spec:
            # Anchor node embedding for the edge type
            anchor_graph = data_dict[edge_type][0]
            anchor_gat_emb = self.gat_encoders[edge_type](anchor_graph)
            anchor.append(anchor_gat_emb[anchor_graph.central_node_id])

            # Positive node embedding for the edge type
            positive_graph = data_dict[edge_type][1]
            positive_gat_emb = self.gat_encoders[edge_type](positive_graph)
            positive.append(positive_gat_emb[positive_graph.central_node_id])

            # Negative node embedding for the edge type
            negative_graph = data_dict[edge_type][2]
            negative_gat_emb = self.gat_encoders[edge_type](negative_graph)
            negative.append(negative_gat_emb[negative_graph.central_node_id])

        anchor = torch.cat(anchor, dim=1)
        positive = torch.cat(positive, dim=1)
        negative = torch.cat(negative, dim=1)

        output_anchor = self.embedding_net(anchor)
        output_positive = self.embedding_net(positive)
        output_negative = self.embedding_net(negative)

        return output_anchor, output_positive, output_negative


class TupleNet(nn.Module):
    def __init__(self, embedding_net: EmbeddingNet, edge_spec: [EdgeType], gat_encoders: dict[EdgeType, nn.Module]):
        super(TupleNet, self).__init__()
        self.edge_spec = edge_spec
        self.gat_encoders = gat_encoders
        self.embedding_net = embedding_net

        for gat in self.gat_encoders.values():
            gat.eval()
            for param in gat.parameters():
                param.requires_grad = False

    def forward(self, data_dict_1, data_dict_2):
        data_1 = []
        data_2 = []

        for edge_type in self.edge_spec:
            # Data1 node embedding for the edge type
            data_1_graph = data_dict_1[edge_type]
            data_1_gat_emb = self.gat_encoders[edge_type](data_1_graph)
            data_1.append(data_1_gat_emb[data_1_graph.central_node_id])

            # Data2 node embedding for the edge type
            data_2_graph = data_dict_2[edge_type]
            data_2_gat_emb = self.gat_encoders[edge_type](data_2_graph)
            data_2.append(data_2_gat_emb[data_2_graph.central_node_id])

        data_1 = torch.cat(data_1, dim=1)
        data_2 = torch.cat(data_2, dim=1)

        output_data_1 = self.embedding_net(data_1)
        output_data_2 = self.embedding_net(data_2)

        return output_data_1, output_data_2