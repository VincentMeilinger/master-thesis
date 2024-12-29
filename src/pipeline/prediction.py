import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn.modules.loss import TripletMarginLoss
import torch.nn as nn

from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, EdgeType
from src.shared.graph_sampling import GraphSampling
from src.model.classifiers.siamese_classifier import TupleNet, EmbeddingNet
from src.model.GAT.gat_encoder import HomoGATv2Encoder

from src.model.util.util_classifier import *
from src.model.training.training_classifier import *
from src.model.util.util import plot_losses, save_training_results, save_dict_to_json
from src.shared.config import get_logger

random.seed(40)
np.random.seed(40)
torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

logger = get_logger("Prediction")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Graph sampling configurations
node_properties = [
    'id',
    'feature_vec',
]


def load_model(config):
    logger.info("Loading GAT Encoders and Embedding Net ...")
    edge_spec = config["edge_spec"]
    node_spec = config["node_spec"]
    gat_list = config["gat_list"]
    gat_specs = config["gat_specs"]

    gat_encoders = {}
    for edge_key, gat_path in gat_list.items():
        gat_encoder = HomoGATv2Encoder(gat_specs['hidden_channels'], gat_specs['out_channels'],
                                       num_heads=gat_specs['num_heads']).to(device)
        gat_encoder.load_state_dict(torch.load(gat_path))
        gat_encoders[edge_key] = gat_encoder

    # Create models
    metadata = (
        [n.value for n in node_spec],
        [edge_pyg_key_vals[r] for r in edge_spec]
    )

    # Embedding model
    embedding_net = EmbeddingNet(
        input_size=gat_specs['classifier_in_channels'],
        hidden_size=gat_specs['classifier_hidden_channels'],
        embedding_size=gat_specs['classifier_out_channels'],
        dropout=gat_specs['classifier_dropout']
    ).to(device)
    embedding_net.load_state_dict(torch.load(config["embedding_net_path"]))

    # Tuple classifier model
    tuple_net = TupleNet(
        embedding_net=embedding_net,
        edge_spec=edge_spec,
        gat_encoders=gat_encoders
    ).to(device)

    tuple_net.eval()
    logger.info("Models loaded successfully!")
    return tuple_net


def load_graphs(gs: GraphSampling, edge_spec: [EdgeType], node_id, config):
    data_dict = {}
    for edge_type in edge_spec:
        g = gs.expand_config_homogeneous(NodeType.PUBLICATION, node_id, max_level=config["gat_specs"]['max_hops'])

        # Convert to PyG Data objects
        data, node_map_a = neo_to_pyg_homogeneous(g, config["gat_specs"]['model_node_feature'])
        data.central_node_id = torch.tensor([node_map_a[node_id]])
        data.publication_id = node_id
        data.to(device)
        data_dict[edge_type] = data

    return data_dict

def predict(
        db: DatabaseWrapper,
        config: dict,
):
    logger.info("Starting Author Disambiguation Pipeline ...")
    edge_spec = config["edge_spec"]
    node_spec = config["node_spec"]

    gs = GraphSampling(
        node_spec=node_spec,
        edge_spec=edge_spec,
        node_properties=node_properties,
        database=config["database"]
    )

    # Load model
    tuple_net = load_model(config)

    # Predict links between publications
    for nodes1 in db.iter_nodes(NodeType.PUBLICATION, attr_keys=["id"]):
        for node1 in nodes1:
            logger.debug(f"Comparing current node: {node1['id']} ...")
            for nodes2 in db.iter_nodes(NodeType.PUBLICATION, attr_keys=["id"]):
                for node2 in nodes2:
                    logger.debug(f"    Against node: {node2['id']}")
                    graphs1 = load_graphs(
                        gs=gs,
                        edge_spec=edge_spec,
                        node_id=node1["id"],
                        config=config
                    )

                    graphs2 = load_graphs(
                        gs=gs,
                        edge_spec=edge_spec,
                        node_id=node2["id"],
                        config=config
                    )

                    try:
                        with torch.no_grad():
                            emb1, emb2 = tuple_net(graphs1, graphs2)
                            dist = F.pairwise_distance(emb1, emb2).item()
                            logger.debug(f"    Distance: {dist}")
                            if dist < config["margin"]:
                                logger.debug("    >>> Predicted a Link. Merging edge <<<")
                                db.merge_edge(NodeType.PUBLICATION, node1["id"], NodeType.PUBLICATION, node2["id"], EdgeType.SAME_AUTHOR, properties={"sim": dist})
                    except Exception as e:
                        print("Error: ", e)

