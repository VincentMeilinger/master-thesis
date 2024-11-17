import os
import json
import random
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import combinations, product
from sentence_transformers import SentenceTransformer

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from src.shared.graph_schema import *
from src.datasets.who_is_who import WhoIsWhoDataset
from src.shared.graph_sampling import GraphSampling
from src.shared.database_wrapper import DatabaseWrapper

def homogeneous_graph_data_valid(data: Data, edge_spec: list):
    try:
        assert data is not None, "Data object is None."
        assert data.num_nodes > 0, "Number of nodes must be greater than 0."
        assert data.num_edges > 0, "Number of edges must be greater than 0."
        assert data.x is not None, "Node features 'x' are missing."

        return True
    except AssertionError as e:
        #print(f"Data check failed: {e}")
        return False


class CentralGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'central_node_id':
            return 0  # Concat along batch dim
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'central_node_id':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)


def neo_to_pyg_homogeneous(
        data,
        node_attr: str,
):
    if not data:
        return None, None

    nodes = data["nodes"]
    relationships = data["relationships"]

    #print(f"Nodes: {len(nodes)}, Relationships: {len(relationships)}")

    # Create a PyG Data object
    pyg_data = CentralGraphData()

    node_features = []
    node_ids = []
    node_id_map = {}

    for node in nodes:
        node_id = node.get("id")
        node_feature = node.get(node_attr, None)
        if node_feature is None:
            print(f"Node {node_id} has no attribute {node_attr}")
            continue

        # Map node id to its index in the list
        idx = len(node_ids)
        node_id_map[node_id] = idx
        node_ids.append(node_id)

        # Convert node features to tensors
        node_feature_tensor = torch.tensor(node_feature, dtype=torch.float32)
        node_features.append(node_feature_tensor)

    # Convert list of features to tensor
    pyg_data.x = torch.vstack(node_features)
    pyg_data.num_nodes = pyg_data.x.size(0)

    # Process relationships
    edge_index_list = [[], []]

    for rel in relationships:
        source_id = rel.start_node.get("id")
        target_id = rel.end_node.get("id")

        if source_id not in node_id_map or target_id not in node_id_map:
            print(f"Edge from {source_id} to {target_id} cannot be mapped to node indices.")
            continue

        source_idx = node_id_map[source_id]
        target_idx = node_id_map[target_id]

        edge_index_list[0].append(source_idx)
        edge_index_list[1].append(target_idx)

    # Convert edge lists to tensor
    pyg_data.edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    return pyg_data, node_id_map


class HomogeneousGraphTripletDataset(Dataset):
    def __init__(self, triplets, gs, config):
        self.triplets = triplets  # List of triplets: (anchor, pos, neg)
        self.gs = gs
        self.config = config

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]
        try:
            # Get n-hop neighbourhood for each paper
            g_a = self.gs.expand_config_homogeneous(NodeType.PUBLICATION, anchor, max_level=self.config['max_hops'])
            g_p = self.gs.expand_config_homogeneous(NodeType.PUBLICATION, pos, max_level=self.config['max_hops'])
            g_n = self.gs.expand_config_homogeneous(NodeType.PUBLICATION, neg, max_level=self.config['max_hops'])

            # Convert to PyG Data objects
            data_a, node_map_a = neo_to_pyg_homogeneous(g_a, self.config['model_node_feature'])
            data_a.central_node_id = torch.tensor([node_map_a[anchor]])

            data_p, node_map_p = neo_to_pyg_homogeneous(g_p, self.config['model_node_feature'])
            data_p.central_node_id = torch.tensor([node_map_p[pos]])

            data_n, node_map_n = neo_to_pyg_homogeneous(g_n, self.config['model_node_feature'])
            data_n.central_node_id = torch.tensor([node_map_n[neg]])

            # Return data and label
            return data_a, data_p, data_n
        except Exception as e:
            print(f"Error processing triplet ({anchor}, {pos}, {neg}): {e}")
            return None


class TripletDataHarvester:
    def __init__(self, db: DatabaseWrapper, gs: GraphSampling, edge_spec: list, config: dict, valid_triplets_save_file: str = "valid_triplets", transformer_model='data/models/all-MiniLM-L6-v2-32dim'):
        self.db = db
        self.gs = gs
        self.triplets = []
        self.edge_spec = edge_spec
        self.config = config
        self.valid_triplets_save_file = valid_triplets_save_file
        self.transformer_model = transformer_model

        self.prepare_triplets()

    def prepare_triplets(self):
        print("Preparing triplets...")
        file_path = f'./data/valid_triplets/{self.valid_triplets_save_file}.json'

        try:
            print("Loading triplets...")
            self.load_triplets(file_path)
            print(f"Loaded {len(self.triplets)} triplets.")
        except FileNotFoundError:
            print("Could not load triplets from file. Generating triplets...")
            self.generate_triplets()
            print(f"Generated {len(self.triplets)} triplets.")
            print("Saving triplets...")
            self.save_triplets(file_path)
            print("Triplets saved.")

    def load_triplets(self, file_path):
        with open(file_path, 'r') as f:
            self.triplets = json.load(f)

    def save_triplets(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.triplets, f)

    def generate_triplets(self):
        # Filter out the papers that are not present in the graph or have less than 2 edges
        paper_ids = []
        print("Checking data validity...")
        total_num_papers = 0
        invalid_papers = 0
        for nodes in self.db.iter_nodes_with_edge_count(NodeType.PUBLICATION, self.edge_spec, ['id', 'true_author']):
            for node in nodes:
                total_num_papers += 1
                data = self.gs.expand_config_homogeneous(NodeType.PUBLICATION, node['id'], max_level=1)
                data = neo_to_pyg_homogeneous(data, self.config['model_node_feature'])[0]
                if not homogeneous_graph_data_valid(data, edge_spec=self.edge_spec):
                    invalid_papers += 1
                    continue
                paper_ids.append(node['id'])

        print(f"Out of {total_num_papers} checked papers, {len(paper_ids)} are valid and {invalid_papers} are invalid.")
        print("Generating hard triplets ...")
        paper_set = set(paper_ids)

        author_data = WhoIsWhoDataset.parse_train()
        paper_data = WhoIsWhoDataset.parse_data()

        for author_id, data in author_data.items():
            for key in data:
                data[key] = [p_id for p_id in data[key] if p_id in paper_set]

        # Load sentence transformer model for embedding paper titles
        model = SentenceTransformer(
            self.transformer_model,
            device='cuda'
        )

        # Generate triplets
        triplets = []

        for author_id, data in author_data.items():
            normal_data = data.get('normal_data', [])
            outliers = data.get('outliers', [])

            if len(normal_data) < 2 or len(outliers) < 1:
                continue

            normal_titles = [paper_data[p_id]['title'] for p_id in normal_data]
            outlier_titles = [paper_data[p_id]['title'] for p_id in outliers]

            # Embed paper titles to find hard triplets
            normal_embeddings = model.encode(normal_titles)
            outlier_embeddings = model.encode(outlier_titles)

            # Compute pairwise distances for normal embeddings
            normal_distances = cdist(normal_embeddings, normal_embeddings, 'euclidean')
            np.fill_diagonal(normal_distances, -np.inf)  # Exclude self-distance

            # Find the furthest positive sample for each anchor
            hardest_positive_indices = np.argmax(normal_distances, axis=1)

            # Compute cosine dist between normal and outlier embeddings
            negative_distances = cdist(normal_embeddings, outlier_embeddings, 'euclidean')

            # Find the closest neg for each anchor
            hardest_negative_indices = np.argmin(negative_distances, axis=1)

            # Form triplets
            for i, anchor_paper_id in enumerate(normal_data):
                hardest_positive_paper_id = normal_data[hardest_positive_indices[i]]
                hardest_negative_paper_id = outliers[hardest_negative_indices[i]]

                triplet = (anchor_paper_id, hardest_positive_paper_id, hardest_negative_paper_id)
                triplets.append(triplet)

        if len(triplets) < 1000:
            print(f"Too few triplets generated: {len(triplets)}. Generating more triplets...")
            new_triplets = []
            for author_id, data in author_data.items():
                normal_data = data.get('normal_data', [])
                outliers = data.get('outliers', [])

                if len(normal_data) < 2 or len(outliers) < 1:
                    continue

                # Generate random triplets
                for anchor_paper_id in normal_data:
                    pos_paper_id = random.choice(normal_data)
                    neg_paper_id = random.choice(outliers)

                    triplet = (anchor_paper_id, pos_paper_id, neg_paper_id)
                    new_triplets.append(triplet)

            triplets.extend(new_triplets)

        print(f"Total triplets generated: {len(triplets)}. Done.")
        self.triplets = triplets