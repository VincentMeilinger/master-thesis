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



def neo_to_pyg_hetero_edges(
        data,
        node_attr: str,
):
    if not data:
        return None, None

    nodes = data["nodes"]
    relationships = data["relationships"]

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
    if node_features:
        pyg_data[NodeType.PUBLICATION.value].x = torch.vstack(node_features)
        pyg_data[NodeType.PUBLICATION.value].num_nodes = pyg_data[NodeType.PUBLICATION.value].x.size(0)
    else:
        print("No node features available.")
        return None, None

    # Process relationships
    edge_dict = {}

    for rel in relationships:
        key = edge_val_to_pyg_key_vals[rel.type]
        if key not in edge_dict:
            edge_dict[key] = [[], []]

        source_id = rel.start_node.get("id")
        target_id = rel.end_node.get("id")

        # Append the indices of the source and target nodes
        edge_dict[key][0].append(node_id_map[source_id])
        edge_dict[key][1].append(node_id_map[target_id])

    # Convert edge lists to tensors
    for key in edge_dict:
        pyg_data[key[0], key[1], key[2]].edge_index = torch.vstack([
            torch.tensor(edge_dict[key][0], dtype=torch.long),
            torch.tensor(edge_dict[key][1], dtype=torch.long)
        ])

        pyg_data[key[0], key[1], key[2]].edge_attr = torch.vstack(
            [edge_one_hot[key[1]] for _ in range(len(edge_dict[key][0]))])

    return pyg_data, node_id_map


def graph_data_valid(data: Data, edge_spec: list):
    try:
        node_type_val = NodeType.PUBLICATION.value
        assert data is not None, "Data object is None."
        assert data.num_nodes > 0, "Number of nodes must be greater than 0."
        assert data.num_edges > 0, "Number of edges must be greater than 0."
        assert data[node_type_val].x is not None, "Node features 'x' are missing."
        assert data[node_type_val].x.size(0) == data.num_nodes, "Mismatch between 'x' size and 'num_nodes'."
        assert data[node_type_val].x.dtype in (torch.float32, torch.float64), "Node features 'x' must be floating point."
        for key in [edge_pyg_key_vals[r] for r in edge_spec]:
            if key not in data:
                continue
            assert data[key].edge_index.size(0) == 2, f"'edge_index' for '{key}' should have shape [2, num_edges]."
            assert data[key].edge_index.size(1) == data[key].num_edges, f"Mismatch between 'edge_index' and 'num_edges' for '{key}'."
            assert data[key].edge_index is not None, f"Edge index for '{key}' is missing."
            assert data[key].edge_index.max() < data.num_nodes, f"'edge_index' for '{key}' contains invalid node indices."
        return True
    except AssertionError as e:
        #print(f"Data check failed: {e}")
        return False


# custom collate function adjusted for GraphPairDataset
def custom_pair_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Skip empty batches

    data1_list = [item[0] for item in batch]
    data2_list = [item[1] for item in batch]

    labels = torch.stack([item[2] for item in batch])

    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)

    return batch1, batch2, labels


def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    # Compute Euclidean distances between embeddings
    distances = F.pairwise_distance(embeddings1, embeddings2)

    # Loss
    loss_pos = labels * distances.pow(2)  # For positive pairs
    loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)  # For negative pairs
    loss = loss_pos + loss_neg
    return loss.mean(), distances


# Plot loss
def plot_loss(losses, epoch_marker_pos=None, plot_title="Loss", window_size=20, plot_avg = False, plot_file=None):
    if plot_file is None:
        plot_file = f'./data/losses/loss.png'
    if epoch_marker_pos is None:
        epoch_marker_pos = []

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=f'Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Compute the moving average
    if plot_avg and window_size > 1 and len(losses) >= window_size:
        # Pad the losses array to align the moving average with the original data
        pad_left = window_size // 2
        pad_right = window_size // 2 - 1 if window_size % 2 == 0 else window_size // 2
        losses_padded = np.pad(losses, (pad_left, pad_right), mode='edge')
        moving_avg = np.convolve(losses_padded, np.ones(window_size) / window_size, mode='valid')
        moving_avg_x = np.arange(0, len(losses))
        plt.plot(moving_avg_x, moving_avg, label=f'Moving Average (window={window_size})', color='orange')

    for ix, x_pos in enumerate(epoch_marker_pos):
        plt.axvline(x=x_pos, color='red', linestyle='dotted', linewidth=1)
        plt.text(
            x_pos,
            max(losses),
            f'Epoch {ix}',
            rotation=90,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            color='red'
        )

    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()


def save_training_results(train_loss, test_loss, eval_results, config, file_path):
    results = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'eval_results': eval_results,
        'config': config,
    }
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)


class GraphPairDataset(Dataset):
    def __init__(self, pairs, gs, config):
        self.pairs = pairs  # List of tuples: (paper_id1, paper_id2, label)
        self.gs = gs
        self.config = config

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        paper_id1, paper_id2, label = self.pairs[idx]
        try:
            # print(f"Processing pair ({paper_id1}, {paper_id2})")
            # Get n-hop neighbourhood for each paper
            graph1 = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, paper_id1, max_level=self.config['max_hops'])
            graph2 = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, paper_id2, max_level=self.config['max_hops'])

            # Convert to PyG Data objects
            data1, node_map_1 = neo_to_pyg_hetero_edges(graph1, self.config['model_node_feature'])
            data1.central_node_id = torch.tensor([node_map_1[paper_id1]])

            data2, node_map_2 = neo_to_pyg_hetero_edges(graph2, self.config['model_node_feature'])
            data2.central_node_id = torch.tensor([node_map_2[paper_id2]])

            # Return data and label
            return data1, data2, torch.tensor(label, dtype=torch.float)
        except Exception as e:
            print(f"Error processing pair ({paper_id1}, {paper_id2}): {e}")
            return None


# This is required for the PyG DataLoader in order to handle the custom mini-batching during training
class CentralGraphData(HeteroData):
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


class PairDataHarvester:
    def __init__(self, db: DatabaseWrapper, gs: GraphSampling, edge_spec: list, config: dict, save_file_postfix: str):
        self.db = db
        self.gs = gs
        self.pairs = []
        self.edge_spec = edge_spec
        self.config = config
        self.save_file_postfix = save_file_postfix
        self.prepare_pairs()

    def prepare_pairs(self):
        print("Preparing pairs...")
        file_path = f'./data/valid_pairs_{self.save_file_postfix}.json'

        try:
            print("Loading pairs...")
            self.load_pairs(file_path)
            print(f"Loaded {len(self.pairs)} pairs.")
        except FileNotFoundError:
            print("Could not load pairs from file. Generating pairs...")
            self.generate_pairs()
            print(f"Generated {len(self.pairs)} pairs.")
            print("Saving pairs...")
            self.save_pairs(file_path)
            print("Pairs saved.")

    def load_pairs(self, file_path):
        with open(file_path, 'r') as f:
            self.pairs = json.load(f)

    def save_pairs(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.pairs, f)

    def generate_pairs(self):
        # Filter out the papers that are not present in the graph or have less than 2 edges
        paper_ids = []
        print("Checking data validity...")
        total_num_papers = 0
        invalid_papers = 0
        for nodes in self.db.iter_nodes_with_edge_count(NodeType.PUBLICATION, self.edge_spec, ['id', 'true_author']):
            for node in nodes:
                total_num_papers += 1
                data = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, node['id'], max_level=1)
                data = neo_to_pyg_hetero_edges(data, self.config['model_node_feature'])[0]
                if not graph_data_valid(data, edge_spec=self.edge_spec):
                    invalid_papers += 1
                    continue
                paper_ids.append(node['id'])

        print(f"Out of {total_num_papers} checked papers, {len(paper_ids)} are valid and {invalid_papers} are invalid.")
        print("Preparing pairs...")
        paper_set = set(paper_ids)

        author_data = WhoIsWhoDataset.parse_train()
        for author_id, data in author_data.items():
            for key in data:
                data[key] = [p_id for p_id in data[key] if p_id in paper_set]

        # Generate pairs with labels
        pairs = []

        for author_id, data in author_data.items():
            normal_data = data.get('normal_data', [])
            outliers = data.get('outliers', [])

            # Positive pairs: combinations of normal_data
            pos_pairs = list(combinations(normal_data, 2))
            if len(pos_pairs) > 50:
                pos_pairs = random.sample(pos_pairs, 50)
            for pair in pos_pairs:
                pairs.append((pair[0], pair[1], 1))

            # Negative pairs: product of normal_data and outliers
            neg_pairs = list(product(normal_data, outliers))
            if len(neg_pairs) > 50:
                neg_pairs = random.sample(neg_pairs, 50)
            elif len(neg_pairs) < len(pos_pairs):
                # Sample random paper ids from other authors
                while len(neg_pairs) < len(pos_pairs):
                    p1 = random.choice(normal_data)
                    p2 = random.choice(paper_ids)
                    if p2 not in normal_data:
                        neg_pairs.append((p1, p2))
            for pair in neg_pairs:
                pairs.append((pair[0], pair[1], 0))

        print(f"Total pairs: {len(pairs)}. Done.")
        self.pairs = pairs


class GraphTripletDataset(Dataset):
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
            g_a = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, anchor, max_level=self.config['max_hops'])
            g_p = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, pos, max_level=self.config['max_hops'])
            g_n = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, neg, max_level=self.config['max_hops'])

            # Convert to PyG Data objects
            data_a, node_map_a = neo_to_pyg_hetero_edges(g_a, self.config['model_node_feature'])
            data_a.central_node_id = torch.tensor([node_map_a[anchor]])

            data_p, node_map_p = neo_to_pyg_hetero_edges(g_p, self.config['model_node_feature'])
            data_p.central_node_id = torch.tensor([node_map_p[pos]])

            data_n, node_map_n = neo_to_pyg_hetero_edges(g_n, self.config['model_node_feature'])
            data_n.central_node_id = torch.tensor([node_map_n[neg]])

            # Return data and label
            return data_a, data_p, data_n
        except Exception as e:
            print(f"Error processing triplet ({anchor}, {pos}, {neg}): {e}")
            return None

# custom collate function adjusted for GraphTripletDataset
def custom_triplet_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Skip empty batches

    data1_list = [item[0] for item in batch]
    data2_list = [item[1] for item in batch]
    data3_list = [item[2] for item in batch]

    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    batch3 = Batch.from_data_list(data3_list)

    return batch1, batch2, batch3


class TripletDataHarvester:
    def __init__(self, db: DatabaseWrapper, gs: GraphSampling, edge_spec: list, config: dict, save_file_postfix: str):
        self.db = db
        self.gs = gs
        self.triplets = []
        self.edge_spec = edge_spec
        self.config = config
        self.save_file_postfix = save_file_postfix
        self.prepare_triplets()

    def prepare_triplets(self):
        print("Preparing triplets...")
        file_path = f'./data/valid_triplets_{self.save_file_postfix}.json'

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
                data = self.gs.n_hop_neighbourhood(NodeType.PUBLICATION, node['id'], max_level=1)
                data = neo_to_pyg_hetero_edges(data, self.config['model_node_feature'])[0]
                if not graph_data_valid(data, edge_spec=self.edge_spec):
                    invalid_papers += 1
                    continue
                paper_ids.append(node['id'])

        print(f"Out of {total_num_papers} checked papers, {len(paper_ids)} are valid and {invalid_papers} are invalid.")
        print("Preparing pairs...")
        paper_set = set(paper_ids)

        author_data = WhoIsWhoDataset.parse_train()
        paper_data = WhoIsWhoDataset.parse_data()

        for author_id, data in author_data.items():
            for key in data:
                data[key] = [p_id for p_id in data[key] if p_id in paper_set]

        # Load sentence transformer model for embedding paper titles
        model = SentenceTransformer(
            'data/models/all-MiniLM-L6-v2-32dim',
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

        print(f"Total triplets: {len(triplets)}. Done.")
        self.triplets = triplets