import os
import json
import random
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import combinations, product
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData


from src.shared.graph_schema import *
from src.datasets.who_is_who import WhoIsWhoDataset
from src.shared.graph_sampling import GraphSampling
from src.shared.database_wrapper import DatabaseWrapper
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm


def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    # Compute Euclidean distances between embeddings
    distances = F.pairwise_distance(embeddings1, embeddings2)

    # Loss
    loss_pos = labels * distances.pow(2)  # For positive pairs
    loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)  # For negative pairs
    loss = loss_pos + loss_neg
    return loss.mean(), distances


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


# Plot loss
def plot_loss(
        losses,
        epoch_len: int = None,
        plot_title="Loss",
        window_size=20,
        plot_avg = False,
        x_label: str = 'Iteration',
        y_label: str = 'Loss',
        line_label='Loss',
        plot_file=None
):
    if plot_file is None:
        plot_file = f'./data/losses/loss.png'

    plt.figure(figsize=(10, 6))

    plt.plot(losses, label=line_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Compute the moving average
    if plot_avg and window_size > 1 and len(losses) >= window_size:
        # Pad the losses array to align the moving average with the original data
        pad_left = window_size // 2
        pad_right = window_size // 2 - 1 if window_size % 2 == 0 else window_size // 2
        losses_padded = np.pad(losses, (pad_left, pad_right), mode='edge')
        moving_avg = np.convolve(losses_padded, np.ones(window_size) / window_size, mode='valid')
        moving_avg_x = np.arange(0, len(losses))
        plt.plot(moving_avg_x, moving_avg, label=f'Moving Average (window size = {window_size})', color='orange')

    for ix, x_pos in enumerate(range(0, len(losses), epoch_len)):
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

def plot_losses(
        losses,
        line_labels,
        epoch_len: int = None,
        plot_title="Loss",
        x_label: str = 'Iteration',
        y_label: str = 'Loss',
        plot_file=None
):
    if plot_file is None:
        plot_file = f'./data/losses/loss.png'

    plt.figure(figsize=(10, 6))

    line_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, loss in enumerate(losses):
        plt.plot(loss, label=line_labels[i], color=line_colors[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for ix, x_pos in enumerate(range(0, len(losses[0]), epoch_len)):
        plt.axvline(x=x_pos, color='red', linestyle='dotted', linewidth=1)
        plt.text(
            x_pos,
            max(losses[0]),
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

def save_dict_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def reverse_dict(author_dict):
    paper_to_author = {}
    for author_id, values in author_dict.items():
        normal_papers = values.get('normal_data', [])
        for paper_id in normal_papers:
            paper_to_author[paper_id] = author_id
    return paper_to_author

def plot_feature_space(model, dataloader, save_file: str):
    model.eval()
    embedding_dict = {}
    author_dict = WhoIsWhoDataset.parse_train()
    paper_to_author_dict = reverse_dict(author_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch_anchor, batch_pos, batch_neg in dataloader:
            batch_anchor = batch_anchor.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            emb_a = model(batch_anchor)
            emb_p = model(batch_pos)
            emb_n = model(batch_neg)

            emb_a_central = emb_a[batch_anchor.central_node_id]
            emb_p_central = emb_p[batch_pos.central_node_id]
            emb_n_central = emb_n[batch_neg.central_node_id]

            for id, emb in zip(batch_anchor.publication_id, emb_a_central):
                embedding_dict[id] = emb.cpu().numpy()
            for id, emb in zip(batch_pos.publication_id, emb_p_central):
                embedding_dict[id] = emb.cpu().numpy()
            for id, emb in zip(batch_neg.publication_id, emb_n_central):
                embedding_dict[id] = emb.cpu().numpy()

    embeddings = []
    true_author_ids = []
    for pub_id, emb in embedding_dict.items():
        if pub_id not in paper_to_author_dict:
            continue
        embeddings.append(emb)
        true_author_ids.append(paper_to_author_dict[pub_id])

    embeddings = np.array(embeddings)
    true_author_ids = np.array(true_author_ids)

    embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

    # Encode author id to integers
    label_encoder = LabelEncoder()
    author_labels = label_encoder.fit_transform(true_author_ids)
    num_authors = len(label_encoder.classes_)
    cmap = cm.get_cmap('nipy_spectral', num_authors)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=author_labels,
        cmap=cmap,
        alpha=0.7,
        edgecolors = 'grey',
        linewidths = 0.5,
    )
    cbar = plt.colorbar(scatter, ticks=range(len(label_encoder.classes_)))
    cbar.ax.set_yticklabels(label_encoder.classes_)
    plt.title('Publication Embeddings in 2D Space')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    plt.savefig(save_file)
    plt.close()