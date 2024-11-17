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


def save_training_results(train_loss, test_loss, eval_results, config, file_path):
    results = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'eval_results': eval_results,
        'config': config,
    }
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)