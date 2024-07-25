import torch
import pandas as pd
from graphdatascience import GraphDataScience

from src.shared import config
from src.shared.run_state import RunState
from src.shared.run_config import RunConfig
from src.model.GAT.gat_encoder import GATEncoder
from src.shared.database_wrapper import DatabaseWrapper

logger = config.get_logger("TrainGAT")

def prepare_data(run_config: RunConfig):


def train_gat():
    run_state = RunState(config.RUN_ID, config.RUN_DIR)
    run_config = RunConfig(config.RUN_DIR)
    data = preprocess_data(nodes, edges)
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(data.device)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
