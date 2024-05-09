from .gat_encoder import GATEncoder
from src.shared import config
import torch

logger = config.get_logger("Train")


def train_supervised(x_train, y_train, x_test, y_test, params: dict):
    # TODO: Define Loss function for supervised learning

    # Create model
    logger.debug("Creating model")
    model_params = params['model']
    model = GATEncoder(
        node_feature_dim=0,
        edge_feature_dim=0,
        embedding_dim=model_params['embedding_dim'],
    )

    # Create optimizer
    logger.debug("Creating optimizer")
    optimizer_params = params['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_params['lr'],
        weight_decay=optimizer_params['weight_decay'],
    )

    # Train model
    train_params = params['train']
    logger.info(f"Supervised model training for {train_params['epochs']} epochs ...")
    for epoch in range(train_params['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(x_train.x, x_train.edge_index, x_train.edge_attr)
        loss = loss(out, x_train)  # TODO: Define loss function
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss {loss.item()}")

    # Save model weights
    logger.info("Training complete. Saving model weights ...")
    torch.save(model.state_dict(), 'model_weights.pth')
