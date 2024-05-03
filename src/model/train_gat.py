from .graph_attention import GAT
import torch.optim as optim
import torch


def train(x_train, x_test, params):
    # TODO: Define Loss function for semi-supervised learning
    model = GAT(
        node_feature_dim=0,
        edge_feature_dim=0,
        embedding_dim=0
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(params['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(x_train.x, x_train.edge_index, x_train.edge_attr)
        loss = loss(out, x_train)  # TODO: Define loss function
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")


