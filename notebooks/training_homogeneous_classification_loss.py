import torch
import torch.nn.functional as F

from notebooks.util import contrastive_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_neg_contrastive_loss(emb1, emb2, margin=1.0):
    labels = torch.zeros(emb1.size(0)).to(device)
    # Compute Euclidean distances between embeddings
    distances = torch.norm(emb1 - emb2, p=2, dim=-1)
    # Compute contrastive loss
    losses = labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2)
    # Return the mean loss over the batch
    return losses.mean()

def train_dual_objective(model, loss_fn, batch_anchor, batch_pos, batch_neg, optimizer):
    model.train()

    optimizer.zero_grad()

    batch_anchor = batch_anchor.to(device)
    batch_pos = batch_pos.to(device)
    batch_neg = batch_neg.to(device)

    emb_a = model(batch_anchor)
    emb_p = model(batch_pos)
    emb_n = model(batch_neg)

    emb_a_central = emb_a[batch_anchor.central_node_id]
    emb_p_central = emb_p[batch_pos.central_node_id]
    emb_n_central = emb_n[batch_neg.central_node_id]

    triplet_loss = loss_fn(emb_a_central, emb_p_central, emb_n_central)

    # Compute second loss
    contrastive_loss = compute_neg_contrastive_loss(emb_a_central, emb_n_central)

    # Combine losses
    alpha = 1.0
    beta = 1.0

    loss = alpha * triplet_loss + beta * contrastive_loss

    loss.backward()
    optimizer.step()

    batch_loss = loss.item()
    # print(f"Batch loss: {batch_loss:.4f}")
    return batch_loss


def test_and_eval(model, loss_fn, dataloader, margin):
    model.eval()
    total_loss = 0
    total_triplet_loss = 0
    total_cross_entropy_loss = 0
    total_num_correct = 0
    total_pos_correct = 0
    total_neg_correct = 0
    total_num_samples = 0

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

            triplet_loss = loss_fn(emb_a_central, emb_p_central, emb_n_central)

            # Compute second loss
            contrastive_loss = compute_neg_contrastive_loss(emb_a_central, emb_n_central)

            # Combine losses
            alpha = 1.0
            beta = 1.0

            loss = alpha * triplet_loss + beta * contrastive_loss

            total_triplet_loss += triplet_loss.item()
            total_cross_entropy_loss += contrastive_loss.item()
            total_loss += loss.item()

            # Compute distances
            d_ap = F.pairwise_distance(emb_a_central, emb_p_central)
            d_an = F.pairwise_distance(emb_a_central, emb_n_central)

            # Determine correct predictions based on margin
            correct_pos = (d_ap < margin).cpu()
            correct_neg = (d_an > margin).cpu()

            # Sum up correct predictions
            num_correct_pos = correct_pos.sum().item()
            num_correct_neg = correct_neg.sum().item()
            num_correct = num_correct_pos + num_correct_neg

            total_num_correct += num_correct
            total_pos_correct += num_correct_pos
            total_neg_correct += num_correct_neg
            total_num_samples += len(batch_anchor)

    # Compute averages
    avg_triplet_loss = total_triplet_loss / len(dataloader)
    avg_cross_entropy_loss = total_cross_entropy_loss / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    avg_correct_pos = total_pos_correct / total_num_samples
    avg_correct_neg = total_neg_correct / total_num_samples
    avg_num_correct = total_num_correct / (2 * total_num_samples)  # Since we have two conditions



    # Compute average loss
    print(f"        Correct positive: {total_pos_correct} ({avg_correct_pos * 100:.2f}%), Correct negative: {total_neg_correct} ({avg_correct_neg * 100:.2f}%)")
    print(f"        Total correct: {total_num_correct} ({avg_num_correct * 100:.2f}%)")
    print(f"        Test/Eval Loss: {avg_loss:.4f}, Test/Eval Accuracy: {avg_num_correct:.4f}")
    print(f"        Triplet Loss: {avg_triplet_loss:.4f}, Cross Entropy Loss: {avg_cross_entropy_loss:.4f}")
    return avg_loss, avg_triplet_loss, avg_cross_entropy_loss, avg_num_correct, avg_correct_pos, avg_correct_neg