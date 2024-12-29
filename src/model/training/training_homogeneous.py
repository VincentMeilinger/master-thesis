import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, loss_fn, batch_anchor, batch_pos, batch_neg, optimizer, margin = 1.0):
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

    loss = loss_fn(emb_a_central, emb_p_central, emb_n_central)

    loss.backward()
    optimizer.step()

    batch_loss = loss.item()

    return batch_loss


def test(model, loss_fn, dataloader, margin):
    model.eval()
    total_loss = 0
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

            loss = loss_fn(emb_a_central, emb_p_central, emb_n_central)
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
    avg_loss = total_loss / len(dataloader)
    avg_correct_pos = total_pos_correct / total_num_samples
    avg_correct_neg = total_neg_correct / total_num_samples
    avg_num_correct = total_num_correct / (2 * total_num_samples)  # Since we have two conditions



    # Compute average loss
    print(f"        Correct positive: {total_pos_correct} ({avg_correct_pos * 100:.2f}%), Correct negative: {total_neg_correct} ({avg_correct_neg * 100:.2f}%)")
    print(f"        Total correct: {total_num_correct} ({avg_num_correct * 100:.2f}%)")
    print(f"        Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_num_correct:.4f}")
    return avg_loss, avg_num_correct, avg_correct_pos, avg_correct_neg


def evaluate(model, loss_fn, dataloader, margin):
    model.eval()
    total_loss = 0
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

            # Compute loss
            loss = loss_fn(emb_a_central, emb_p_central, emb_n_central)
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
    avg_loss = total_loss / len(dataloader)
    avg_correct_pos = total_pos_correct / total_num_samples
    avg_correct_neg = total_neg_correct / total_num_samples
    avg_num_correct = total_num_correct / (2 * total_num_samples)  # Since we have two conditions

    print(f"        Correct positive: {total_pos_correct} ({avg_correct_pos * 100:.2f}%), Correct negative: {total_neg_correct} ({avg_correct_neg * 100:.2f}%)")
    print(f"        Total correct: {total_num_correct} ({avg_num_correct * 100:.2f}%)")
    print(f"        Eval Loss: {avg_loss:.4f}, Eval Accuracy: {avg_num_correct:.4f}")

    return avg_loss, avg_num_correct, avg_correct_pos, avg_correct_neg


def test_and_eval(model, loss_fn, dataloader, margin):
    model.eval()
    total_loss = 0
    total_true = 0
    total_true_pos = 0
    total_true_neg = 0
    total_num_triplets = 0

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

            loss = loss_fn(emb_a_central, emb_p_central, emb_n_central)

            total_loss += loss.item()

            # Compute distances
            d_ap = F.pairwise_distance(emb_a_central, emb_p_central)
            d_an = F.pairwise_distance(emb_a_central, emb_n_central)

            # Determine correct predictions based on margin
            true_pos = (d_ap < margin).cpu()
            true_neg = (d_an > margin).cpu()

            # Sum up correct predictions
            true_pos = true_pos.sum().item()
            true_neg = true_neg.sum().item()
            true_pos_neg = true_pos + true_neg

            total_true += true_pos_neg
            total_true_pos += true_pos
            total_true_neg += true_neg
            total_num_triplets += len(batch_anchor)

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_true_pos = total_true_pos / total_num_triplets
    avg_true_neg = total_true_neg / total_num_triplets
    avg_num_correct = total_true / (2 * total_num_triplets)  # Since we have two conditions

    # True positive: Correctly predicted positive pairs (total_true_pos)
    # False positive: Negative pairs incorrectly predicted as positive (total_num_triplets - total_true_neg)
    # True negative: Correctly predicted negative pairs (total_true_neg)
    # False negative: Positive pairs incorrectly predicted as negative (total_num_triplets - total_true_pos)
    precision, recall, f1 = compute_metrics(
        true_pos=total_true_pos,
        false_pos=total_num_triplets - total_true_neg,
        false_neg=total_num_triplets - total_true_pos
    )

    # Compute average loss
    print(f"        Correct positive: {total_true_pos} ({avg_true_pos * 100:.2f}%), Correct negative: {total_true_neg} ({avg_true_neg * 100:.2f}%)")
    print(f"        Total correct: {total_true} ({avg_num_correct * 100:.2f}%)")
    print(f"        Test/Eval Loss: {avg_loss:.4f}, Test/Eval Accuracy: {avg_num_correct:.4f}")
    print(f"        Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return avg_loss, avg_num_correct, avg_true_pos, avg_true_neg, precision, recall, f1


def compute_metrics(true_pos, false_pos, false_neg):
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

