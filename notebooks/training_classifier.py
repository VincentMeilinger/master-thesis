import torch
import torch.nn.functional as F

from src.shared.graph_schema import EdgeType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(triplet_classifier_model, loss_fn, data_dict, optimizer):
    triplet_classifier_model.embedding_net.train()
    optimizer.zero_grad()

    emb_a, emb_p, emb_n = triplet_classifier_model(data_dict)

    loss = loss_fn(emb_a, emb_p, emb_n)

    loss.backward()
    optimizer.step()

    batch_loss = loss.item()
    # print(f"Batch loss: {batch_loss:.4f}")
    return batch_loss


def test(triplet_classifier_model, loss_fn, dataloader, margin):
    triplet_classifier_model.embedding_net.eval()
    total_loss = 0
    total_num_correct = 0
    total_pos_correct = 0
    total_neg_correct = 0
    total_num_samples = 0

    with torch.no_grad():
        for data_dict in dataloader:
            emb_a, emb_p, emb_n = triplet_classifier_model(data_dict)

            loss = loss_fn(emb_a, emb_p, emb_n)
            total_loss += loss.item()

            # Compute distances
            d_ap = F.pairwise_distance(emb_a, emb_p)
            d_an = F.pairwise_distance(emb_a, emb_n)

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
            for values in data_dict.values():
                batch_len = len(values[0])
                break
            total_num_samples += batch_len

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_correct_pos = total_pos_correct / total_num_samples
    avg_correct_neg = total_neg_correct / total_num_samples
    avg_num_correct = total_num_correct / (2 * total_num_samples)

    # Compute average loss
    print(f"        Correct positive: {total_pos_correct} ({avg_correct_pos * 100:.2f}%), Correct negative: {total_neg_correct} ({avg_correct_neg * 100:.2f}%)")
    print(f"        Total correct: {total_num_correct} ({avg_num_correct * 100:.2f}%)")
    print(f"        Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_num_correct:.4f}")
    return avg_loss, avg_num_correct, avg_correct_pos, avg_correct_neg


def evaluate(triplet_classifier_model, loss_fn, dataloader, margin):
    triplet_classifier_model.embedding_net.eval()
    total_loss = 0
    total_num_correct = 0
    total_pos_correct = 0
    total_neg_correct = 0
    total_num_samples = 0
    with torch.no_grad():
        for data_dict in dataloader:
            emb_a, emb_p, emb_n = triplet_classifier_model(data_dict)

            loss = loss_fn(emb_a, emb_p, emb_n)
            total_loss += loss.item()

            # Compute distances
            d_ap = F.pairwise_distance(emb_a, emb_p)
            d_an = F.pairwise_distance(emb_a, emb_n)

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
            for values in data_dict.values():
                batch_len = len(values[0])
                break
            total_num_samples += batch_len

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_correct_pos = total_pos_correct / total_num_samples
    avg_correct_neg = total_neg_correct / total_num_samples
    avg_num_correct = total_num_correct / (2 * total_num_samples)  # Since we have two conditions

    print(f"        Correct positive: {total_pos_correct} ({avg_correct_pos * 100:.2f}%), Correct negative: {total_neg_correct} ({avg_correct_neg * 100:.2f}%)")
    print(f"        Total correct: {total_num_correct} ({avg_num_correct * 100:.2f}%)")
    print(f"        Eval Loss: {avg_loss:.4f}, Eval Accuracy: {avg_num_correct:.4f}")

    return avg_loss, avg_num_correct, avg_correct_pos, avg_correct_neg