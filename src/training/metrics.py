"""
Training metrics helpers.
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


@torch.no_grad()
def evaluate_epoch(model, loader, device, num_classes, criterion=None):
    """Evaluate a model over a DataLoader. Returns dict of metrics.

    Works for both train (grads off but used to report metrics) and val/test.
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item()
            n_batches += 1
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(probs, axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "n_samples": int(len(labels)),
    }
    if n_batches:
        metrics["loss"] = total_loss / n_batches
    return metrics, probs, labels


def compute_class_weights_from_loader(loader, num_classes, device):
    """Balanced class weights computed ONLY from a training loader."""
    counts = np.zeros(num_classes)
    for _, y in loader:
        labels = y.numpy()
        for l in labels:
            counts[l] += 1
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)