"""Automated metrics: accuracy + confusion, run against the golden set."""
from collections import Counter, defaultdict


def accuracy(preds: list, golds: list) -> float:
    assert len(preds) == len(golds)
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(golds) if golds else 0.0


def confusion(preds: list, golds: list) -> dict:
    cm = defaultdict(Counter)
    for p, g in zip(preds, golds):
        cm[g][p] += 1
    return {g: dict(c) for g, c in cm.items()}


def per_class_report(preds: list, golds: list) -> dict:
    labels = sorted(set(golds) | set(preds))
    report = {}
    for label in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, golds) if p != label and g == label)
        support = sum(1 for g in golds if g == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {
            "support": support,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        }
    return report
