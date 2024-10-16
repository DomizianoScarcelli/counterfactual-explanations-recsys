from typing import Dict, Tuple


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, accuracy, recall

def print_confusion_matrix(tp: int, fp: int, tn: int, fn: int) -> None:
    print(f"""
    Confusion matrix:
    ---------------
    | TP: {tp}  | FP: {fp} |
    ---------------
    | FN: {fn}  | TN: {tn} |
    ---------------
    """)

