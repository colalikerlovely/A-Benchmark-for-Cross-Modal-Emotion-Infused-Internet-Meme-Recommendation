from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch
import pandas as pd


def compute_recall_at_k(x, loaded_matrix_csv, k):
    top_k_indices = np.argsort(x, axis=1)[:, -k:]
    num_correct = np.sum(np.any(loaded_matrix_csv[np.arange(x.shape[0])[:, None], top_k_indices] == 1, axis=1))
    return (num_correct / np.sum(loaded_matrix_csv == 1)) * 100

def new_compute_metrics(x, loaded_matrix_csv):
    metrics = {}

    # Recall@1
    metrics['R@1'] = compute_recall_at_k(x, loaded_matrix_csv, 1)

    # Recall@5
    metrics['R@5'] = compute_recall_at_k(x, loaded_matrix_csv, 5)

    # Recall@10
    metrics['R@10'] = compute_recall_at_k(x, loaded_matrix_csv, 10)

    # Recall@20
    metrics['R@20'] = compute_recall_at_k(x, loaded_matrix_csv, 20)

    # Recall@50
    metrics['R@50'] = compute_recall_at_k(x, loaded_matrix_csv, 50)

    return metrics


def compute_precision_at_k(x, loaded_matrix_csv, k):
    top_k_indices = np.argsort(x, axis=1)[:, -k:]
    num_correct = 0
    for i in range(x.shape[0]):
        if np.any(loaded_matrix_csv[i, top_k_indices[i]] == 1):
            num_correct += 1
    return (num_correct / x.shape[0]) * 100

def compute_metrics_precision(x, loaded_matrix_csv):
    metrics = {}
    # Precision@10
    metrics['P@10'] = compute_precision_at_k(x, loaded_matrix_csv, 10)

    # Precision@20
    metrics['P@20'] = compute_precision_at_k(x, loaded_matrix_csv, 20)

    # Precision@50
    metrics['P@50'] = compute_precision_at_k(x, loaded_matrix_csv, 50)

    return metrics