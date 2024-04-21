import torch
import numpy as np

def accuracy(predicted, target):
    """
    (TP + TN) / (TP + TN + FP + FN)
    """
    correct = (predicted == target).sum().item()
    total = len(target)
    accuracy = correct / total
    return accuracy

def precision(predicted, target):
    """
    TP / (TP + FP)
    """
    TP = ((1-predicted) * (1-target)).sum().item()
    FP = ((1-predicted) * target).sum().item()
    precision = TP / (TP + FP)
    return precision

def recall(predicted, target):
    """
    TP / (TP + FN)
    """

    TP = ((1-predicted) * (1-target)).sum().item()
    FN = (predicted * (1-target)).sum().item()
    recall = TP / (TP + FN)
    return recall

def f1(predicted, target):
    """
    2 * (precision * recall) / (precision + recall)
    """
    prec = precision(predicted, target)
    rec = recall(predicted, target)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1