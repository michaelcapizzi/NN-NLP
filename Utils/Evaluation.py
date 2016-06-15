from sklearn.metrics import *
import numpy as np

def accuracy(correct, total):
    if correct == 0:
        return float(0)
    else:
        return float(correct) / float(total)

##################

def precision_manual(tp, fp):
    if tp == 0:
        return float(0)
    else:
        return float(tp) / float(tp + fp)


def precision_sk(y_true, y_pred):
    return precision_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=0
    )

##################

def recall_manual(tp, fn):
    if tp == 0:
        return 0
    else:
        return float(tp) / float(tp + fn)


def recall_sk(y_true, y_pred):
    return recall_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=0
    )

##################

def f1_manual(p, r):
    if p == 0 or r == 0:
        return float(0)
    else:
        return float(2 * p * r) / float(p + r)


def f1_sk(y_true, y_pred):
    return f1_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=0
    )