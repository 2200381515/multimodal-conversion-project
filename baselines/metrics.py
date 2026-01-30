from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


def binary_metrics(y_true, y_prob, thr: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    # 若 test 折只有单一类别，AUC 会无定义
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = float(tp / (tp + fn + 1e-12))
    spe = float(tn / (tn + fp + 1e-12))

    return {
        "AUC": auc,
        "F1": f1,
        "ACC": acc,
        "SEN": sen,
        "SPE": spe,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }
