from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def train_logreg(Xtr: np.ndarray, ytr: np.ndarray) -> Pipeline:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])
    clf.fit(Xtr, ytr)
    return clf


def train_svm_rbf(Xtr: np.ndarray, ytr: np.ndarray) -> Pipeline:
    # probability=True 用于输出 predict_proba -> AUC
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced"
        ))
    ])
    clf.fit(Xtr, ytr)
    return clf


def predict_prob_positive(clf: Pipeline, X: np.ndarray) -> np.ndarray:
    prob = clf.predict_proba(X)[:, 1]
    return prob.astype(float)
