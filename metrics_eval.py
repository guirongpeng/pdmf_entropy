from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def eval_subset(train_df: pd.DataFrame, test_df: pd.DataFrame, subset_cols, target_name: str):
    model = LogisticRegression(max_iter=2000)
    model.fit(train_df[subset_cols], train_df[target_name])
    pred = model.predict(test_df[subset_cols])
    return compute_metrics(test_df[target_name], pred)

