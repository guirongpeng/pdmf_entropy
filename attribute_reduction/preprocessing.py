"""Shared data preprocessing utilities for FRS attribute reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def prepare_dataframe(file_path=None, data=None, delete_columns=None, target_name="target"):
    """Load and normalize dataframe for downstream algorithms."""
    if data is None and file_path is None:
        raise ValueError("Either file_path or data must be provided.")
    if data is None:
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                data = pd.read_csv(file_path, encoding=enc)
                break
            except UnicodeDecodeError as err:
                last_err = err
        else:
            raise last_err
    else:
        data = data.copy()

    delete_columns = delete_columns or []
    for col in delete_columns:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)

    y = data[target_name].values
    data = data.drop(target_name, axis=1)
    data[target_name] = y

    for col in data.columns:
        try:
            data[col] = data[col].astype(float)
        except Exception:
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col])
        if col == target_name:
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col])

    data = data.fillna(data.mean(numeric_only=True))
    data[target_name] = data[target_name].astype(int)
    return data


def split_fold_array(
    data: pd.DataFrame,
    train_index,
    target_name="target",
    scaler=1,
):
    """Extract fold array and apply scaler to condition attributes only."""
    fold = data.iloc[train_index].copy()
    fold.index = list(range(fold.shape[0]))
    y = fold[target_name].values
    x_cols = list(fold.columns[:-1].values)

    if scaler == 1:
        x = np.array(fold[x_cols])
    elif scaler == 2:
        x = MinMaxScaler().fit_transform(fold[x_cols])
    else:
        x = StandardScaler().fit_transform(fold[x_cols])

    xy = np.c_[x, y]
    return x, xy, x_cols

