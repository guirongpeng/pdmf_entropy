"""IARFCIE attribute reduction module."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_distances(df: pd.DataFrame):
    """Build per-attribute normalized distance matrices."""
    result = {}
    for column in df.columns:
        col_values = df[column].values.astype(float)
        denominator = col_values.max() - col_values.min()
        if denominator == 0:
            distance_matrix = np.zeros((len(col_values), len(col_values)))
        else:
            distance_matrix = np.abs(col_values[:, np.newaxis] - col_values) / denominator
        result[column] = distance_matrix
    return result


def calculate_rp_optimized(p, select_col, threshold):
    n = p[select_col[0]].shape[0]
    rp = np.zeros((n, n))
    matrices = [p[col] for col in select_col]
    for matrix_distances in matrices:
        rp += (matrix_distances < threshold) * matrix_distances
    return rp


def calculate_conditional_entropy(rp, d):
    n = rp.shape[0]
    unique_classes = np.unique(d)
    conditional_entropy = 0.0

    for i in range(n):
        theta_rp_i = rp[i]
        rp_theta_count = np.sum(theta_rp_i)
        if rp_theta_count == 0:
            continue
        for class_j in unique_classes:
            class_idx = np.where(d == class_j)[0]
            intersection_count = np.sum(theta_rp_i[class_idx])
            if intersection_count == 0:
                continue
            class_probability = intersection_count / rp_theta_count
            conditional_entropy -= (intersection_count / n) * np.log2(class_probability)
    return conditional_entropy


def h(select_col, threshold, d, p):
    # 当选择集为空时，返回正无穷，使得调用方判定“移除该属性会改变度量”，从而不保留空集
    if not select_col:
        return float("inf")
    rp = calculate_rp_optimized(p, select_col, threshold)
    return calculate_conditional_entropy(rp, d)


def reduce_iarfcie(data: pd.DataFrame, target_name="target", threshold=0.2):
    # 实验层遍历参数，再在外层比较效果（通常用分类准确率挑最优）
    
    """Run IARFCIE and return reduced feature names."""
    d = data[target_name].values
    cond_df = data.drop(columns=[target_name])
    columns = list(cond_df.columns)
    p = calculate_distances(cond_df)

    s = columns[:]
    init_h = h(s, threshold, d, p)
    for col in columns:
        if col not in s:
            continue
        s.remove(col)
        if h(s, threshold, d, p) != init_h:
            s.append(col)
    return s

