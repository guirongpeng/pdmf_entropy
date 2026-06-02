"""ARPDMF 属性约简（理论见 gaussian_pdmf/arpdmf.md）。

Gaussian PDMF 样本模糊熵 → 权重分布 → 属性子集 Shannon 熵 E(S) → 三阶段约简。

仅在本模块内：对「当前传入的 train_df」剔除条件属性里取值完全常数的列
（nunique<=1），再参与模糊化与 E(S)。不修改全局数据，也不作用于其他约简算法。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .gaussian_pdmf import (
    create_gaussian_pdmf_from_column,
    fuzzy_entropy_fast,
    normalize_gaussian_pdmf_batch_definition2,
)


def condition_columns(train_df: pd.DataFrame, target_name: str) -> list[str]:
    """条件属性列名：数据框中除目标列外的列。"""
    return [c for c in train_df.columns if c != target_name]


def _arpdmf_usable_condition_columns(train_df: pd.DataFrame, target_name: str) -> list[str]:
    """
    ARPDMF 专用：在当前折 train_df 上，去掉「条件属性取值无变化」的列。
    常数列下 PDMF 宽度退化、样本熵无区分度，且易导致下游仅选到单常数特征。
    """
    cols = condition_columns(train_df, target_name)
    return [c for c in cols if train_df[c].nunique(dropna=False) > 1]


def _attribute_subset_entropy(E_mat: np.ndarray, col_indices: Sequence[int], eps: float = 1e-300) -> float:
    """E(S)：E_i^(S)=1-∏(1-E_i^(a))，再 w_i^(S)∝1-E_i^(S)，Shannon 熵。空集 E(∅)=ln n。"""
    n = E_mat.shape[0]
    if n == 0:
        return 0.0
    if len(col_indices) == 0:
        return float(np.log(n))

    sub = E_mat[:, list(col_indices)]
    one_minus_prod = np.prod(1.0 - sub, axis=1)
    E_i_S = np.clip(1.0 - one_minus_prod, 0.0, 1.0)
    strength = 1.0 - E_i_S
    s = float(np.sum(strength))
    if s <= eps:
        w = np.full(n, 1.0 / n)
    else:
        w = strength / s
    w = np.clip(w, eps, 1.0)
    w = w / np.sum(w)
    return float(-np.sum(w * np.log(w)))


def _compute_sample_entropy_matrix(
    train_df: pd.DataFrame,
    columns: Sequence[str],
    k: int,
    mu_method: str,
    entropy_num_points: int,
    mu_round_digits: int,
    fuzzy_batch_normalize: bool = True,
) -> np.ndarray:
    """每列模糊化后各样本的样本模糊熵 E_i^(a)，形状 (n, m)。

    fuzzy_batch_normalize：为 True 时，对「该列上全体样本模糊数」先按 normalization.md 定义 2
    批归一化，再算熵；为 False 时保持旧行为（仅模糊化，不归一化模糊数）。
    """
    n = train_df.shape[0]
    m = len(columns)
    if m == 0:
        return np.zeros((n, 0))
    cols_E: list[np.ndarray] = []
    for col in columns:
        x = train_df[col].to_numpy(dtype=float)
        fuzzy_nums = create_gaussian_pdmf_from_column(
            x,
            k=k,
            mu_method=mu_method,
            eps=1e-8,
        )
        if fuzzy_batch_normalize:
            fuzzy_nums = normalize_gaussian_pdmf_batch_definition2(fuzzy_nums)
        cols_E.append(
            np.array(
                [
                    fuzzy_entropy_fast(
                        fn,
                        num_points=entropy_num_points,
                        mu_round_digits=mu_round_digits,
                    )
                    for fn in fuzzy_nums
                ],
                dtype=float,
            )
        )
    return np.column_stack(cols_E)


def prepare_arpdmf_entropy_matrix(
    train_df: pd.DataFrame,
    target_name: str,
    k: int,
    mu_method: str,
    entropy_num_points: int = 101,
    mu_round_digits: int = 3,
    fuzzy_batch_normalize: bool = True,
) -> tuple[list[str], np.ndarray]:
    """
    模糊化 + 批归一化 + 样本模糊熵，得到 E_mat（形状 n×m）。
    仅依赖 (k, mu_method)，与 delta 无关；宜按 (k, mu) 每折缓存一次。
    """
    cond_cols = _arpdmf_usable_condition_columns(train_df, target_name)
    if not cond_cols:
        return [], np.zeros((train_df.shape[0], 0))
    E_mat = _compute_sample_entropy_matrix(
        train_df,
        cond_cols,
        k=k,
        mu_method=mu_method,
        entropy_num_points=entropy_num_points,
        mu_round_digits=mu_round_digits,
        fuzzy_batch_normalize=fuzzy_batch_normalize,
    )
    return cond_cols, E_mat


def reduce_arpdmf_from_matrix(
    cond_cols: list[str],
    E_mat: np.ndarray,
    delta: float,
) -> list[str]:
    """
    在已有 E_mat 上执行三阶段约简（Sig_out / Sig_in / δ），不含模糊化与归一化。
    """
    if not cond_cols or E_mat.size == 0:
        return []

    m = len(cond_cols)
    idx_all = list(range(m))
    E_C = _attribute_subset_entropy(E_mat, idx_all)

    # 阶段 1：Sig_out > 0
    red: list[int] = []
    while True:
        best_j: int | None = None
        best_sig = -np.inf
        E_red = _attribute_subset_entropy(E_mat, red)
        for j in range(m):
            if j in red:
                continue
            sig_out = _attribute_subset_entropy(E_mat, red + [j]) - E_red
            if sig_out > best_sig:
                best_sig = sig_out
                best_j = j
        if best_j is not None and best_sig > 0.0:
            red.append(best_j)
        else:
            break

    # 阶段 2：Sig_in <= 0 则删
    changed = True
    while changed:
        changed = False
        E_red = _attribute_subset_entropy(E_mat, red)
        for j in list(red):
            red_without = [i for i in red if i != j]
            sig_in = E_red - _attribute_subset_entropy(E_mat, red_without)
            if sig_in <= 0.0:
                red.remove(j)
                changed = True

    # 阶段 3：|E(red)-E(C)| > delta
    while abs(_attribute_subset_entropy(E_mat, red) - E_C) > float(delta):
        if len(red) >= m:
            break
        best_j: int | None = None
        best_val = -np.inf
        for j in range(m):
            if j in red:
                continue
            val = _attribute_subset_entropy(E_mat, red + [j])
            if val > best_val:
                best_val = val
                best_j = j
        if best_j is None:
            break
        red.append(best_j)
        if abs(_attribute_subset_entropy(E_mat, red) - E_C) <= float(delta):
            break

    return [cond_cols[i] for i in sorted(red)]


def reduce_arpdmf(
    train_df: pd.DataFrame,
    target_name: str,
    delta: float,
    k: int = 3,
    mu_method: str = "A",
    entropy_num_points: int = 101,
    mu_round_digits: int = 3,
    fuzzy_batch_normalize: bool = True,
) -> list[str]:
    """
    完整 ARPDMF：先 prepare（模糊化+熵矩阵），再三阶段约简。
    实验计时请用 prepare + reduce_arpdmf_from_matrix 分离，仅对后者计时。
    """
    cond_cols, E_mat = prepare_arpdmf_entropy_matrix(
        train_df,
        target_name=target_name,
        k=k,
        mu_method=mu_method,
        entropy_num_points=entropy_num_points,
        mu_round_digits=mu_round_digits,
        fuzzy_batch_normalize=fuzzy_batch_normalize,
    )
    if not cond_cols:
        return []
    return reduce_arpdmf_from_matrix(cond_cols, E_mat, delta)
