"""
auto_construct
==============

按 create_d_miu.md 的规则，从单个数值属性列自动构造高斯-PDMF 模糊数。
"""

from __future__ import annotations

from typing import List, Literal, Sequence

import numpy as np

from .models import GaussianPDMF

MuMethod = Literal["A", "B", "C"]


def _validate_input_column(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("输入列不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError("输入列包含 NaN/Inf，无法构造模糊数")
    return arr


def _compute_d_from_sorted(x_sorted: np.ndarray, k: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n = x_sorted.size
    d_minus = np.zeros(n, dtype=float)
    d_plus = np.zeros(n, dtype=float)

    if n == 1:
        return np.array([eps], dtype=float), np.array([eps], dtype=float)

    for i in range(n):
        xi = x_sorted[i]
        m_l = min(k, i)
        m_r = min(k, n - 1 - i)

        if m_l > 0:
            left_vals = x_sorted[i - m_l : i]
            d_minus[i] = float(np.mean(xi - left_vals))
        if m_r > 0:
            right_vals = x_sorted[i + 1 : i + 1 + m_r]
            d_plus[i] = float(np.mean(right_vals - xi))

        # 单侧补齐：缺哪侧，用另一侧补
        if m_l == 0 and m_r > 0:
            d_minus[i] = d_plus[i]
        if m_r == 0 and m_l > 0:
            d_plus[i] = d_minus[i]

    d_minus = np.maximum(d_minus, eps)
    d_plus = np.maximum(d_plus, eps)
    return d_minus, d_plus


def _knn_avg_distance_1d(x: np.ndarray, k: int, eps: float) -> np.ndarray:
    n = x.size
    if n == 1:
        return np.array([0.0], dtype=float)

    m = min(k, n - 1)
    avg_dist = np.zeros(n, dtype=float)
    for i in range(n):
        dist = np.abs(x - x[i])
        dist[i] = np.inf
        nearest = np.partition(dist, m - 1)[:m]
        avg_dist[i] = float(np.mean(nearest))
    return np.maximum(avg_dist, 0.0) + 0.0 * eps


def _compute_mu(x: np.ndarray, method: MuMethod, k: int, eps: float) -> np.ndarray:
    method = method.upper()  # type: ignore[assignment]
    if method == "A":
        avg_dist = _knn_avg_distance_1d(x, k=k, eps=eps)
        rho = 1.0 / (avg_dist + eps)
        rho_min = float(np.min(rho))
        rho_max = float(np.max(rho))
        if rho_max - rho_min <= eps:
            rho_norm = np.full_like(rho, 0.5)
        else:
            rho_norm = (rho - rho_min) / (rho_max - rho_min)
        mu = 2.0 * rho_norm - 1.0
    elif method == "B":
        mean_x = float(np.mean(x))
        std_x = float(np.std(x))
        z = np.abs(x - mean_x) / (std_x + eps)
        mu = 1.0 - z
    elif method == "C":
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max - x_min <= eps:
            x_norm = np.full_like(x, 0.5)
        else:
            x_norm = (x - x_min) / (x_max - x_min + eps)
        mu = 2.0 * x_norm - 1.0
    else:
        raise ValueError("mu_method 仅支持 'A'、'B'、'C'")

    return np.clip(mu, -1.0, 1.0)


def create_gaussian_pdmf_from_column(
    x: Sequence[float],
    k: int = 3,
    mu_method: MuMethod = "A",
    eps: float = 1e-8,
) -> List[GaussianPDMF]:
    """
    对单个数值属性列自动构造高斯-PDMF 模糊数列表（输出顺序与原输入一致）。
    """
    if k < 1:
        raise ValueError("k 必须 >= 1")
    if eps <= 0:
        raise ValueError("eps 必须 > 0")

    x_arr = _validate_input_column(x)
    n = x_arr.size

    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    d_minus_sorted, d_plus_sorted = _compute_d_from_sorted(x_sorted=x_sorted, k=k, eps=eps)

    d_minus = np.empty(n, dtype=float)
    d_plus = np.empty(n, dtype=float)
    d_minus[order] = d_minus_sorted
    d_plus[order] = d_plus_sorted

    mu = _compute_mu(x_arr, method=mu_method, k=k, eps=eps)

    return [
        GaussianPDMF(
            x0=float(x_arr[i]),
            d_minus=float(d_minus[i]),
            d_plus=float(d_plus[i]),
            mu_minus=float(mu[i]),
            mu_plus=float(mu[i]),
        )
        for i in range(n)
    ]


def fuzzify_column_to_array(
    x: Sequence[float],
    k: int = 3,
    mu_method: MuMethod = "A",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    便捷输出：返回 shape=(n,5) 的数组，每行是 [x, d-, d+, mu-, mu+]。
    """
    fuzzy_list = create_gaussian_pdmf_from_column(x=x, k=k, mu_method=mu_method, eps=eps)
    out = np.array(
        [[f.x0, f.d_minus, f.d_plus, f.mu_minus, f.mu_plus] for f in fuzzy_list],
        dtype=float,
    )
    return out

