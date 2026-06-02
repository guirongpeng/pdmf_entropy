"""
有限高斯-PDMF 模糊数集合的归一化（normalization.md 定义 2 / 式 (3)）。

对索引集 I = {1,...,N} 上的一组模糊数 {x̃_i}，得到归一化表示 {ñ_i}：

    ñ_i = ⟨ (x_i - m)/(M - m), exp(d_i^- / M_d), exp(d_i^+ / M_d), μ_i^-, μ_i^+ ⟩

其中 m = min_i x_i, M = max_i x_i；M_d = max{ d_i^-, d_i^+ : i ∈ I }。

实现说明：
- 与 GaussianPDMF 五元组 ⟨x0; d-, d+, μ-, μ+⟩ 一一对应；归一化后仍构造为 GaussianPDMF，
  供 membership / fuzzy_entropy_fast 等后续使用。
- 当 M = m（全体中心相同）时，(x_i-m)/(M-m) 无定义，退化为全体 x0' = 0.5。
- 当 batch 为空或单元素时仍给出确定行为，避免除零。
"""

from __future__ import annotations

import math
from typing import Sequence

from .models import GaussianPDMF


def normalize_gaussian_pdmf_batch_definition2(
    fuzzy_nums: Sequence[GaussianPDMF],
    eps: float = 1e-12,
) -> list[GaussianPDMF]:
    """
    对一批 GaussianPDMF 应用 normalization.md 定义 2。

    :param fuzzy_nums: 同一属性列上、与样本顺序一致的模糊数列表（长度 N）。
    :param eps: 避免 M-m、M_d 过小导致数值问题。
    :return: 同长度的归一化模糊数列表。
    """
    if len(fuzzy_nums) == 0:
        return []

    xs = [float(f.x0) for f in fuzzy_nums]
    m = min(xs)
    M = max(xs)
    span = M - m
    if span <= eps:
        x0_norm = [0.5] * len(fuzzy_nums)
    else:
        x0_norm = [(xi - m) / span for xi in xs]

    d_all: list[float] = []
    for f in fuzzy_nums:
        d_all.append(float(f.d_minus))
        d_all.append(float(f.d_plus))
    M_d = max(d_all) if d_all else eps
    if M_d <= eps:
        M_d = eps

    out: list[GaussianPDMF] = []
    for i, f in enumerate(fuzzy_nums):
        dm = float(f.d_minus)
        dp = float(f.d_plus)
        d_m_n = math.exp(dm / M_d)
        d_p_n = math.exp(dp / M_d)
        out.append(
            GaussianPDMF(
                x0=float(x0_norm[i]),
                d_minus=d_m_n,
                d_plus=d_p_n,
                mu_minus=float(f.mu_minus),
                mu_plus=float(f.mu_plus),
            )
        )
    return out
