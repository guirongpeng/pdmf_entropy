"""
metrics
=======

Gaussian-PDMF 模糊数的熵与相似度计算。

实现对应文档：
- entropy.md
- similarity.md
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .core import gaussian_cdf, subjective_perception_h
from .models import GaussianPDMF


def _safe_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip probability values to avoid log(0) in entropy integrand."""
    return np.clip(p, eps, 1.0 - eps)


def _h_integrand_for_mu(mu: float, x_grid: np.ndarray) -> np.ndarray:
    """
    Integrand: -f*ln(f) - (1-f)*ln(1-f), where f(x;mu) = Phi(t(x); mu, 1).
    """
    t = subjective_perception_h(x_grid)
    # gaussian_cdf is scalar-friendly; vectorized wrapper keeps code consistent.
    f = np.vectorize(lambda v: gaussian_cdf(v, mu), otypes=[float])(t)
    f = _safe_prob(f)
    return -(f * np.log(f) + (1.0 - f) * np.log(1.0 - f))


def h_mu(mu: float, num_points: int = 4001, edge_eps: float = 1e-6) -> float:
    """
    计算 H(mu) = integral_0^1 [ -f ln f - (1-f) ln(1-f) ] dx.

    数值实现细节：
    - 在 (edge_eps, 1-edge_eps) 上做均匀网格积分，避开 tan 在端点的奇异；
    - 使用梯形积分。
    """
    if num_points < 101:
        raise ValueError("num_points 建议至少为 101")
    if not (0.0 < edge_eps < 1e-2):
        raise ValueError("edge_eps 需在 (0, 1e-2) 内")

    x_grid = np.linspace(edge_eps, 1.0 - edge_eps, num_points)
    integrand = _h_integrand_for_mu(mu, x_grid)
    return float(np.trapz(integrand, x_grid))


@lru_cache(maxsize=4096)
def _h_mu_rounded_cached(mu_rounded: float, num_points: int, edge_eps: float) -> float:
    """Cache h_mu on rounded mu for speed in repetitive calls."""
    return h_mu(mu=mu_rounded, num_points=num_points, edge_eps=edge_eps)


@lru_cache(maxsize=1)
def _h0_cached() -> float:
    """
    Cache H(0) as denominator normalization constant.
    该值在当前数值配置（sigma=1, num_points=4001, edge_eps=1e-6）下固定为约 0.3702632681。
    """
    return 0.3702632680756489


def fuzzy_entropy(fuzzy_num: GaussianPDMF) -> float:
    """
    文档公式：
    E(x~) = [1/(2*H(0))] * [ exp(-1/d-) * H(mu-) + exp(-1/d+) * H(mu+) ].
    """
    h0 = _h0_cached()
    if h0 <= 0:
        raise ValueError("H(0) 计算异常，无法完成归一化")

    left = math.exp(-1.0 / fuzzy_num.d_minus) * h_mu(fuzzy_num.mu_minus)
    right = math.exp(-1.0 / fuzzy_num.d_plus) * h_mu(fuzzy_num.mu_plus)
    ent = (left + right) / (2.0 * h0)
    # 数值误差下做轻微裁剪
    return float(np.clip(ent, 0.0, 1.0))


def fuzzy_entropy_fast(
    fuzzy_num: GaussianPDMF,
    num_points: int = 801,
    edge_eps: float = 1e-6,
    mu_round_digits: int = 3,
) -> float:
    """
    快速熵：对 mu 做量化并缓存 H(mu)，降低重复积分开销。
    """
    if mu_round_digits < 0:
        raise ValueError("mu_round_digits 必须 >= 0")
    if num_points < 101:
        raise ValueError("num_points 建议至少为 101")
    if not (0.0 < edge_eps < 1e-2):
        raise ValueError("edge_eps 需在 (0, 1e-2) 内")

    h0 = _h0_cached()
    if h0 <= 0:
        raise ValueError("H(0) 计算异常，无法完成归一化")

    mu_minus = round(float(fuzzy_num.mu_minus), mu_round_digits)
    mu_plus = round(float(fuzzy_num.mu_plus), mu_round_digits)
    h_minus = _h_mu_rounded_cached(mu_minus, int(num_points), float(edge_eps))
    h_plus = _h_mu_rounded_cached(mu_plus, int(num_points), float(edge_eps))

    left = math.exp(-1.0 / fuzzy_num.d_minus) * h_minus
    right = math.exp(-1.0 / fuzzy_num.d_plus) * h_plus
    ent = (left + right) / (2.0 * h0)
    return float(np.clip(ent, 0.0, 1.0))


def fuzzy_similarity(a: GaussianPDMF, b: GaussianPDMF, lam: float = 0.5) -> float:
    """
    文档公式：
    S_lam(a,b) = lam*exp(-|x1-x2|)
                 + [(1-lam)/(2*H(0))] * [
                     exp(-|d1- - d2-|) * H(mu1- - mu2-)
                     + exp(-|d1+ - d2+|) * H(mu1+ - mu2+)
                   ].
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("lam 必须在 (0, 1) 内")

    h0 = _h0_cached()
    if h0 <= 0:
        raise ValueError("H(0) 计算异常，无法完成归一化")

    part_center = lam * math.exp(-abs(a.x0 - b.x0))
    part_shape = (
        math.exp(-abs(a.d_minus - b.d_minus)) * h_mu(a.mu_minus - b.mu_minus)
        + math.exp(-abs(a.d_plus - b.d_plus)) * h_mu(a.mu_plus - b.mu_plus)
    )
    sim = part_center + ((1.0 - lam) / (2.0 * h0)) * part_shape
    # 理论上 (0,1]，数值误差裁剪到 [0,1]
    return float(np.clip(sim, 0.0, 1.0))

