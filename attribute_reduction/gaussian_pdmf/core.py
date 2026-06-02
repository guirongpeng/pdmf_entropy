"""
core
====

存放与具体模糊数无关的**基础数学工具函数**：

- 主观感知函数 `h(t)`：将比例刻度映射到实轴；
- 高斯 PDF/CDF：提供“形状”和“累积概率”信息，用于把 `h(t)` 映射回 [0, 1] 的隶属度。

这些函数都是纯函数，不依赖任何全局状态。
"""

import numpy as np
from scipy.stats import norm


def subjective_perception_h(t: float) -> float:
    """
    主观感知函数 h: [0,1] → R（论文公式 2.2）

    通过一个类似 logit 的单调变换，把比例刻度 `t` 映射到整个实数轴：

    - 当 t → 0⁺ 时，h(t) → -∞；
    - 当 t → 1⁻ 时，h(t) → +∞。

    这样后续可以在实轴上叠加高斯形状，再通过 CDF 映回 [0, 1] 作为隶属度。

    :param t: 比例刻度，数学上应满足 0 < t < 1。
    :return: 实轴上的映射值 h(t)。
    """
    return np.tan(np.pi * t - np.pi / 2)


def gaussian_pdf(t: float, mu: float, sigma: float = 1.0) -> float:
    """
    一维高斯概率密度函数（论文公式 2.2 所依赖的形状函数）。

    这里 sigma 默认固定为 1，对应论文中“标准化”的设定，
    只让位置参数 `mu` 控制形状的平移，而不单独控制“尖锐/平缓”程度。

    :param t: 自变量，取值范围为实数。
    :param mu: 位置参数（均值），决定峰值所在位置。
    :param sigma: 标准差，默认 1，可选修改用于拓展模型。
    :return: 在 t 处的 PDF 值。
    """
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(t - mu) ** 2 / (2 * sigma**2))


def gaussian_cdf(t: float, mu: float) -> float:
    """
    高斯累积分布函数（CDF），是 PDF 在 (-∞, t] 上的积分。

    在本项目里，它被用作“把感知空间 h(t) 映回 [0, 1]”的桥梁：

    - 先把比例刻度 t 通过 h(t) 映到实轴；
    - 再用 CDF(h(t), mu) 得到一个位于 [0, 1] 的隶属度值。

    :param t: 实轴上的输入（通常是 h(t0) 之类的中间量）。
    :param mu: 位置参数，表示“形状的中心”。
    :return: CDF(t; mu, sigma=1) ∈ [0, 1]。
    """
    return norm.cdf(t, loc=mu, scale=1.0)

