"""
constructors
============

从“控制点 + 比例刻度”的直观输入，构造出高斯-PDMF 模糊数：

- 控制点 P = (Px, Py)、Q = (Qx, Qy) 给出两侧某一具体 x 位置的隶属度；
- 比例刻度 t-、t+ 则决定“控制点在各自支撑区间内所处的相对位置”。

通过这些量，可以反推出支撑长度 d- / d+，以及高斯形状参数 mu- / mu+，
最终得到一个 `GaussianPDMF` 实例。
"""

from typing import Tuple

from .core import subjective_perception_h
from .models import GaussianPDMF
from scipy.stats import norm


def calculate_support_length(x_ctrl: float, x0: float, t: float) -> float:
    """
    根据控制点位置和比例刻度计算支撑长度 d- 或 d+（对应论文定理 2.1）。

    思想：将控制点视为支撑区间上的一个内点，其相对位置用比例刻度 t 描述，
    例如在左侧有：

        Px = x0 - (1 - t) * d-

    对其变形即可解出：

        d- = (x0 - Px) / (1 - t)

    右侧类似，只是方向相反。

    :param x_ctrl: 控制点的 x 坐标（Px 或 Qx）。
    :param x0: 模糊数中心值。
    :param t: 对应控制点的比例刻度，满足 0 < t < 1。
    :return: 左/右支撑长度 d- 或 d+，始终为正数。
    """
    if x_ctrl < x0:  # 左控制点 → 计算 d-
        return (x0 - x_ctrl) / (1 - t)
    else:  # 右控制点 → 计算 d+
        return (x_ctrl - x0) / (1 - t)


def solve_mu(h_t: float, y_ctrl: float) -> float:
    """
    根据感知值 h(t) 与控制点隶属度 y 反解高斯形状参数 mu（论文公式 2.4 的逆运算）。

    设：
        y = Phi(h(t); mu, sigma=1)
    其中 Phi 是标准高斯 CDF。记 z = Phi^{-1}(y) 为标准正态分位数，则有：

        h(t) = mu + z  ⇒  mu = h(t) - z

    :param h_t: 比例刻度 t 通过 h 函数后的值 h(t)。
    :param y_ctrl: 控制点的隶属度 y，必须满足 0 < y < 1。
    :return: 对应的高斯位置参数 mu。
    """
    return h_t - norm.ppf(y_ctrl)


def create_gaussian_pdmf(
    x0: float,
    p: Tuple[float, float],
    q: Tuple[float, float],
    t_minus: float,
    t_plus: float,
) -> GaussianPDMF:
    """
    从直观输入构造一个高斯-PDMF 模糊数（完整流程）。

    参数含义与典型约束：

    - `x0`      ：中心值，即“最典型”的取值，隶属度为 1；
    - `p=(Px,Py)`：左控制点，要求 Px < x0 且 0 < Py < 1；
    - `q=(Qx,Qy)`：右控制点，要求 Qx > x0 且 0 < Qy < 1；
    - `t_minus` ：左侧比例刻度，0 < t- < 1，决定 P 在 [x0 - d-, x0] 上的相对位置；
    - `t_plus`  ：右侧比例刻度，0 < t+ < 1，决定 Q 在 [x0, x0 + d+] 上的相对位置。

    构造步骤：

    1. 校验上述约束，防止产生退化或不合理的模糊数；
    2. 使用 `calculate_support_length` 反推支撑长度 d-、d+；
    3. 使用 `subjective_perception_h` 和 `solve_mu` 反推 mu-、mu+；
    4. 封装为 `GaussianPDMF` 返回。
    """
    px, py = p
    qx, qy = q
    if not (px < x0 < qx):
        raise ValueError("控制点位置约束：需要满足 Px < x0 < Qx")
    if not (0 < py < 1 and 0 < qy < 1):
        raise ValueError("控制点隶属度约束：Py 和 Qy 必须严格位于 (0, 1) 内")
    if not (0 < t_minus < 1 and 0 < t_plus < 1):
        raise ValueError("比例刻度约束：t- 和 t+ 必须严格位于 (0, 1) 内")

    # 由控制点坐标与比例刻度反推左右支撑长度
    d_minus = calculate_support_length(px, x0, t_minus)
    d_plus = calculate_support_length(qx, x0, t_plus)

    # 由感知映射值与控制点隶属度反推左右形状参数 mu
    h_t_minus = subjective_perception_h(t_minus)
    h_t_plus = subjective_perception_h(t_plus)
    mu_minus = solve_mu(h_t_minus, py)
    mu_plus = solve_mu(h_t_plus, qy)

    return GaussianPDMF(x0, d_minus, d_plus, mu_minus, mu_plus)

