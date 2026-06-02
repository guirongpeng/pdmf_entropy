"""
membership
==========

给定一个高斯-PDMF 模糊数和论域中的点 x，计算其隶属度 μ(x)。

实现严格对应论文中的公式 2.3–2.4：

- 支撑区间之外：μ(x) = 0；
- 中心点 x0：   μ(x0) = 1；
- 左右支撑内部：先把 x 归一化为区间上的比例刻度 t，再通过 h(t) 和高斯 CDF 得到 μ(x)。
"""

from .core import subjective_perception_h, gaussian_cdf
from .models import GaussianPDMF


def calculate_membership(fuzzy_num: GaussianPDMF, x: float) -> float:
    """
    计算给定点 x 的隶属度 μ(x)。

    整体分为四种情况：

    1. `x <= x0 - d-` 或 `x >= x0 + d+`：在支撑区间之外，μ(x) = 0；
    2. `x == x0`：中心点，定义为 μ(x0) = 1；
    3. `x0 - d- < x < x0`：位于左支撑内部，
       - 先将 x 线性映射到比例刻度 t ∈ (0, 1)；
       - 再计算 h(t)，并代入左侧形状参数 mu- 的高斯 CDF 得到 μ(x)；
    4. `x0 < x < x0 + d+`：右支撑内部，过程与左侧类似，但使用 mu+。

    :param fuzzy_num: 要评估的高斯-PDMF 模糊数。
    :param x: 实数轴上的任一点。
    :return: 隶属度值 μ(x)，位于 [0, 1]。
    """
    x0 = fuzzy_num.x0
    d_minus = fuzzy_num.d_minus
    d_plus = fuzzy_num.d_plus
    mu_minus = fuzzy_num.mu_minus
    mu_plus = fuzzy_num.mu_plus

    # 支撑区间外：隶属度为 0（对应论文公式 2.3）
    if x <= x0 - d_minus or x >= x0 + d_plus:
        return 0.0
    # 精确中心点：定义为 1
    elif x == x0:
        return 1.0
    # 左支撑内部：先做线性归一化，再通过 h 和高斯 CDF 得到 μ
    elif x0 - d_minus < x < x0:
        t = (x - (x0 - d_minus)) / d_minus
        h_t = subjective_perception_h(t)
        return gaussian_cdf(h_t, mu_minus)
    # 右支撑内部：与左侧类似，只是归一化方向相反，使用 mu_plus
    else:
        t = ((x0 + d_plus) - x) / d_plus
        h_t = subjective_perception_h(t)
        return gaussian_cdf(h_t, mu_plus)

