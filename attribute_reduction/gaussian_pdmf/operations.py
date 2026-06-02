"""
operations
==========

高斯-PDMF 模糊数之间的代数运算，实现论文定义 3.1 中的若干运算：

- 加法     `⊕`
- 减法     `⊖`
- 乘法     `⊗`
- 标量乘法 `λ ⊗ A`

每个运算都接收一个或多个 `GaussianPDMF` 作为输入，返回一个新的 `GaussianPDMF`，
不修改原有对象，从而保持纯函数与不可变风格。
"""

import numpy as np

from .models import GaussianPDMF


def fuzzy_add(a: GaussianPDMF, b: GaussianPDMF) -> GaussianPDMF:
    """
    模糊数加法 `a ⊕ b`（论文定义 3.1(1)）。

    这里采用一种简单的参数合成规则：

    - 中心值：直接相加，x0 = x0_a + x0_b；
    - 支撑长度：相乘，d- = d-_a · d-_b，d+ = d+_a · d+_b；
    - 形状参数：相加，mu- = mu-_a + mu-_b，mu+ = mu+_a + mu+_b。

    具体形式可以依据实际建模需求调整，本实现保持与论文中示意一致。
    """
    x0 = a.x0 + b.x0
    d_minus = a.d_minus * b.d_minus
    d_plus = a.d_plus * b.d_plus
    mu_minus = a.mu_minus + b.mu_minus
    mu_plus = a.mu_plus + b.mu_plus
    return GaussianPDMF(x0, d_minus, d_plus, mu_minus, mu_plus)


def fuzzy_subtract(a: GaussianPDMF, b: GaussianPDMF) -> GaussianPDMF:
    """
    模糊数减法 `a ⊖ b`（论文定义 3.1(3)）。

    直观理解：在中心值和形状参数上做“相减”，
    在支撑长度上做“与乘法相反”的操作（此处实现为相除）。
    """
    x0 = a.x0 - b.x0
    d_minus = a.d_minus / b.d_minus
    d_plus = a.d_plus / b.d_plus
    mu_minus = a.mu_minus - b.mu_minus
    mu_plus = a.mu_plus - b.mu_plus
    return GaussianPDMF(x0, d_minus, d_plus, mu_minus, mu_plus)


def fuzzy_multiply(a: GaussianPDMF, b: GaussianPDMF) -> GaussianPDMF:
    """
    模糊数乘法 `a ⊗ b`（论文定义 3.1(4)）。

    示例实现采用：

    - 中心值：普通实数乘法 x0 = x0_a · x0_b；
    - 支撑长度：对数幂形式 d- = (d-_a)^{ln(d-_b)}，d+ 同理；
    - 形状参数：普通实数乘法 mu- = mu-_a · mu-_b，mu+ 同理。

    该形式体现出“尺度与形状的耦合”，但在数值上要求 d-_a、d-_b、d+_a、d+_b > 0。
    """
    x0 = a.x0 * b.x0
    d_minus = a.d_minus ** np.log(b.d_minus)
    d_plus = a.d_plus ** np.log(b.d_plus)
    mu_minus = a.mu_minus * b.mu_minus
    mu_plus = a.mu_plus * b.mu_plus
    return GaussianPDMF(x0, d_minus, d_plus, mu_minus, mu_plus)


def fuzzy_scalar_multiply(lambda_: float, fuzzy_num: GaussianPDMF) -> GaussianPDMF:
    """
    标量与模糊数的乘法 `λ ⊗ A`（论文定义 3.1(2)）。

    这里将 λ 同时作用在“位置”和“形状”两个层面：

    - 中心值：x0' = λ · x0；
    - 支撑长度：d-' = (d-)^λ，d+' = (d+)^λ；
    - 形状参数：mu-' = λ · mu-，mu+' = λ · mu+。

    当 |λ| > 1 时，模糊数在某种意义上被“放大”；当 0 < |λ| < 1 时则被“压缩”。
    """
    x0 = lambda_ * fuzzy_num.x0
    d_minus = fuzzy_num.d_minus ** lambda_
    d_plus = fuzzy_num.d_plus ** lambda_
    mu_minus = lambda_ * fuzzy_num.mu_minus
    mu_plus = lambda_ * fuzzy_num.mu_plus
    return GaussianPDMF(x0, d_minus, d_plus, mu_minus, mu_plus)

