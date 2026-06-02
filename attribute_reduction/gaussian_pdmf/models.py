class GaussianPDMF:
    """
    高斯-PDMF 模糊数的数据结构，对应论文中的五元组：

        <x0; d-, d+, mu-, mu+>

    - `x0`      ：模糊数的“中心值”，在该点隶属度为 1；
    - `d_minus` ：左支撑长度，控制左侧支撑区间 [x0 - d-, x0] 的范围；
    - `d_plus`  ：右支撑长度，控制右侧支撑区间 [x0, x0 + d+] 的范围；
    - `mu_minus`：左侧高斯形状参数（位置），影响左侧隶属度曲线的弯曲方式；
    - `mu_plus` ：右侧高斯形状参数（位置），影响右侧隶属度曲线的弯曲方式。

    这个类本身不包含任何计算逻辑，只是一个轻量的不可变数据容器：
    所有运算（构造、隶属度计算、加减乘等）都通过独立函数完成，
    以保持函数式风格和更清晰的数学对应关系。
    """

    def __init__(self, x0: float, d_minus: float, d_plus: float, mu_minus: float, mu_plus: float):
        """
        :param x0: 模糊数中心值。
        :param d_minus: 左支撑长度，必须严格大于 0。
        :param d_plus: 右支撑长度，必须严格大于 0。
        :param mu_minus: 左侧形状参数。
        :param mu_plus: 右侧形状参数。
        :raises ValueError: 当 d_minus 或 d_plus 非正时抛出，防止退化为点或反向区间。
        """
        if d_minus <= 0 or d_plus <= 0:
            raise ValueError("支撑长度 d- 和 d+ 必须大于 0")
        self.x0 = x0
        self.d_minus = d_minus
        self.d_plus = d_plus
        self.mu_minus = mu_minus
        self.mu_plus = mu_plus

    def __repr__(self) -> str:
        """
        便于调试与打印，在示例和交互式环境中快速查看模糊数主要参数。
        """
        return (
            f"GaussianPDMF(x0={self.x0:.2f}, "
            f"d-={self.d_minus:.2f}, d+={self.d_plus:.2f}, "
            f"mu-={self.mu_minus:.4f}, mu+={self.mu_plus:.4f})"
        )

