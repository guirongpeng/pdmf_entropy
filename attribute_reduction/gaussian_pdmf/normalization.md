## B. 归一化

处理有限模糊数集合时，有用的预处理步骤是归一化（参见），它将主导因子和支撑参数置于可比较的尺度上，并便于后续计算。 [crad.ict.ac](https://crad.ict.ac.cn/fileJSJYJYFZ/journal/article/jsjyjyfz/HTML/2019-12-2578.shtml)

给定索引集 $I = \{1,2,\dots,N\}$，我们有

**定义 2.** 设 $\{\tilde{x}_i\}_{i\in I}$ 是 $X$ 中的一组 fuzzy numbers。其归一化表示为 $\{\tilde{n}_i\}_{i\in I}$，通过以下变换得到：
$$
\tilde{n}_i = \left\langle \frac{x_i - m}{M - m}, e^{d_i^- / M_d}, e^{d_i^+ / M_d}, \mu_i^-, \mu_i^+ \right\rangle, \tag{3}
$$
其中 $m = \min_{i\in I} x_i$，$M = \max_{i\in I} x_i$ 且 $M_d = \max_{d_i^- \in I \cup d_i^+ \in I} d$。
