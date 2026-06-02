# 属性约简算法使用与选优说明（结合原项目）

本文档说明 `@frs/attribute_reduction` 中各算法：
- 是否有参数
- 参数是否在算法层遍历
- 原项目如何找到“最优约简子集”
- 现在在 `@frs` 中如何调用

---

## 一、原项目的统一选优思路

原项目（主要在 notebook）整体是两层流程：

1. **约简层**：给定一组参数，得到一个或一批约简子集  
2. **评估层**：把候选约简子集送入分类器（通常多模型）评估准确率  
3. **选优**：取准确率最高的子集，并记录其对应参数组合

也就是说，很多“最优参数”不是算法函数内部自动搜索得到，而是实验层外部遍历后比较得到。

---

## 二、各算法说明

## `MFREN`

- **当前入口**：`reduce_mfren(columns, metric_fsr, fsr_obj, lambda_v=0.0)`
- **关键参数**：
  - `lambda_v`（默认 `0.0`）：固定
  - 数据预处理相关参数在 `prepare_mfren_fold(...)`：`scaler`、`columns_nominal`
- **算法层是否参数遍历**：否（单次运行）
- **原项目是否做外层遍历**：是  
  常见外层遍历为 `Scaler in [1,2,3]`、`prime_columns_nominal in [True,False]`，然后做交叉验证分类评估选优。
- **如何找最优子集**：外层比较分类准确率后选出对应约简子集。

---

## `FSFrMI`

- **当前入口**：`reduce_fsfrmi(columns, metric_fsr, fsr_obj, lambda_v=0.0)`
- **关键参数**：
  - `lambda_v`（默认 `0.0`）：固定
  - `prepare_fsfrmi_fold(...)` 中的 `scaler`、`columns_nominal`
- **算法层是否参数遍历**：否（单次运行）
- **原项目是否做外层遍历**：是（同 MFREN 的实验层策略）
- **如何找最优子集**：外层按分类准确率选优。

---

## `FNRS`

- **当前入口**：`reduce_fnrs(data, train_index, target_name="target")`
- **关键参数**（在算法内部固定网格遍历）：
  - `lamda`: `0.1 ~ 0.5`，步长 `0.05`
  - `alpha`: `0.5 ~ 1.0`，步长 `0.05`
- **算法层是否参数遍历**：是（内置网格遍历）
- **当前返回**：一批候选约简子集（按长度排序后返回）
- **原项目如何选最优**：
  1. 先拿到这批 `(lamda, alpha)` 对应的候选子集
  2. 在外层跑分类准确率
  3. 选准确率最高的子集，并得到对应 `(lamda, alpha)`

---

## `FRAR`

- **当前入口**：`reduce_frar(data, train_index, target_name="target", class_counts=None)`
- **关键参数**：
  - `class_counts`（影响使用哪些隶属函数）
  - 内部为贪心增量，不做显式网格参数遍历
- **算法层是否参数遍历**：否（无显式网格）
- **原项目是否做外层遍历**：通常会与数据预处理策略一起在外层做实验对比
- **如何找最优子集**：若存在多候选实验配置，仍是外层按分类准确率选优。

---

## `IARFCIE`

- **当前入口**：`reduce_iarfcie(data, target_name="target", threshold=0.2)`
- **关键参数**：
  - `threshold`（原代码常写 `throd`）
- **算法层是否参数遍历**：否（单次运行）
- **原项目是否做外层遍历**：是  
  在原 notebook 里常见 `for throd in ...` 的参数扫描。
- **如何找最优子集**：外层对不同 `threshold` 结果做分类评估，选准确率最高者。

---

## `MFIGI`

- **当前入口**：`reduce_mfigi(data, target_name="target", w=0.005)`
- **关键参数**：
  - `w`（属性加入阈值）
- **算法层是否参数遍历**：否（单次运行）
- **原项目是否做外层遍历**：以“按数据集手工设不同 `w`”为主（如 `0.005`、`0.002`），可视作外层调参。
- **如何找最优子集**：外层比较不同 `w` 下的分类准确率选优。

---

## 三、`@frs` 当前脚本与“选优”关系

`frs/run_all_reductions_demo.py` 目前是**单次演示脚本**，作用是：
- 跑通算法调用
- 输出每个算法本次参数下的约简结果

它**不是完整选优脚本**，目前没有：
- 对 `IARFCIE.threshold`、`MFIGI.w` 做参数网格搜索
- 对候选子集跑分类器并按准确率自动选最优

---

## 四、建议的标准选优流程（你后续可据此实现）

1. 设定每个算法参数候选集合  
2. 对每个参数组合运行约简，得到候选子集  
3. 对每个候选子集做统一分类评估（同一折分、同一模型集）  
4. 记录 `accuracy` 与 `subset_size`  
5. 以 `accuracy` 最大为主、`subset_size` 最小为辅选最优  
6. 输出：最优参数、最优子集、最优准确率

---

## 五、接口速查

- `MFREN`：`prepare_mfren_fold(...)` + `reduce_mfren(...)`
- `FSFrMI`：`prepare_fsfrmi_fold(...)` + `reduce_fsfrmi(...)`
- `FNRS`：`reduce_fnrs(...)`
- `FRAR`：`reduce_frar(...)`
- `IARFCIE`：`reduce_iarfcie(...)`
- `MFIGI`：`reduce_mfigi(...)`

导出位置：`frs/attribute_reduction/__init__.py`


# 缩放规则
1. ARPDMF：约简用的是 train_df_raw（外层 MinMaxScaler 之前的本折训练特征 + 目标）。
2. MFNMI除了走外层 MinMax ，之后还可以走123：1 不额外缩放、2 MinMax、3 Standard（默认 None 时等价 3）
  2.1 选 1 =「MFNMI 这一步不再叠一层缩放」，并不等于「原始未缩放数据」；上一步外层的 MinMax 仍然已经作用在数据上了
3. 其余算法只走MinMax 


# 模糊数归一化
ARPDMF 的流程是：
用当前折、原始尺度的 train_df_raw 做约简输入；
按列把实数值模糊化成一批高斯 PDMF；
对每一列上全体样本的这批模糊数做 normalization.md 定义 2 的批归一化（fuzzy_batch_normalize=True，默认）；
在归一化后的模糊数上算样本模糊熵，再按 ARPDMF 三阶段约简。
## 1. 「一批模糊数」指什么？
**1. 固定某一列 \(j\)** 时，**该列上所有样本**  
\(\{x_{1j}, x_{2j}, \ldots, x_{nj}\}\)  
各自对应一个模糊数，一共 **\(n\) 个**模糊数，这一列上的 **这 \(n\) 个**为「一批」，用来算 \(m,M,M_d\) 并做定义 2。
**对每个属性列 \(j\)，有一批（\(n\) 个）样本模糊数**，对应每个样本 \(i\) 一个。
## 2–3. 模糊化与按列归一化
- **2.** 对第 \(j\) 列，用整列 \(\{x_{ij}\}_{i=1}^n\)（以及 \(k,\mu\) 等）调用 `create_gaussian_pdmf_from_column`，得到 **\(n\) 个**高斯 PDMF，第 \(i\) 个对应样本 \(i\) 在该列上的模糊数，可记为 \(\tilde{x}_{ij}\)。
- **3.** 对**同一列 \(j\)** 的这 \(n\) 个模糊数调用 `normalize_gaussian_pdmf_batch_definition2`，得到 **\(n\) 个**归一化模糊数 \(\tilde{n}_{ij}\)（实现里仍是 `GaussianPDMF`）。
## 4. 约简
- 对每个 \(\tilde{n}_{ij}\) 算样本模糊熵，得到矩阵 \(E_{ij}\)（形状 \(n \times |\text{条件属性}|\)），再在 **`arpdmf` 里用 \(E\)** 做三阶段约简。
## 和「每个 \(x_{ij}\)」这句话怎么对应？
- **数据层面**：每个 **单元格** \(x_{ij}\) 参与构造 **一个** \(\tilde{x}_{ij}\)（列内第 \(i\) 个模糊数）。  
- **归一化层面**：**不能**只对单独一个 \(\tilde{x}_{ij}\) 做定义 2；必须对**该列全部** \(\{\tilde{x}_{ij}\}_{i=1}^n\) 一起归一化，才出现 \(m,M,M_d\)。  
所以你写的「每个模糊数按列归一化」在实现上的准确含义是：**对每个 \(j\)，对 \(\{\tilde{x}_{ij}\}_{i=1}^n\) 这一批一起做归一化，得到 \(\{\tilde{n}_{ij}\}_{i=1}^n\)**。
**结论：是的，当前实现就是这样（且 `fuzzy_batch_normalize=True` 时生效）。**