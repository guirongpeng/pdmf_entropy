"""FNRS attribute reduction module."""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd


def gaussian(x):
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    for column in range(len(means)):
        # Avoid division by zero on constant-valued columns.
        denom = 2 * np.power(stds[column], 2)
        if denom == 0:
            x[:, column] = 1.0
            continue
        x[:, column] = np.exp(-np.power(x[:, column] - means[column], 2) / denom)
    return x


@nb.jit(nopython=True)
def get_fsr_all_ai(array, m, n):
    fsr_all_ai = np.zeros((m, n, n))
    for k in range(m):
        for i in range(n):
            for j in range(i + 1, n):
                value = min(array[:, k][i], array[:, k][j])
                fsr_all_ai[k][i][j] = value
                fsr_all_ai[k][j][i] = value
    return fsr_all_ai


@nb.jit(nopython=True)
def fuzzy_relation_matrix_a(array, n):
    fsr_a = np.zeros((n, n))
    for i in range(n):
        fsr_a[i, i] = 1
        for j in range(i + 1, n):
            fsr_a[i, j] = np.minimum(array[i], array[j]).min()
            fsr_a[j, i] = fsr_a[i, j]
    return fsr_a


def fuzzy_decisions(fsr_a, true_class, n, u):
    fuzzy_class = np.zeros((len(set(u)), n))
    for di in range(fuzzy_class.shape[0]):
        for xi in range(n):
            fuzzy_class[di][xi] = sum(np.take(fsr_a[xi], true_class[di])) / sum(fsr_a[xi])
    return fuzzy_class


@nb.jit(nopython=True)
def fuzzy_relation_matrix_ai(fsr_all_ai, n, fsr_a, lamda):
    fsr_part_ai = np.zeros((n, n))
    for i in range(n):
        fsr_part_ai[i, i] = 1
        for j in range(i + 1, n):
            min_values = np.min(fsr_all_ai[:, i, j])
            if min_values >= lamda:
                fsr_part_ai[i, j] = fsr_a[i, j]
                fsr_part_ai[j, i] = fsr_a[i, j]
    return fsr_part_ai


def fuzzy_decisions_ai(fsr_ai, fuzzy_class_a, true_class, alpha, n):
    pos_ai_la = 0
    for di_index, di in enumerate(true_class):
        values = np.sum(fsr_ai[di] <= fuzzy_class_a[di_index], axis=1) / n
        indexes = np.where(values >= alpha)[0]
        pos_ai_la += len(indexes)
    return pos_ai_la / n


def reduce_fnrs(data, train_index, target_name="target"):
    fold = data.iloc[train_index].copy()
    fold.index = list(range(fold.shape[0]))
    u = fold[target_name].values
    x = np.array(fold.drop(target_name, axis=1))
    x = gaussian(x)
    n, m = x.shape

    results = []
    # Keep class index groups as a list of variable-length arrays.
    true_class = [np.where(u == i)[0] for i in np.unique(u)]
    fsr_all_ai = get_fsr_all_ai(x, m, n)
    fsr_a = fuzzy_relation_matrix_a(x, n)
    fuzzy_class_a = fuzzy_decisions(fsr_a, true_class, n, u)

    for lamda in np.arange(0.1, 0.55, 0.05):
        for alpha in np.arange(0.5, 1.05, 0.05):
            r = []
            b = [i for i in range(m) if i not in r]
            dependency_r = 0
            start = True
            while start:
                dependencies = np.zeros((m))
                for ai in b:
                    r.append(ai)
                    fsr_part_ai = fuzzy_relation_matrix_ai(fsr_all_ai[r, :, :], n, fsr_a, lamda=lamda)
                    dependency_ai_la = fuzzy_decisions_ai(fsr_part_ai, fuzzy_class_a, true_class, alpha, n)
                    dependencies[ai] = dependency_ai_la
                    r.remove(ai)
                max_index = np.argmax(dependencies)
                if dependencies[max_index] > dependency_r:
                    dependency_r = dependencies[max_index]
                    r.append(max_index)
                    b.remove(max_index)
                else:
                    start = False
            results.append((lamda, alpha, r))

    results = sorted(results, key=lambda x: len(x[2]), reverse=True)
    return [item[-1] for item in results if len(item[-1]) >= 1]

'''
FNRS 在算法层先做：
    参数网格遍历（lamda × alpha）
    每组得到一个约简子集 R
    汇总成一批候选子集（你现在代码里就是 results.append((lamda, alpha, r))）
然后在原项目实验层再做：
    对这些候选子集逐个分类评估
    找到准确率最优的那个子集
    同时拿到它对应的参数组合 (lamda, alpha)。
'''


def generate_fnrs_candidates(
    train_df: pd.DataFrame,
    target_name: str,
    lamdas,
    alphas,
):
    """
    基于当前折训练集，遍历 lamda × alpha 网格，按依赖度贪心生成候选子集。
    返回列表元素形式：(列名子集, 参数dict)。
    """
    all_idx = list(range(train_df.shape[0]))
    fold = train_df.iloc[all_idx].copy()
    u = fold[target_name].values
    x = np.array(fold.drop(target_name, axis=1))
    x = gaussian(x)
    n, m = x.shape
    columns = list(fold.drop(target_name, axis=1).columns)

    true_class = [np.where(u == i)[0] for i in np.unique(u)]
    fsr_all_ai = get_fsr_all_ai(x, m, n)
    fsr_a = fuzzy_relation_matrix_a(x, n)
    fuzzy_class_a = fuzzy_decisions(fsr_a, true_class, n, u)

    out = []
    for lamda in lamdas:
        for alpha in alphas:
            r = []
            b = [i for i in range(m)]
            dependency_r = 0.0
            start = True
            while start:
                dependencies = np.zeros((m))
                for ai in b:
                    r.append(ai)
                    fsr_part_ai = fuzzy_relation_matrix_ai(fsr_all_ai[r, :, :], n, fsr_a, lamda=lamda)
                    dep = fuzzy_decisions_ai(fsr_part_ai, fuzzy_class_a, true_class, alpha, n)
                    dependencies[ai] = dep
                    r.remove(ai)
                max_index = int(np.argmax(dependencies))
                if dependencies[max_index] > dependency_r:
                    dependency_r = float(dependencies[max_index])
                    r.append(max_index)
                    b.remove(max_index)
                else:
                    start = False
            if r:
                out.append(([columns[i] for i in r], {"lamda": lamda, "alpha": alpha}))
    return out