"""FRAR attribute reduction module."""

from __future__ import annotations

import copy
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler


def triangular_transform_x(x):
    x = np.array(x)
    result = np.zeros_like(x)
    result[x < -0.5] = 0
    result[x > 0.5] = 0
    result[(x < 0) & (x >= -0.5)] = 2 * x[(x < 0) & (x >= -0.5)] + 1
    result[(x >= 0) & (x <= 0.5)] = -2 * x[(x >= 0) & (x <= 0.5)] + 1
    return result


def trapezoidal_transform_x(x):
    x = np.array(x)
    result = np.zeros_like(x)
    result[x >= 0] = 0
    result[x <= -0.5] = 1
    result[(x > -0.5) & (x < 0)] = -2 * x[(x > -0.5) & (x < 0)]
    return result


def sigmoid(x):
    a = -0.5
    x = np.array(x)
    means = x.mean(axis=0)
    for column in range(len(means)):
        x[:, column] = 1 / (1 + np.exp(a * (x[:, column] - means[column])))
    return x


def reverse_sigmoid(x):
    a = 0.5
    x = np.array(x)
    means = x.mean(axis=0)
    for column in range(len(means)):
        x[:, column] = 1 / (1 + np.exp(a * (x[:, column] - means[column])))
    return x


def sawtooth(x):
    k = 1
    x = np.array(x)
    for column in range(x.shape[1]):
        values = x[:, column]
        x[:, column] = np.piecewise(
            values,
            [values < 0, (0 <= values) & (values < 1 / k), (1 / k <= values) & (values < 2 / k)],
            [0, lambda v: k * v, lambda v: k * (1 - v)],
        )
    return x


def infmax(mu_f_values, di, _n):
    mu_f_values = np.array(mu_f_values).squeeze()
    di = np.array(di)
    values = np.maximum(1 - mu_f_values, di)
    return values.min()


def mu_f(combined_matrix, p, k):
    combined_matrix = combined_matrix[:, :, p]
    if len(p) == 1:
        return combined_matrix
    mu_f_values = []
    iterables = [range(0, k) for _ in range(len(p))]
    combinations = itertools.product(*iterables)
    for combination in combinations:
        values = []
        for column, i in enumerate(combination):
            values.append(combined_matrix[i][:, column])
        mu_f_values.append(np.min(values, axis=0))
    return np.array(mu_f_values)


def get_dependency(p, combined_matrix, n, di_s, k):
    if len(p) == 0:
        return 0
    mu_f_values = mu_f(combined_matrix, p, k)
    infmax_values = np.array([[infmax(mu_f_values[f], di, n) for f in range(mu_f_values.shape[0])] for di in di_s])
    temp_dependency = 0
    for x in range(n):
        mu_f_values_reshaped = mu_f_values[:, x].reshape(1, -1)
        min_values = np.minimum(mu_f_values_reshaped, infmax_values)
        max_values = np.max(min_values, axis=1)
        temp_dependency += np.max(max_values)
    return temp_dependency / n


def reduce_frar(data, train_index, target_name="target", class_counts=None):
    fold = data.iloc[train_index].copy()
    u = fold[target_name].values
    x = fold.drop(target_name, axis=1)
    x.index = list(range(x.shape[0]))
    array = StandardScaler().fit_transform(x)

    x1 = trapezoidal_transform_x(array)
    x2 = triangular_transform_x(array)
    matrix_list = [x1, x2]
    if class_counts is None:
        class_counts = len(set(u))
    if class_counts >= 3:
        matrix_list.append(reverse_sigmoid(array))
    if class_counts >= 4:
        matrix_list.append(sawtooth(array))
    if class_counts >= 5:
        matrix_list.append(sigmoid(array))

    combined_matrix = np.stack(matrix_list, axis=0)
    di_s = []
    classes = tuple([np.where(u == i)[0] for i in set(u)])
    for class_i in classes:
        mask = np.zeros(len(u), dtype=bool)
        mask[class_i] = True
        di_s.append(mask)

    n, m = array.shape
    k = len(set(u))
    r = []
    epoch_dependency = 0
    temp_dependency = -1
    all_columns = set(range(m))

    while epoch_dependency != temp_dependency:
        t = copy.deepcopy(r)
        epoch_dependency = get_dependency(t, combined_matrix, n, di_s, k)
        temp_dependency = temp_dependency
        for col in [i for i in all_columns if i not in r]:
            r.append(col)
            current = get_dependency(r, combined_matrix, n, di_s, k)
            if current > temp_dependency:
                t = copy.deepcopy(r)
                temp_dependency = current
            r.remove(col)
        r = copy.deepcopy(t)
    return [r]

