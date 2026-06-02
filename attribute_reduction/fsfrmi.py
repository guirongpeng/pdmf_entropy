"""FSFrMI attribute reduction module."""

from __future__ import annotations

import copy
import numba as nb
import numpy as np

from .preprocessing import split_fold_array


class FSFMyFSR:
    def __init__(self, my_array):
        self.R_D = my_array[:, -1]
        self.my_array = my_array[:, :-1]
        self.n, self.m = self.my_array.shape

    @staticmethod
    @nb.jit(nopython=True)
    def fsf_metric_fsr(my_array, m, n, columns_nominal=np.array([])):
        fsr = np.zeros((m, n, n))
        for k in range(m):
            if k in columns_nominal:
                for i in range(n):
                    for j in range(i + 1, n):
                        fsr[k][i][j] = 0 if my_array[:, k][i] == my_array[:, k][j] else 1
                        fsr[k][j][i] = fsr[k][i][j]
            else:
                for i in range(n):
                    for j in range(i + 1, n):
                        value = 1 - abs(my_array[:, k][i] - my_array[:, k][j])
                        fsr[k][i][j] = value
                        fsr[k][j][i] = value
        return fsr

    @staticmethod
    @nb.jit(nopython=True)
    def fsf_r_d_expanded(r_d):
        n = len(r_d)
        expanded = np.zeros((n, n))
        for i in range(n):
            idx = np.where(r_d == r_d[i])
            expanded[i][idx] = 1
        return expanded

    def calculate_metric_fsr(self, columns_nominal=np.array([])):
        self.metric_fsr_value = self.fsf_metric_fsr(self.my_array, self.m, self.n, columns_nominal)
        return self.metric_fsr_value

    def calculate_r_d(self):
        self.R_D = self.fsf_r_d_expanded(self.R_D)
        return self.R_D


class FSFMyEntropy:
    def __init__(self, columns_list, fsr, r_d):
        self.columns_list = columns_list
        self.fsr = fsr
        self.r_d = r_d
        self.mu_in_cd = self.mutual_info(columns_list)

    def sig_in(self, column, columns):
        return -(self.mu_in_cd - self.mutual_info([col for col in columns if col != column]))

    def sig_out(self, re_column, red, current_red_value=-999):
        new_red = copy.deepcopy(red)
        new_red.append(re_column)
        if current_red_value == -999:
            return -(self.mutual_info(new_red) - self.mutual_info(red))
        return -(self.mutual_info(new_red) - current_red_value)

    def mutual_info(self, f_columns):
        if len(f_columns) == 0:
            return 0
        indices = [self.columns_list.index(col) for col in f_columns]
        sub_fsr = np.min(self.fsr[indices], axis=0)
        c = np.minimum(self.r_d, sub_fsr)
        sum_r_d = np.sum(self.r_d, axis=1)
        sum_sub_fsr = np.sum(sub_fsr, axis=1)
        sum_c = np.sum(c, axis=1)
        # 数值稳定性：防止分子/分母为0导致log2发散
        eps = 1e-12
        ratio = (sum_c + eps) / (sum_r_d * sum_sub_fsr + eps)
        return -np.sum(np.log2(ratio)) / c.shape[0]


def prepare_fold(data, train_index, target_name="target", scaler=1, columns_nominal=None):
    x, xy, columns = split_fold_array(data, train_index, target_name=target_name, scaler=scaler)
    columns_nominal = np.array(columns_nominal or [])
    fsr_obj = FSFMyFSR(xy)
    metric_fsr = fsr_obj.calculate_metric_fsr(columns_nominal)
    fsr_obj.calculate_r_d()
    return x, xy, fsr_obj, metric_fsr, columns


def reduce_fsfrmi(columns, metric_fsr, fsr_obj, lambda_v=0.0):
    zero_red = []
    entropy = FSFMyEntropy(columns, metric_fsr, fsr_obj.R_D)

    for column in columns:
        if entropy.sig_in(column, columns) > lambda_v:
            zero_red.append(column)
    one_red = copy.deepcopy(zero_red)

    init_muinfo = entropy.mu_in_cd
    while entropy.mutual_info(one_red) != init_muinfo:
        remaining = [x for x in columns if x not in one_red]
        current = entropy.mutual_info(one_red)
        out_cols = [(entropy.sig_out(col, one_red, current), col) for col in remaining]
        out_cols = sorted(out_cols, key=lambda x: x[0])
        one_red.append(out_cols[-1][1])
    two_red = copy.deepcopy(one_red)

    init_muinfo = entropy.mutual_info(two_red)
    three_red = copy.deepcopy(two_red)
    for column in two_red:
        candidate = [x for x in three_red if x != column]
        if entropy.mutual_info(candidate) == init_muinfo:
            three_red = candidate
    return three_red

