"""MFIGI attribute reduction module."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MFIGI:
    def __init__(self, df: pd.DataFrame, w=0.005, target_name="target"):
        self.df = df
        self.w = w
        self.target_name = target_name

        self.c = [col for col in df.columns if col != target_name]
        self.n = df.shape[0]

        cond_df = df[self.c]
        target_df = df[[target_name]]
        self.p = self.calculate_distances(cond_df)
        self.matrix_d = self.calculate_distances(target_df)[0]

        self.init_figi = self.figi(self.c)
        self.red = []

    @staticmethod
    def calculate_distances(df: pd.DataFrame):
        n = df.shape[0]
        num_features = df.shape[1]
        result = np.zeros((num_features, n, n))

        for index, column in enumerate(df.columns):
            col_values = df[column].values.astype(float)
            col_min = col_values.min()
            col_max = col_values.max()
            denominator = col_max - col_min
            if denominator == 0:
                norm_values = np.zeros_like(col_values)
            else:
                norm_values = (col_values - col_min) / denominator

            std = norm_values.std()
            diff_matrix = np.abs(norm_values[:, np.newaxis] - norm_values)
            distance_matrix = np.maximum(0, std - diff_matrix) * std
            result[index] = distance_matrix
        return result

    def figi(self, sub_columns):
        if not sub_columns:
            return float("inf")
        indices = [self.c.index(col) for col in sub_columns]
        sub_p = np.max(self.p[indices], axis=0)
        max_every = np.maximum(sub_p, self.matrix_d)
        return -(max_every.sum() - self.matrix_d.sum()) / (self.n**2)

    def sig(self, red, attr):
        return self.figi(red) - self.figi(red + [attr])

    def run_main(self):
        candidates = [a for a in self.c if a not in self.red]
        while self.figi(self.red) != self.init_figi:
            max_figi = float("-inf")
            best_candidate = None
            for candidate in candidates:
                curr_figi = self.sig(self.red, candidate)
                if curr_figi > max_figi:
                    max_figi = curr_figi
                    best_candidate = candidate
            if max_figi > self.w and best_candidate is not None:
                self.red.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
        return self.red


def reduce_mfigi(data: pd.DataFrame, target_name="target", w=0.005):
    """Run MFIGI and return reduced feature names."""
    model = MFIGI(data, w=w, target_name=target_name)
    return model.run_main()

