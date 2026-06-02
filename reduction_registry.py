from __future__ import annotations

import itertools
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from frs.attribute_reduction import (
        prepare_fsfrmi_fold,
        reduce_fsfrmi,
        reduce_arpdmf,
        prepare_arpdmf_entropy_matrix,
        reduce_arpdmf_from_matrix,
        reduce_frar,
        reduce_iarfcie,
        reduce_mfigi,
    )
    from frs.attribute_reduction import fnrs as fnrs_mod
    from frs.attribute_reduction import mfnmi as mfnmi_mod
except ModuleNotFoundError:
    from attribute_reduction import (
        prepare_fsfrmi_fold,
        reduce_fsfrmi,
        reduce_arpdmf,
        prepare_arpdmf_entropy_matrix,
        reduce_arpdmf_from_matrix,
        reduce_frar,
        reduce_iarfcie,
        reduce_mfigi,
    )
    from attribute_reduction import fnrs as fnrs_mod
    from attribute_reduction import mfnmi as mfnmi_mod


def run_reductions_for_fold(
    train_df: pd.DataFrame,
    train_df_raw: pd.DataFrame | None,
    all_train_idx: List[int],
    target_name: str,
    grids: Dict,
    enabled_algos: List[str],
    columns_all: List[str],
    mfnmi_scaler: int | None = None,
) -> Tuple[List[Dict], Dict[str, float], List[float]]:
    """
    在当前折上执行选中的属性约简算法，返回：
    - candidates: 列表，元素为 {算法, 参数, 子集}
    - fold_reduction_times: 本折各算法约简耗时（秒）
    - arpdmf_prepare_times_fold: 本折每次实际执行 prepare 的耗时（秒），供全局取均值后计入 ARPDMF 总耗时一次
    """
    candidates: List[Dict] = []
    fold_reduction_times: Dict[str, float] = {}
    arpdmf_prepare_times_fold: List[float] = []

    # FSFrMI（单候选）
    if "FSFrMI" in enabled_algos:
        t0 = time.perf_counter()
        _, _, fsfrmi_fsr_obj, fsfrmi_metric, cols_fsfrmi = prepare_fsfrmi_fold(
            train_df, all_train_idx, target_name=target_name, columns_nominal=[]
        )
        t_one = time.perf_counter()
        fsfrmi_subset = reduce_fsfrmi(cols_fsfrmi, fsfrmi_metric, fsfrmi_fsr_obj, lambda_v=0.0)
        t_red_one = round(time.perf_counter() - t_one, 6)
        candidates.append(
            {
                "算法": "FSFrMI",
                "参数": {"lambda_v": 0.0},
                "子集": fsfrmi_subset,
                "约简耗时秒": t_red_one,
            }
        )
        fold_reduction_times["FSFrMI"] = round(time.perf_counter() - t0, 6)

    # ARPDMF：按 (k,mu) 每折缓存 E_mat；折内仅对 δ 三阶段约简计入 fold 耗时；prepare 耗时记入 arpdmf_prepare_times_fold
    if "ARPDMF" in enabled_algos:
        # arpdmf_train_df = train_df if train_df_raw is None else train_df_raw # [print_notice]
        arpdmf_train_df = train_df  # arpdmf事前归一化:最新：arpdmf跑归一化

        
        g_arp = grids["ARPDMF"]
        deltas = g_arp.get("delta", [0.1])
        ks = g_arp.get("k", [3])
        mus = g_arp.get("mu_method", ["A"])
        entropy_cache: dict[tuple[int, str], tuple[list[str], np.ndarray]] = {}
        t_reduce_fold = 0.0
        for k_arp, mu_arp in itertools.product(ks, mus):
            km_key = (int(k_arp), str(mu_arp))
            if km_key not in entropy_cache:
                t_prep = time.perf_counter()
                cond_cols, E_mat = prepare_arpdmf_entropy_matrix(
                    arpdmf_train_df,
                    target_name=target_name,
                    k=int(k_arp),
                    mu_method=str(mu_arp),
                )
                arpdmf_prepare_times_fold.append(round(time.perf_counter() - t_prep, 6))
                entropy_cache[km_key] = (cond_cols, E_mat)
            else:
                cond_cols, E_mat = entropy_cache[km_key]
            for d_arp in deltas:
                t_one = time.perf_counter()
                if cond_cols:
                    sub_arp = reduce_arpdmf_from_matrix(cond_cols, E_mat, float(d_arp))
                else:
                    sub_arp = []
                t_red_one = round(time.perf_counter() - t_one, 6)
                t_reduce_fold += t_red_one
                candidates.append(
                    {
                        "算法": "ARPDMF",
                        "参数": {"delta": float(d_arp), "k": int(k_arp), "mu_method": str(mu_arp)},
                        "子集": sub_arp,
                        "约简耗时秒": t_red_one,
                    }
                )
        fold_reduction_times["ARPDMF"] = round(t_reduce_fold, 6)

    # FNRS（网格多候选）
    if "FNRS" in enabled_algos:
        t0 = time.perf_counter()
        for subset, params in fnrs_mod.generate_fnrs_candidates(train_df, target_name, grids["FNRS"]["lamda"], grids["FNRS"]["alpha"]):
            candidates.append(
                {
                    "算法": "FNRS",
                    "参数": params,
                    "子集": subset,
                    # FNRS 生成器未暴露逐参数耗时；用折内总耗时按候选均分近似。
                    "约简耗时秒": 0.0,
                }
            )
        fnrs_total = round(time.perf_counter() - t0, 6)
        fnrs_cnt = sum(1 for c in candidates if c["算法"] == "FNRS")
        if fnrs_cnt > 0:
            per_fnrs = round(fnrs_total / fnrs_cnt, 6)
            for c in candidates:
                if c["算法"] == "FNRS":
                    c["约简耗时秒"] = per_fnrs
        fold_reduction_times["FNRS"] = fnrs_total

    # FRAR（单候选）
    if "FRAR" in enabled_algos:
        t0 = time.perf_counter()
        t_one = time.perf_counter()
        frar_idx = reduce_frar(train_df, all_train_idx, target_name=target_name)
        t_red_one = round(time.perf_counter() - t_one, 6)
        frar_subset = [columns_all[i] for i in (frar_idx[0] if frar_idx else [])]
        candidates.append({"算法": "FRAR", "参数": {}, "子集": frar_subset, "约简耗时秒": t_red_one})
        fold_reduction_times["FRAR"] = round(time.perf_counter() - t0, 6)

    # IARFCIE（网格多候选）
    if "IARFCIE" in enabled_algos:
        t0 = time.perf_counter()
        for th in grids["IARFCIE"]["threshold"]:
            t_one = time.perf_counter()
            subset = reduce_iarfcie(train_df, target_name=target_name, threshold=th)
            t_red_one = round(time.perf_counter() - t_one, 6)
            candidates.append({"算法": "IARFCIE", "参数": {"threshold": th}, "子集": subset, "约简耗时秒": t_red_one})
        fold_reduction_times["IARFCIE"] = round(time.perf_counter() - t0, 6)

    # MFIGI（网格多候选）
    if "MFIGI" in enabled_algos:
        t0 = time.perf_counter()
        for w in grids["MFIGI"]["w"]:
            t_one = time.perf_counter()
            subset = reduce_mfigi(train_df, target_name=target_name, w=w)
            t_red_one = round(time.perf_counter() - t_one, 6)
            candidates.append({"算法": "MFIGI", "参数": {"w": w}, "子集": subset, "约简耗时秒": t_red_one})
        fold_reduction_times["MFIGI"] = round(time.perf_counter() - t0, 6)

    # MFNMI（delta × theta 网格）
    if "MFNMI" in enabled_algos:
        t0 = time.perf_counter()
        scaler_val = 1 if mfnmi_scaler is None else int(mfnmi_scaler)
        for delta in grids["MFNMI"]["delta"]:
            t_prepare = time.perf_counter()
            _, _, mfnmi_fsr_obj, mfnmi_metric, cols_mfnmi = mfnmi_mod.prepare_fold(
                train_df, all_train_idx, target_name=target_name, scaler=scaler_val, columns_nominal=[], delta=delta
            )
            prepare_cost = time.perf_counter() - t_prepare
            thetas = grids["MFNMI"]["theta"]
            share_prepare = prepare_cost / len(thetas) if thetas else 0.0
            for theta in thetas:
                t_one = time.perf_counter()
                subset = mfnmi_mod.reduce_mfren(cols_mfnmi, mfnmi_metric, mfnmi_fsr_obj, lambda_v=0.0, theta=theta)
                t_red_one = round((time.perf_counter() - t_one) + share_prepare, 6)
                candidates.append({"算法": "MFNMI", "参数": {"delta": delta, "theta": theta}, "子集": subset, "约简耗时秒": t_red_one})
        fold_reduction_times["MFNMI"] = round(time.perf_counter() - t0, 6)

    # ALL（基线）
    if "ALL" in enabled_algos:
        candidates.append({"算法": "ALL", "参数": {}, "子集": list(columns_all), "约简耗时秒": 0.0})
        fold_reduction_times["ALL"] = 0.0

    return candidates, fold_reduction_times, arpdmf_prepare_times_fold

