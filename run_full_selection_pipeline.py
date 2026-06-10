"""完整属性约简与选优流程（外层10折、同折评估、中文JSON输出）。

实现口径（主口径）：
- 候选收集（按算法 × 参数 × 10 折）：
  1) 每折在“该折训练集”上产出候选子集（不去重）；
  2) 在“该折测试集”上评估候选并记录指标。
- 结果汇总（按最优参数）：
  - 对每个算法按参数组合聚合多模型指标，选 score(p)=多模型均值acc 最高的参数；
  - summary 仅输出最优参数口径（不再输出旧版“算法级折内均值”口径）。
- 时间口径：
  - summary 中「算法参数网格搜索总耗时秒」：约简网格耗时（跨10折累加）；
  - ARPDMF 另加一次「全部 prepare 耗时之均值」（模糊化+批归一化+熵矩阵，与 δ 无关，只加一份）。

实现细节：
- 归一化：每折仅用训练集拟合 MinMaxScaler，再变换 train/test，避免信息泄露；
- 折内候选评估缓存：同一折内相同子集只训练评估一次，其它复用，以节省计算；
- 耗时统计：记录每折每算法的“约简阶段”耗时，算法10折总耗时与 times.json 的对应字段一致。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from results_io import save_full_and_splits
from fold_eval import eval_candidates_fold
from config import get_param_grids, get_model_factories, get_model_groups_raw, DEFAULT_RANDOM_STATE, get_enabled_reductions
from reduction_registry import run_reductions_for_fold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

try:
    from frs.attribute_reduction import prepare_dataframe
except ModuleNotFoundError:
    from attribute_reduction import prepare_dataframe


def _subset_has_varying_feature(train_df: pd.DataFrame, subset: list[str]) -> bool:
    """训练折上至少有一列非常数；全为常数时树模型（如 CatBoost）会报错。"""
    if not subset:
        return False
    X = train_df[subset]
    for c in subset:
        if X[c].nunique(dropna=False) > 1:
            return True
    return False


def _zero_metrics_bundle() -> dict:
    """无法训练/预测时的占位指标（与正常 dict 键一致，便于下游聚合）。"""
    return {
        "acc": 0.0,
        "balanced_acc": 0.0,
        "precision_micro": 0.0,
        "recall_micro": 0.0,
        "f1_micro": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "precision_macro": 0.0,
        "precision_weighted": 0.0,
        "recall_macro": 0.0,
        "recall_weighted": 0.0,
        "mcc": 0.0,
        "roc_auc_macro_ovr": 0.0,
    }


def _param_key(param: dict) -> tuple[tuple[str, str], ...]:
    """将参数字典规范化为可排序键（用于按参数聚合/并列打破）。"""
    if not param:
        return tuple()
    return tuple(sorted((str(k), str(v)) for k, v in param.items()))


def _param_key_to_dict(key: tuple[tuple[str, str], ...]) -> dict:
    """规范键转回可读参数字典。"""
    out: dict = {}
    for k, v in key:
        # 尝试恢复数值类型，失败则保留字符串。
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def _stats_from_metric_bag(bag: dict) -> dict:
    """模型在某参数下、多折候选上的指标列表 → 均值与折间 std。"""
    stats: dict = {}
    for mk, vals in bag.items():
        if vals:
            stats[mk] = float(np.mean(vals))
            stats[f"{mk}_std"] = float(np.std(vals))
        else:
            stats[mk] = 0.0
            stats[f"{mk}_std"] = 0.0
    return stats


def _per_model_avg_from_bag(model_bag: dict) -> dict[str, dict]:
    return {model_name: _stats_from_metric_bag(bag) for model_name, bag in model_bag.items()}


def _summary_avg_from_per_model(per_model_avg: dict[str, dict]) -> dict:
    """各模型均值 → 多模型宏平均（与 summary 平均指标 一致）。"""
    if not per_model_avg:
        return {"acc": 0.0, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0}
    mean_keys: set[str] = set()
    for mv in per_model_avg.values():
        for mk in mv.keys():
            if not mk.endswith("_std"):
                mean_keys.add(mk)
    summary_avg: dict = {}
    for mk in mean_keys:
        model_means = [mv.get(mk, 0.0) for mv in per_model_avg.values()]
        summary_avg[mk] = float(np.mean(model_means)) if model_means else 0.0
        summary_avg[f"{mk}_std"] = float(np.std(model_means)) if model_means else 0.0
    return summary_avg


def _per_group_from_per_model(
    per_model_avg: dict[str, dict],
    model_groups: dict[str, list[str]],
) -> dict[str, dict]:
    """各模型平均指标按 config 六大类再平均。"""
    per_group: dict[str, dict] = {}
    for group_name, group_models in model_groups.items():
        present = [m for m in group_models if m in per_model_avg]
        if not present:
            per_group[group_name] = {
                "acc": 0.0,
                "f1_macro": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            }
            continue
        mean_keys: set[str] = set()
        for m in present:
            for k in per_model_avg[m].keys():
                if not k.endswith("_std"):
                    mean_keys.add(k)
        group_stats: dict = {}
        for mk in mean_keys:
            vals = [per_model_avg[m].get(mk, 0.0) for m in present]
            group_stats[mk] = float(np.mean(vals)) if vals else 0.0
            group_stats[f"{mk}_std"] = float(np.std(vals)) if vals else 0.0
        per_group[group_name] = group_stats
    return per_group


def main():
    parser = argparse.ArgumentParser(description="完整属性约简与选优流程")
    default_csv = Path(__file__).resolve().parent / "datas" / "Absenteeism_two.csv"
    parser.add_argument("--csv", type=str, default=str(default_csv))
    parser.add_argument("--target", type=str, default="Absenteeism time in hours")
    parser.add_argument("--outname", type=str, default="", help="输出目录名（位于 outputs/ 下）。为空则使用数据文件名（不含扩展名）。")
    parser.add_argument(
        "--mfnmi-scaler",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="仅对 MFNMI 生效的scaler选择：1=不缩放，2=MinMaxScaler，3=StandardScaler；默认不覆盖现有设置",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    data = prepare_dataframe(file_path=args.csv, target_name=args.target)
    target_name = args.target
    x_all = data.drop(columns=[target_name])
    y_all = data[target_name]
    columns_all = list(x_all.columns)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3220822)

    # 参数网格集中配置
    grids = get_param_grids(len(columns_all))
    fnrs_lamdas = grids["FNRS"]["lamda"]
    fnrs_alphas = grids["FNRS"]["alpha"]
    iarfcie_thresholds = grids["IARFCIE"]["threshold"]
    mfigi_ws = grids["MFIGI"]["w"]
    mfnmi_thetas = grids["MFNMI"]["theta"]
    mfnmi_deltas = grids["MFNMI"]["delta"]

    fold_records = []
    subset_summary = defaultdict(lambda: {"子集": [], "出现次数": 0, "来源": [], "折准确率": [], "折指标": []})
    reduction_time = 0.0
    reduction_time_by_algo = defaultdict(float)
    # ARPDMF：每一折的 prepare 耗时按“只计一次”口径处理：
    # 统计时先把该折内所有 prepare（不同 (k,μ) 的缓存未命中）累加为一份，再对 10 折取均值，只加一次。
    arpdmf_prepare_times_fold_sums: list[float] = []
    evaluation_time = 0.0

    # 分类模型注册与分组（集中于 config.py）
    all_model_factories = get_model_factories(DEFAULT_RANDOM_STATE)
    model_groups_raw = get_model_groups_raw()
    # 按分组为主：仅评估分组里声明的模型；先按分组顺序收集唯一模型名
    selected_model_names = []
    for lst in model_groups_raw.values():
        for name in lst:
            if name not in selected_model_names:
                selected_model_names.append(name)
    # 实际可用模型（依赖可用性裁剪），且仅限于分组中列出的模型
    models_registry = {name: all_model_factories[name] for name in selected_model_names if name in all_model_factories}
    # 将分组映射为仅包含“实际被评估的模型”
    model_groups = {g: [m for m in lst if m in models_registry] for g, lst in model_groups_raw.items()}

    # 聚合容器：按 算法 -> 模型 -> 指标名 -> 值列表（不去重）
    model_metrics_by_algo = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # 参数维度聚合：按 算法 -> 参数键 -> 模型 -> 指标名 -> 值列表
    model_metrics_by_algo_param = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    # 额外：MFNMI 参数面聚合容器，按 (delta, theta) -> [子集综合acc]
    mfnmi_param_to_accs = defaultdict(list)

    fold_iter = tqdm(
        enumerate(skf.split(x_all, y_all), start=1),
        total=10,
        desc="外层10折",
    )
    for fold_id, (train_index, test_index) in fold_iter:
        fold_iter.set_postfix_str(f"当前第{fold_id}折")
        # 每折归一化：只用训练集拟合，再变换 train/test，防止泄露测试信息
        scaler = MinMaxScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_all.iloc[train_index]), columns=columns_all)
        x_test = pd.DataFrame(scaler.transform(x_all.iloc[test_index]), columns=columns_all)
        x_train_raw = x_all.iloc[train_index].reset_index(drop=True)
        y_train = y_all.iloc[train_index].reset_index(drop=True)
        y_test = y_all.iloc[test_index].reset_index(drop=True)
        train_df = x_train.copy()
        test_df = x_test.copy()
        train_df[target_name] = y_train.values
        test_df[target_name] = y_test.values
        train_df_raw = x_train_raw.copy()
        train_df_raw[target_name] = y_train.values

        all_train_idx = list(range(train_df.shape[0]))
        candidates = []

        t_reduce_start = time.perf_counter()

        fold_reduction_times = {}

        # 根据配置执行启用的属性约简算法（抽离到注册表）
        enabled_algos = get_enabled_reductions()
        run_cands, run_times, prep_times = run_reductions_for_fold(
            train_df=train_df,
            train_df_raw=train_df_raw,
            all_train_idx=all_train_idx,
            target_name=target_name,
            grids=grids,
            enabled_algos=enabled_algos,
            columns_all=columns_all,
            mfnmi_scaler=args.mfnmi_scaler,
        )
        candidates.extend(run_cands)
        fold_reduction_times.update(run_times)
        arpdmf_prepare_times_fold_sums.append(round(float(sum(prep_times)), 6))
        for k, v in run_times.items():
            reduction_time_by_algo[k] += v

        reduction_time += time.perf_counter() - t_reduce_start

        # 同折评估：
        # - 在该折 test 上逐个评估“本折产生的候选子集”；
        # - 当折内对子集做去重缓存，相同子集只训练/预测一次，其余直接复用结果；
        # - 但返回的明细仍按“不去重候选”展开，保证后续“摊平一次均值”的分母正确。
        t_eval_start = time.perf_counter()
        current_fold_records = []
        # 当折内对子集去重缓存：相同子集只评一次，其余复用
        fold_items = eval_candidates_fold(train_df, test_df, candidates, target_name)
        
        for item in fold_items:
            item["折号"] = fold_id
            current_fold_records.append(item)
            key = tuple(sorted(item["子集"]))
            subset_summary[key]["子集"] = list(key)
            subset_summary[key]["出现次数"] += 1
            subset_summary[key]["来源"].append({"折号": fold_id, "算法": item["算法"], "参数": item["参数"], "指标": item["指标"], "准确率": item["准确率"]})
            subset_summary[key]["折准确率"].append(item["准确率"])
            subset_summary[key]["折指标"].append(item["指标"])
        evaluation_time += time.perf_counter() - t_eval_start

        # 额外：对本折所有唯一子集，遍历“所有模型”进行评估，汇总到 model_metrics_by_algo
        # 子集缓存：每个模型在本折对同一子集只评一次
        unique_key_to_subset = {}
        for it in fold_items:
            key = tuple(sorted(it["子集"]))
            unique_key_to_subset[key] = it["子集"]
        # 针对每个模型做一次批量评估
        t_models_start = time.perf_counter()
        # 收集：该折“每个子集”的逐模型 acc，用于方案A计算“子集综合acc”
        subset_to_model_accs = defaultdict(dict)
        for model_name, model_factory in models_registry.items():
            model_cache = {}
            for key, subset in unique_key_to_subset.items():
                # 缓存命中
                if key in model_cache:
                    metrics_m = model_cache[key]
                else:
                    clf = model_factory()
                    metrics_m: dict | None = None
                    if _subset_has_varying_feature(train_df, subset):
                        try:
                            clf.fit(train_df[subset], train_df[target_name])
                            y_pred = clf.predict(test_df[subset])
                            y_true = test_df[target_name]
                            metrics_m = {
                                "acc": float(accuracy_score(y_true, y_pred)),
                                "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
                                "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
                                "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
                                "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                                "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                                "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                                "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                                "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                            }
                            y_score = None
                            try:
                                if hasattr(clf, "predict_proba"):
                                    y_score = clf.predict_proba(test_df[subset])
                                elif hasattr(clf, "decision_function"):
                                    y_score = clf.decision_function(test_df[subset])
                                if y_score is not None:
                                    metrics_m["roc_auc_macro_ovr"] = float(
                                        roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
                                    )
                            except Exception:
                                metrics_m["roc_auc_macro_ovr"] = 0.0
                        except Exception:
                            metrics_m = None
                    if metrics_m is None:
                        metrics_m = _zero_metrics_bundle()
                    model_cache[key] = metrics_m
                # 记录该子集在该模型上的 acc（用于方案A）
                if "acc" in metrics_m:
                    subset_to_model_accs[key][model_name] = float(metrics_m["acc"])
            # 将该模型的指标，按“算法维度”聚合（不去重，需回填到每个候选记录的算法上）
            for it in fold_items:
                algo = it["算法"]
                key = tuple(sorted(it["子集"]))
                pkey = _param_key(it.get("参数", {}))
                m = model_cache.get(key)
                if not m:
                    continue
                for k, v in m.items():
                    model_metrics_by_algo[algo][model_name][k].append(v)
                    model_metrics_by_algo_param[algo][pkey][model_name][k].append(v)
        evaluation_time += time.perf_counter() - t_models_start

        # 方案A：对 MFNMI 的每个候选子集，先对“逐模型acc”做一次均值，得到“子集综合acc”；
        # 然后按 (delta, theta) 聚合到参数面容器（不去重，跨折累积）。
        for it in fold_items:
            if it.get("算法") != "MFNMI":
                continue
            params = it.get("参数", {})
            delta = params.get("delta")
            theta = params.get("theta")
            key = tuple(sorted(it.get("子集", [])))
            model_accs = list(subset_to_model_accs.get(key, {}).values())
            if not model_accs:
                continue
            mean_acc = float(np.mean(model_accs))
            mfnmi_param_to_accs[(delta, theta)].append(mean_acc)
        fold_records.append(
            {
                "折号": fold_id,
                "约简耗时秒": fold_reduction_times,
                "候选数量": len(current_fold_records),
                "明细": current_fold_records,
            }
        )

    # 汇总与最优选择
    unique_subsets = []
    for value in subset_summary.values():
        accs = value["折准确率"]
        value["平均准确率"] = float(np.mean(accs)) if accs else 0.0
        value["准确率标准差"] = float(np.std(accs)) if accs else 0.0
        value["最佳单折准确率"] = float(np.max(accs)) if accs else 0.0
        # 平均多指标
        if value["折指标"]:
            keys = value["折指标"][0].keys()
            value["平均指标"] = {k: float(np.mean([m[k] for m in value["折指标"]])) for k in keys}
        else:
            value["平均指标"] = {"acc": 0.0, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0}
        unique_subsets.append(value)

    unique_subsets = sorted(unique_subsets, key=lambda x: (-x["平均准确率"], len(x["子集"])))

    total_time = time.perf_counter() - t0

    # ARPDMF：全部 prepare 耗时取均值，只加一次到总耗时（folds.json 仍仅为各折三阶段约简）
    if arpdmf_prepare_times_fold_sums:
        # t_fuzzy_once = round(float(np.mean(arpdmf_prepare_times_fold_sums)), 6) # [notice]单折
        t_fuzzy_once = round(float(np.sum(arpdmf_prepare_times_fold_sums)), 6)    # [notice]全部

        reduction_time_by_algo["ARPDMF"] = round(
            reduction_time_by_algo.get("ARPDMF", 0.0) + t_fuzzy_once,
            6,
        )

    # 按算法维度组织输出
    algo_param_grids = {
        # "MFREN": {"lambda_v": [0.0]},
        "FSFrMI": {"lambda_v": [0.0]},
        "ARPDMF": grids["ARPDMF"],
        "FNRS": {"lamda": fnrs_lamdas, "alpha": fnrs_alphas},
        "FRAR": {},
        "IARFCIE": {"threshold": iarfcie_thresholds},
        "MFIGI": {"w": mfigi_ws},
        "MFNMI": {"delta": mfnmi_deltas, "theta": mfnmi_thetas},
        "ALL": {},
    }
    # 注：如需按 (算法, 参数, 子集) 统计出现次数/分布，可在此处补充聚合逻辑；
    # 当前输出不使用该辅助视图，故不保留冗余聚合以保持简洁。

    # 新主口径：按“参数组合”先算 score(p)=多模型均值acc，再取最优参数 p*
    # tie-break: acc 高优先 -> len 短优先 -> time(平均每折) 短优先 -> 参数字典序
    algo_best = {}
    best_param_models: dict[str, dict] = {}
    # 补充参数组合下的长度/耗时/覆盖折数
    param_meta = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "len_sum": 0.0, "time_sum": 0.0, "folds": set()})
    )
    for fold_item in fold_records:
        for rec in fold_item["明细"]:
            algo = rec.get("算法")
            pkey = _param_key(rec.get("参数", {}))
            meta = param_meta[algo][pkey]
            meta["count"] += 1
            meta["len_sum"] += float(len(rec.get("子集", [])))
            if "约简耗时秒" in rec:
                meta["time_sum"] += float(rec.get("约简耗时秒", 0.0))
            meta["folds"].add(int(rec.get("折号", 0)))

    for algo_name in algo_param_grids.keys():
        best_item = None
        best_per_model: dict[str, dict] = {}
        best_per_group: dict[str, dict] = {}
        for pkey, model_bag in model_metrics_by_algo_param.get(algo_name, {}).items():
            per_model_avg = _per_model_avg_from_bag(model_bag)
            summary_avg_metrics = _summary_avg_from_per_model(per_model_avg)
            per_group_avg = _per_group_from_per_model(per_model_avg, model_groups)

            meta = param_meta.get(algo_name, {}).get(pkey, {})
            cnt = int(meta.get("count", 0))
            avg_len = float(meta["len_sum"] / cnt) if cnt else 0.0
            fold_cnt = len(meta.get("folds", set()))
            avg_time_per_fold = float(meta["time_sum"] / fold_cnt) if fold_cnt else 0.0
            acc = float(summary_avg_metrics.get("acc", 0.0))
            candidate = {
                "参数键": pkey,
                "参数": _param_key_to_dict(pkey),
                "平均指标": summary_avg_metrics,
                "候选子集总数": cnt,
                "平均约简子集长度": avg_len,
                "算法约简平均每折耗时秒": round(avg_time_per_fold, 6),
                "覆盖折数": fold_cnt,
            }
            if best_item is None:
                best_item = candidate
                best_per_model = per_model_avg
                best_per_group = per_group_avg
                continue
            cur = (
                -acc,
                avg_len,
                avg_time_per_fold,
                pkey,
            )
            best = (
                -float(best_item["平均指标"].get("acc", 0.0)),
                float(best_item["平均约简子集长度"]),
                float(best_item["算法约简平均每折耗时秒"]),
                best_item["参数键"],
            )
            if cur < best:
                best_item = candidate
                best_per_model = per_model_avg
                best_per_group = per_group_avg

        if best_item is None:
            algo_best[algo_name] = {
                "最优参数": {},
                "平均指标": {"acc": 0.0, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0},
                "候选子集总数": 0,
                "平均约简子集长度": 0.0,
                "算法参数网格搜索总耗时秒": 0.0,
                "覆盖折数": 0,
            }
            best_param_models[algo_name] = {
                "最优参数": {},
                "各模型平均指标": {},
                "各模型族平均指标": {},
            }
        else:
            algo_best[algo_name] = {
                "最优参数": best_item["参数"],
                "平均指标": best_item["平均指标"],
                "候选子集总数": best_item["候选子集总数"],
                "平均约简子集长度": best_item["平均约简子集长度"],
                "算法参数网格搜索总耗时秒": round(reduction_time_by_algo.get(algo_name, 0.0), 6),
                "覆盖折数": best_item["覆盖折数"],
            }
            best_param_models[algo_name] = {
                "最优参数": best_item["参数"],
                "各模型平均指标": best_per_model,
                "各模型族平均指标": best_per_group,
            }

    # 导出：每个属性约简算法 × 每个模型/每个分组 的平均指标（不去重简单均值）
    models_overview_by_algo = {}
    for algo_name, model_bag in model_metrics_by_algo.items():
        # per_model
        per_model = {}
        for model_name, bag in model_bag.items():
            per_model[model_name] = _stats_from_metric_bag(bag)
        per_group = _per_group_from_per_model(per_model, model_groups)
        models_overview_by_algo[algo_name] = {"per_model": per_model, "per_group": per_group}

    # 构建 MFNMI 参数面结果（仅 JSON 输出使用）
    mfnmi_param_surface = []
    if mfnmi_param_to_accs:
        for (delta, theta), acc_list in sorted(mfnmi_param_to_accs.items(), key=lambda x: (x[0][0], x[0][1])):
            if not acc_list:
                continue
            mfnmi_param_surface.append({
                "delta": float(delta) if delta is not None else None,
                "theta": float(theta) if theta is not None else None,
                "acc_mean": float(np.mean(acc_list)),
                "count": int(len(acc_list)),
            })

    # 供 arpdmf_by_params.json 与 summary 对齐：按「各模型在 ARPDMF 候选上的指标序列」做分组聚合
    arpdmf_model_series: dict = {}
    if "ARPDMF" in model_metrics_by_algo:
        for model_name, metric_dict in model_metrics_by_algo["ARPDMF"].items():
            arpdmf_model_series[model_name] = {k: list(v) for k, v in metric_dict.items()}

    out_data = {
        "任务说明": "外层10折，同折train约简+test评估；按参数组合选最优后汇总",
        "数据集": str(Path(args.csv).resolve()),
        "目标列": target_name,
        "样本数": int(data.shape[0]),
        "特征数": int(len(columns_all)),
        "折数": 10,
        "参数网格": algo_param_grids,
        "每折记录": fold_records,
        "去重后子集总数": len(unique_subsets),
        "去重后子集": unique_subsets,
        "按算法结果_最优参数": algo_best,
        "最优参数_各模型指标": best_param_models,
        "ARPDMF_模型指标序列": arpdmf_model_series,
        "模型_按算法": models_overview_by_algo,
        # 仅保存阶段使用：额外输出 MFNMI 参数面
        "MFNMI_参数面": mfnmi_param_surface,
        "耗时秒": {
            "约简阶段": round(reduction_time, 6),
            "约简阶段_按算法汇总": {k: round(v, 6) for k, v in reduction_time_by_algo.items()},
            "评估阶段": round(evaluation_time, 6),
            "总耗时": round(total_time, 6),
        },
    }

    csv_path = Path(args.csv).resolve()
    dataset_name = args.outname.strip() if args.outname.strip() else csv_path.stem
    # 若指定了 mfnmi-scaler，在目录名后追加 _<s> 以区分
    if args.mfnmi_scaler in (1, 2, 3):
        dataset_name = f"{dataset_name}_{args.mfnmi_scaler}"
    out_path = (Path(__file__).resolve().parent / "outputs" / dataset_name / "full_pipeline_result.json").resolve()
    save_full_and_splits(out_data, out_path)

    print(f"完成，结果已保存：{out_path}\n并已拆分至目录：{out_path.parent}")


if __name__ == "__main__":
    main()

#   python .\run_full_selection_pipeline.py --csv "datas\Absenteeism_two.csv" --target "Absenteeism time in hours" --outname abs_custom --mfnmi-scaler 2
#   python .\run_full_selection_pipeline.py --csv "datas\Absenteeism_two.csv" --target "Absenteeism time in hours" --mfnmi-scaler 2
