from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def _mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _arpdmf_model_series_len(model_seqs: dict[str, dict[str, list]]) -> int:
    """与「明细」中 ARPDMF 候选顺序对齐的序列长度（取任一非 *_std 指标列表长度）。"""
    for m_dict in model_seqs.values():
        for k, vals in m_dict.items():
            if not k.endswith("_std"):
                return len(vals)
    return 0


def _aggregate_arpdmf_by_indices_like_summary(
    model_seqs: dict[str, dict[str, list]],
    indices: list[int],
) -> dict[str, float]:
    """
    与 run_full_selection_pipeline 中按算法 summary 相同：
    先对每个模型在 indices 上求各指标均值（及该模型内折间 std），再对各模型的「均值」跨模型取 mean/std。
    """
    if not model_seqs or not indices:
        return {}
    per_model_avg: dict[str, dict[str, float]] = {}
    for model_name, m_dict in model_seqs.items():
        stats: dict[str, float] = {}
        for mk, vals in m_dict.items():
            if mk.endswith("_std"):
                continue
            picked = [vals[i] for i in indices if i < len(vals)]
            if not picked:
                continue
            stats[mk] = float(np.mean(picked))
            stats[f"{mk}_std"] = float(np.std(picked)) if len(picked) > 1 else 0.0
        if stats:
            per_model_avg[model_name] = stats
    if not per_model_avg:
        return {}
    all_metric_keys: set[str] = set()
    for mv in per_model_avg.values():
        for key in mv.keys():
            if not key.endswith("_std"):
                all_metric_keys.add(key)
    summary_avg: dict[str, float] = {}
    for mk in sorted(all_metric_keys):
        model_means = [mv.get(mk, 0.0) for mv in per_model_avg.values()]
        summary_avg[mk] = float(np.mean(model_means)) if model_means else 0.0
        summary_avg[f"{mk}_std"] = float(np.std(model_means)) if len(model_means) > 1 else 0.0
    return summary_avg


def _lr_only_metrics_from_records_for_param(
    out_data: dict[str, Any],
    param_key: tuple[float, int, str],
) -> dict[str, float]:
    """无 ARPDMF_模型指标序列 时的回退：仅用明细中的 LR 指标（与旧版一致）。"""
    acc: list[float] = []
    f1_macro: list[float] = []
    precision_macro: list[float] = []
    recall_macro: list[float] = []
    for fold in out_data.get("每折记录", []):
        for rec in fold.get("明细", []):
            if rec.get("算法") != "ARPDMF":
                continue
            p = rec.get("参数") or {}
            try:
                key = (float(p.get("delta")), int(p.get("k")), str(p.get("mu_method", "")))
            except (TypeError, ValueError):
                continue
            if key != param_key:
                continue
            m = rec.get("指标") or {}
            acc.append(float(m.get("acc", 0.0)))
            f1_macro.append(float(m.get("f1_macro", 0.0)))
            precision_macro.append(float(m.get("precision_macro", 0.0)))
            recall_macro.append(float(m.get("recall_macro", 0.0)))
    return {
        "acc": _mean(acc),
        "f1_macro": _mean(f1_macro),
        "precision_macro": _mean(precision_macro),
        "recall_macro": _mean(recall_macro),
    }


def _build_arpdmf_by_params(out_data: dict[str, Any]) -> dict[str, Any]:
    """
    从「每折记录」汇总 ARPDMF：按 (delta, k, mu_method) 分组。
    若存在 ARPDMF_模型指标序列，则平均指标与 summary 中 ARPDMF 口径一致（多模型：先各模型组内均值，再跨模型均值）；
    否则回退为明细中的 LR 四项指标。
    """
    from collections import defaultdict

    model_seqs: dict[str, dict[str, list]] = out_data.get("ARPDMF_模型指标序列") or {}

    bucket_indices: dict[tuple[float, int, str], list[int]] = defaultdict(list)
    bucket_meta: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"子集长度": [], "约简耗时秒": []}
    )
    global_idx = 0
    for fold in out_data.get("每折记录", []):
        for rec in fold.get("明细", []):
            if rec.get("算法") != "ARPDMF":
                continue
            p = rec.get("参数") or {}
            try:
                key = (float(p.get("delta")), int(p.get("k")), str(p.get("mu_method", "")))
            except (TypeError, ValueError):
                continue
            bucket_indices[key].append(global_idx)
            global_idx += 1
            b = bucket_meta[key]
            b["子集长度"].append(len(rec.get("子集", [])))
            if "约简耗时秒" in rec:
                b["约简耗时秒"].append(float(rec["约简耗时秒"]))

    expected_n = global_idx
    use_multi = bool(model_seqs) and (
        expected_n == 0 or _arpdmf_model_series_len(model_seqs) == expected_n
    )

    rows: list[dict[str, Any]] = []
    for key in sorted(bucket_indices.keys(), key=lambda x: (x[0], x[1], x[2])):
        d, k, mu = key
        indices = bucket_indices[key]
        bag = bucket_meta[key]
        n = len(indices)

        if use_multi and indices:
            avg_metrics = _aggregate_arpdmf_by_indices_like_summary(model_seqs, indices)
        else:
            avg_metrics = _lr_only_metrics_from_records_for_param(out_data, key)

        row: dict[str, Any] = {
            "参数": {"delta": d, "k": k, "mu_method": mu},
            "有效折记录数": n,
            "平均指标": avg_metrics,
            "平均约简子集长度": _mean(bag["子集长度"]),
        }
        if bag["约简耗时秒"]:
            row["平均约简耗时秒"] = _mean(bag["约简耗时秒"])
            row["约简耗时秒_合计"] = round(sum(bag["约简耗时秒"]), 6)
        rows.append(row)

    说明 = (
        "仅 ARPDMF；按 (delta, k, mu_method) 分组。"
        + (
            "平均指标与 summary 中 ARPDMF 一致：多模型下先各模型在组内候选上求指标均值，再对各模型均值取平均。"
            if use_multi
            else "平均指标为明细中 LR 评估（旧版无多模型序列时的回退）。"
        )
    )
    return {"说明": 说明, "按参数组合": rows}


def save_full_and_splits(out_data: dict, out_path: Path) -> None:
    """Save full result and split views next to it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) full
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # 2) splits
    split_dir = out_path.parent
    temp_dir = split_dir / "sub_info"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 主口径：直接使用“最优参数组合”结果
    algo_fold_avg = out_data.get("按算法结果_最优参数", {})
    # 排名规则：
    # 1) acc 降序；
    # 2) acc 相同时，平均约简子集长度升序；
    # 3) 仍相同时，算法参数网格搜索总耗时秒升序；
    # 4) 再相同时，最优参数字典序升序（保证稳定可复现）。
    sortable = []
    for name, obj in algo_fold_avg.items():
        acc = obj.get("平均指标", {}).get("acc", 0.0)
        avg_len = obj.get("平均约简子集长度", float("inf"))
        avg_time = obj.get("算法参数网格搜索总耗时秒", float("inf"))
        p = obj.get("最优参数", {})
        pkey = tuple(sorted((str(k), str(v)) for k, v in p.items()))
        sortable.append((name, acc, avg_len, avg_time, pkey, obj))
    sortable.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))

    # 标准竞赛排名（1,2,2,2,5）：同分同名次，下一名次=当前索引(1-based)
    name_to_rank = {}
    last_acc = None
    last_rank = 0
    for idx, (name, acc, _avg_len, _avg_time, _pkey, _obj) in enumerate(sortable, start=1):
        if last_acc is None or acc != last_acc:
            last_rank = idx
            last_acc = acc
        name_to_rank[name] = last_rank
    # 以排序后的顺序构建输出，保证 summary.json 中键的顺序即为排序顺序
    algo_fold_avg_with_rank = {}
    for name, _acc, _avg_len, _avg_time, _pkey, obj in sortable:
        new_obj = dict(obj)
        new_obj["rank"] = name_to_rank.get(name, None)
        algo_fold_avg_with_rank[name] = new_obj

    base_summary = {
        "任务说明": out_data["任务说明"],
        "数据集": out_data["数据集"],
        "目标列": out_data["目标列"],
        "样本数": out_data["样本数"],
        "特征数": out_data["特征数"],
        "折数": out_data["折数"],
        "算法最优参数效果（10折，多模型）": algo_fold_avg_with_rank,
    }
    (split_dir / "summary.json").write_text(
        json.dumps(base_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    arpdmf_view = _build_arpdmf_by_params(out_data)
    (split_dir / "arpdmf_by_params.json").write_text(
        json.dumps(arpdmf_view, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_param_models = out_data.get("最优参数_各模型指标") or {}
    best_param_payload = {
        "说明": (
            "各约简算法在 summary 所选最优参数下："
            "各模型平均指标=该模型10折均值；"
            "各模型族平均指标=族内各模型均值的再平均（六大类）；"
            "summary.json 的平均指标=各模型 acc 等等权平均（非六族等权）。"
        ),
        "按算法": best_param_models,
    }
    (split_dir / "best_param_models.json").write_text(
        json.dumps(best_param_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (temp_dir / "param_grids.json").write_text(
        json.dumps(out_data["参数网格"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (temp_dir / "folds.json").write_text(
        json.dumps(out_data["每折记录"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subsets_payload = {
        "去重后子集总数": out_data["去重后子集总数"],
        "去重后子集": out_data["去重后子集"],
    }
    (temp_dir / "unique_subsets.json").write_text(
        json.dumps(subsets_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 取消旧版“算法级折内均值”视图，仅保留主口径 summary.json

    # 新增：每个“属性约简算法 × 分类模型”的平均指标视图
    if "模型_按算法" in out_data:
        (split_dir / "models_overview_by_algorithm.json").write_text(
            json.dumps(out_data["模型_按算法"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 额外输出：MFNMI 参数面（delta, theta, acc_mean, count）
    if "MFNMI_参数面" in out_data and out_data["MFNMI_参数面"]:
        (temp_dir / "mfnmi_param_surface.json").write_text(
            json.dumps(out_data["MFNMI_参数面"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 同步生成三维图（仅PNG）。若缺少绘图库则忽略。
        try:
            from plot_mfnmi_surface import plot_surface_from_json  # type: ignore
            plot_surface_from_json(
                json_path=temp_dir / "mfnmi_param_surface.json",
                out_dir=temp_dir,
                title="MFNMI Parameter Surface",
                cmap_name="viridis",
                dpi=300,
                elev=35.0,
                azim=-60.0,
                include_contour=True,
            )
        except Exception:
            pass

    (temp_dir / "times.json").write_text(
        json.dumps(out_data["耗时秒"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields_doc = """
本目录各JSON字段说明（固定格式）：

1) summary.json
- 任务说明：字符串；流程概述。
- 数据集：绝对路径。
- 目标列：用于分类的目标字段名。
- 样本数/特征数/折数：整型。
- 算法最优参数效果（10折，多模型）：字典；每个算法包含：
  - 最优参数：按 score(p)=10折多模型均值acc 选出的参数组合。
  - 平均指标：最优参数组合下的多模型平均指标（acc/f1_macro/precision_macro/recall_macro 等）。
  - 候选子集总数：最优参数组合在10折中产生的候选子集总数（通常≈折数）。
  - 平均约简子集长度：最优参数组合下候选子集长度均值。
  - 算法参数网格搜索总耗时秒：该算法在该数据集上完成参数网格搜索（跨10折）的约简总耗时（秒，不含分类评估）。
      ARPDMF = 10折三阶段约简耗时之和 + mean(每次 prepare_arpdmf_entropy_matrix 耗时)（prepare 只加一份均值；
      实际 prepare 次数为各折各 (k,μ) cache 未命中次数；folds.json 中 ARPDMF 仍仅为该折三阶段约简）。
  - rank：按 acc 降序；并列时依次按长度短、耗时短、参数字典序。

2) param_grids.json
- 各算法的参数候选集合（字典）。

3) folds.json
- 每折记录数组：
  - 折号：1..K
  - 约简耗时秒：字典，键为算法名，值为该折该算法约简耗时（秒）。
  - 候选数量：该折产生的候选子集数量。
  - 明细：数组：
    - 折号/算法/参数/子集/准确率。

4) unique_subsets.json
- 去重后子集总数：整型。
- 去重后子集：数组，元素为：
  - 子集：属性名列表（排序）。
  - 出现次数：该子集在10折中出现的次数。
  - 来源：产生该子集的折号/算法/参数/准确率。
  - 折准确率：该子集在各折上的准确率列表。
  - 平均准确率/准确率标准差/最佳单折准确率：统计指标。

（已取消输出 algorithms_overview.json 与 algo_*.json，summary.json 已覆盖所需信息）

5) arpdmf_by_params.json（与 summary.json 同目录）
- 仅 ARPDMF；按参数组合 (delta, k, mu_method) 分组汇总 10 折明细。
- 平均指标：与 summary 中 ARPDMF 一致（多模型：各模型在组内候选上求均值，再对各模型均值取平均）；旧结果无 ARPDMF_模型指标序列 时回退为明细 LR 四项。
- 按参数组合：数组；每项含 参数、有效折记录数、平均指标、平均约简子集长度、平均约简耗时秒、约简耗时秒_合计。

6) best_param_models.json（与 summary.json 同目录）
- 按算法：最优参数、各模型平均指标（LR/RF/KNN…）、各模型族平均指标（六大类）。
- 与 summary 同参数口径；summary 的平均指标=各模型指标再对模型等权平均。

7) times.json
- 约简阶段：所有算法在10折上的总约简耗时（秒）。
- 约简阶段_按算法汇总：每个算法的10折总约简耗时（秒，不含评估）。与 summary 中各算法“算法参数网格搜索总耗时秒”一致。
- 评估阶段：10折分类评估总耗时（秒）。
- 总耗时：全流程总耗时（秒）。
""".strip()
    (temp_dir / "FIELDS_README.txt").write_text(fields_doc + "\n", encoding="utf-8")

