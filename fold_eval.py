from __future__ import annotations

from typing import List, Dict, Any

from metrics_eval import eval_subset


def eval_candidates_fold(train_df, test_df, candidates: List[Dict[str, Any]], target_name: str) -> List[Dict[str, Any]]:
    """
    评估当折候选子集（不去重口径），当折内对子集去重做缓存：相同子集只计算一次，其余复用结果。
    返回的列表仍保留原候选粒度（包含重复），以便后续“按算法不去重求均值”的分母正确。
    """
    cache = {}
    results: List[Dict[str, Any]] = []
    for cand in candidates:
        subset = list(dict.fromkeys(cand.get("子集", [])))
        if not subset:
            continue
        key = tuple(sorted(subset))
        if key not in cache:
            metrics = eval_subset(train_df, test_df, subset, target_name)
            cache[key] = metrics
        else:
            metrics = cache[key]
        row = {
            "算法": cand.get("算法"),
            "参数": cand.get("参数", {}),
            "子集": subset,
            "指标": metrics,
            "准确率": metrics.get("acc", 0.0),
        }
        if "约简耗时秒" in cand:
            row["约简耗时秒"] = cand["约简耗时秒"]
        results.append(row)
    return results

