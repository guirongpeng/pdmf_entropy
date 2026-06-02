"""检查某一折训练集上：哪些列在 MinMax 后非常数、哪些候选子集会导致「全列常数」。

用法：python diagnose_constant_features.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from attribute_reduction import prepare_dataframe
from config import get_param_grids, get_enabled_reductions
from reduction_registry import run_reductions_for_fold


def subset_all_constant_on_train(train_df: pd.DataFrame, subset: list[str], target_name: str) -> tuple[bool, list[tuple[str, int]]]:
    if not subset:
        return True, []
    bad: list[tuple[str, int]] = []
    for c in subset:
        if c == target_name:
            continue
        nu = int(train_df[c].nunique(dropna=False))
        if nu <= 1:
            bad.append((c, nu))
    return len(bad) == len(subset), bad


def main():
    target = "Absenteeism time in hours"
    data = prepare_dataframe(file_path="datas/Absenteeism_two.csv", target_name=target)
    x_all = data.drop(columns=[target])
    y_all = data[target]
    columns_all = list(x_all.columns)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3220822)
    train_index, _ = next(iter(skf.split(x_all, y_all)))

    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_all.iloc[train_index]), columns=columns_all)
    y_train = y_all.iloc[train_index].reset_index(drop=True)
    train_df = x_train.copy()
    train_df[target] = y_train.values
    x_train_raw = x_all.iloc[train_index].reset_index(drop=True)
    train_df_raw = x_train_raw.copy()
    train_df_raw[target] = y_train.values

    grids = get_param_grids(len(columns_all))
    enabled = get_enabled_reductions()

    # 1) 该折训练集上，逐列唯一值个数（MinMax 后）
    print("=== 第1折训练集（MinMax 后）各列 nunique ===")
    const_cols = []
    for c in columns_all:
        nu = int(train_df[c].nunique(dropna=False))
        if nu <= 1:
            const_cols.append(c)
        print(f"  {c[:50]:50s} nunique={nu}")
    print(f"\n在该折上为常数的列数: {len(const_cols)}")
    if const_cols:
        print("常数列:", const_cols)
    else:
        print("（无单列常数；问题来自「子集内全部为常数」需组合检查）")

    # 2) 各算法候选子集：是否导致 CatBoost 不可训练（子集内每一列都常数）
    print("\n=== 各算法候选：子集内是否全部常数（仅当子集含≥1列且全为常数列时触发）===")
    all_train_idx = list(range(len(train_df)))
    cands, _, _ = run_reductions_for_fold(
        train_df, train_df_raw, all_train_idx, target, grids, enabled, columns_all, None
    )
    print(f"候选总数: {len(cands)}")

    problems = []
    for c in cands:
        sub = list(c.get("子集", []))
        if not sub:
            problems.append((c["算法"], c.get("参数"), sub, "empty_subset"))
            continue
        # 子集内是否「列列都是常数」—— CatBoost 要求至少一列非常数
        nonconst = any(train_df[col].nunique(dropna=False) > 1 for col in sub if col != target)
        if not nonconst:
            bad = [(col, int(train_df[col].nunique(dropna=False))) for col in sub if col != target]
            problems.append((c["算法"], c.get("参数"), sub, bad))

    print(f"会导致「无非常数特征」的候选数: {len(problems)}")
    for algo, params, sub, info in problems[:50]:
        print("---")
        print(f"算法: {algo}  params: {params}")
        print(f"  子集长度={len(sub)}  详情: {info}")
        if isinstance(sub, list) and sub:
            print(f"  子集(前10): {sub[:10]}")

    # 3) 单特征子集且该特征在该折常数 → 谁选了它
    print("\n=== 单特征子集且该特征在该折为常数 ===")
    single_bad = [
        (c["算法"], c.get("参数"), sub[0])
        for c in cands
        for sub in [list(c.get("子集", []))]
        if len(sub) == 1 and sub[0] in const_cols
    ]
    for row in single_bad[:20]:
        print(row)


if __name__ == "__main__":
    main()
