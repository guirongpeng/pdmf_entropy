"""
汇总 outputs 下各实验的 arpdmf_by_params.json：
- 每个数据集上 acc 最优的参数组合；
- 同一 (delta, k, mu_method) 在多个数据集上的平均 acc（网格一致时可比）；
- 单维边际：只看 delta / 只看 k / 只看 mu_method 时的平均表现（对另两维取平均）。

用法（项目根目录）:
  python diagnose/analyze_arpdmf_by_params.py
  python diagnose/analyze_arpdmf_by_params.py --metric f1_macro
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("按参数组合", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="跨数据集分析 ARPDMF 参数效果")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="含各数据集子目录的根目录（默认仓库下 outputs/）",
    )
    parser.add_argument(
        "--metric",
        default="acc",
        help="用于排序与汇总的指标键（平均指标下的主值，不含 _std）",
    )
    args = parser.parse_args()

    root: Path = args.outputs_dir
    files = sorted(root.glob("*/arpdmf_by_params.json"))
    if not files:
        print(f"未找到 arpdmf_by_params.json: {root}")
        return

    metric = args.metric
    # 收集: key=(d,k,mu) -> [acc per dataset]
    by_combo: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    by_delta: dict[float, list[float]] = defaultdict(list)
    by_k: dict[int, list[float]] = defaultdict(list)
    by_mu: dict[str, list[float]] = defaultdict(list)

    per_dataset_best: list[tuple[str, float, dict, dict]] = []

    for fp in files:
        name = fp.parent.name
        rows = load_rows(fp)
        if not rows:
            print(f"[跳过无数据] {name}")
            continue
        scored: list[tuple[float, dict]] = []
        for row in rows:
            p = row.get("参数") or {}
            m = row.get("平均指标") or {}
            if metric not in m:
                print(f"[警告] {name} 缺少指标 {metric}，跳过该行")
                continue
            v = float(m[metric])
            scored.append((v, row))

        scored.sort(key=lambda x: -x[0])
        best_v, best_row = scored[0]
        per_dataset_best.append((name, best_v, best_row.get("参数") or {}, best_row))

        for v, row in scored:
            p = row.get("参数") or {}
            try:
                key = (float(p.get("delta")), int(p.get("k")), str(p.get("mu_method", "")))
            except (TypeError, ValueError):
                continue
            by_combo[key].append((name, v))
            by_delta[key[0]].append(v)
            by_k[key[1]].append(v)
            by_mu[key[2]].append(v)

    print("=" * 72)
    print(f"指标: {metric}  |  数据集数: {len(files)}  |  参与汇总的数据集: {len(per_dataset_best)}")
    print("=" * 72)

    print("\n【各数据集上该指标最优的一组参数】\n")
    for name, v, params, _ in sorted(per_dataset_best, key=lambda x: -x[1]):
        print(f"  {name:24s}  {metric}={v:.6f}  {params}")

    # 跨数据集：同一完整参数组的平均指标（仅当所有数据集都出现该组时给「稳健」样本数；否则为出现过的平均）
    print("\n【完整 (delta, k, mu_method) 组合：跨数据集平均 " + metric + "（按均值降序 Top 15）】\n")
    combo_summary = []
    for key, pairs in by_combo.items():
        accs = [p[1] for p in pairs]
        combo_summary.append(
            (
                statistics.mean(accs),
                statistics.stdev(accs) if len(accs) > 1 else 0.0,
                len(accs),
                key,
            )
        )
    combo_summary.sort(key=lambda x: -x[0])
    for mean_v, std_v, n, key in combo_summary[:15]:
        d, k, mu = key
        print(f"  δ={d:g}  k={k}  μ={mu!r}  |  mean={mean_v:.6f}  std={std_v:.6f}  n数据集={n}")

    def marginal(title: str, bucket: dict, key_fmt) -> None:
        print(f"\n【边际：{title}（对其它维度与所有样本行取平均后再按数据集内聚合，此处为池化后的单值分布均值）】")
        print("说明: 下列为「所有 (参数行) 中该维度取某值时的 " + metric + " 的算术平均」，用于粗看偏好区间。\n")
        items = []
        for k0, vals in bucket.items():
            if not vals:
                continue
            items.append((statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals), k0))
        items.sort(key=lambda x: -x[0])
        for mean_v, std_v, n, k0 in items:
            print(f"  {key_fmt(k0):50s}  mean={mean_v:.6f}  std={std_v:.6f}  n行={n}")

    marginal("delta", by_delta, lambda d: f"delta = {d:g}")
    marginal("k", by_k, lambda k: f"k = {k}")
    marginal("mu_method", by_mu, lambda m: f"mu_method = {m!r}")

    print("\n" + "=" * 72)
    print("解读提示: ")
    print("  - 「各数据集最优」受数据噪声影响大；跨数据集 mean 高的组合更稳。")
    print("  - 边际均值会被网格不均匀与样本量扭曲，仅作粗参考；以完整三元组 Top 为准更可靠。")
    print("=" * 72)


if __name__ == "__main__":
    main()
