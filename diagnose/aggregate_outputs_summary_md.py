"""
从 outputs 下各数据集子目录的 summary.json 汇总为 Markdown 表格（最优参数口径）。

用法（在仓库根目录 d:\\bnu\\frs）:
  python diagnose/aggregate_outputs_summary_md.py
  python diagnose/aggregate_outputs_summary_md.py --root outputs/未归一化 --out outputs/未归一化/汇总_约简算法对比.md
  python diagnose/aggregate_outputs_summary_md.py --root outputs/归一化 --out outputs/归一化/汇总_约简算法对比.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# 仓库根目录（本文件位于 diagnose/ 下）
_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_summary_block(obj: dict) -> dict:
    block = obj.get("算法最优参数效果（10折，多模型）")
    if block is None:
        raise RuntimeError("summary.json 缺少主口径字段：算法最优参数效果（10折，多模型）")
    return block


def fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def _algo_has_results(info: dict) -> bool:
    """本次实验是否实际跑过该算法（summary 中未启用算法占位为 候选子集总数=0）。"""
    try:
        return int(info.get("候选子集总数") or 0) > 0
    except (TypeError, ValueError):
        return False


def _same_number(a: object, b: object) -> bool:
    """与 json 中读出的标量比较（允许 float 与 int 混用）。"""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if float(a).is_integer() and float(b).is_integer():
            return int(a) == int(b)
    return math.isclose(fa, fb, rel_tol=1e-12, abs_tol=1e-12)


def verify_against_json_on_disk(
    root: Path,
    data: dict[str, dict[str, dict]],
    algos: list[str],
) -> None:
    """
    二次独立读取每个子目录下的 summary.json，与已解析的 data 逐字段比对。
    若不一致则抛出异常，避免表格与源 JSON 脱节（脚本内无数值硬编码）。
    """
    for sp in sorted(root.glob("*/summary.json"), key=lambda p: p.parent.name.lower()):
        ds = sp.parent.name
        obj = load_summary(sp)
        block = _get_summary_block(obj)
        for algo in algos:
            if algo not in block:
                if algo in data and ds in data.get(algo, {}):
                    raise RuntimeError(f"校验失败：{sp} 中无算法 {algo}，但 data 中有记录")
                continue
            info = block[algo]
            if not _algo_has_results(info):
                raise RuntimeError(f"校验失败：{sp} 中算法 {algo} 候选子集总数为 0，不应出现在汇总列")
            m = info.get("平均指标", {})
            expected = {
                "acc": m.get("acc"),
                "rank": info.get("rank"),
                "time": info.get("算法参数网格搜索总耗时秒", info.get("算法约简平均每折耗时秒")),
                "avg_len": info.get("平均约简子集长度"),
            }
            actual = data.get(algo, {}).get(ds)
            if actual is None:
                raise RuntimeError(f"校验失败：{sp} / {algo} 在 data 中缺失")
            for k, ev in expected.items():
                av = actual.get(k)
                if k == "rank":
                    if ev is not None and av is not None and int(ev) != int(av):
                        raise RuntimeError(
                            f"校验失败 rank: {sp} {algo} json={ev} data={av}"
                        )
                    if (ev is None) != (av is None):
                        raise RuntimeError(f"校验失败 rank None: {sp} {algo}")
                else:
                    if not _same_number(ev, av):
                        raise RuntimeError(
                            f"校验失败 {k}: {sp} {algo} json={ev!r} data={av!r}"
                        )


def build_analysis_section(
    data: dict[str, dict[str, dict]],
    algos: list[str],
    datasets: list[str],
    root: Path,
) -> list[str]:
    """基于已加载的 best-summary 数据与本目录下 arpdmf_by_params.json 生成简要文字与表格。"""
    out: list[str] = ["## 简要分析\n"]

    # 1) 算法平均 acc 排序（仅含实际参与实验的算法列）
    means: list[tuple[str, float]] = []
    for a in algos:
        vals: list[float] = []
        for ds in datasets:
            cell = data.get(a, {}).get(ds, {})
            v = cell.get("acc")
            if v is not None:
                vals.append(float(v))
        if vals:
            means.append((a, sum(vals) / len(vals)))
    means.sort(key=lambda x: -x[1])

    out.append("### 1. 约简算法（跨数据集平均 acc）\n")
    out.append("由上表「平均结果」行可知，算法按平均准确率大致排序如下（数值越高越好）：\n")
    for i, (a, m) in enumerate(means, 1):
        out.append(f"{i}. **{a}**：平均 acc ≈ {m:.4f}")
    out.append(
        "\n**选型提示**：优先在验证集/交叉验证上对比排名靠前的算法；**ALL** 为全特征基线，"
        "若某约简算法平均 acc 接近或低于 ALL，说明在该批数据上约简收益有限或需调参。"
        "耗时与平均子集长度可结合上表权衡（更短子集通常更易解释，但未必 acc 更高）。\n"
    )

    # 2) ARPDMF：(δ,k,μ) 跨数据集平均
    ap_files = sorted(root.glob("*/arpdmf_by_params.json"))
    out.append("### 2. ARPDMF 参数（δ, k, μ）\n")
    if not ap_files:
        out.append("本目录下未找到各数据集子目录中的 `arpdmf_by_params.json`，无法汇总参数面。\n")
        return out

    bucket: dict[tuple[float, int, str], list[float]] = defaultdict(list)
    for fp in ap_files:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        for row in obj.get("按参数组合", []):
            p = row.get("参数") or {}
            m = row.get("平均指标") or {}
            acc = m.get("acc")
            if acc is None:
                continue
            try:
                key = (float(p.get("delta")), int(p.get("k")), str(p.get("mu_method", "")))
            except (TypeError, ValueError):
                continue
            bucket[key].append(float(acc))

    if not bucket:
        out.append("`arpdmf_by_params.json` 中无有效参数行。\n")
        return out

    ranked: list[tuple[float, int, tuple[float, int, str]]] = []
    for k, vs in bucket.items():
        ranked.append((sum(vs) / len(vs), len(vs), k))
    ranked.sort(key=lambda x: (-x[0], -x[1]))

    n_ds = len(datasets)
    out.append(
        "下表为 **(δ, k, μ) 组合** 在各数据集上的 **平均 acc**（每个组合在若干数据集上出现则对出现次数取平均）。"
        f"本批共 **{n_ds}** 个数据集。\n"
    )
    out.append("| 排名 | δ | k | μ | 平均 acc | 样本数（数据集条数） |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for i, (mean_acc, cnt, ktuple) in enumerate(ranked[:12], 1):
        d, kk, mu = ktuple
        out.append(f"| {i} | {d:g} | {kk} | {mu!r} | {mean_acc:.4f} | {cnt} |")
    out.append(
        "\n**参数怎么选更好（经验性、仅供初筛）**：\n"
        "- 优先在 **平均 acc 高** 且 **样本数=数据集数×1**（即 9 个数据集都有该组合）的 δ–k–μ 上微调；若样本数较少，可能是部分数据上缺该网格。\n"
        "- **δ**：控制约简与全集熵差阈值，通常宜在网格中部先试，再向两头扫；以本表 Top 区间为参考。\n"
        "- **k**（邻域）与 **μ（A/B/C）**：对边际影响常小于 δ，可先在 Top 组合附近固定 k、μ，主要调 δ。\n"
        "- 最终应以 **同一验证方案**（如当前外层折）在目标数据集上复现为准。\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "outputs" / "未归一化",
        help="含各数据集子目录的根路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 Markdown 路径（默认: <root>/汇总_约简算法对比.md）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="文档一级标题（默认: <根目录名> 实验汇总（来自各数据集 summary.json））",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="不追加「简要分析」与 ARPDMF 参数表",
    )
    args = parser.parse_args()
    root: Path = args.root
    if not root.is_absolute():
        root = (_REPO_ROOT / root).resolve()
    out_path = args.out or (root / "汇总_约简算法对比.md")
    if out_path is not None and not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()

    summaries = sorted(root.glob("*/summary.json"), key=lambda p: p.parent.name.lower())
    if not summaries:
        print(f"未找到 summary.json: {root}")
        return

    datasets: list[str] = []
    data: dict[str, dict[str, dict]] = {}
    all_algos: set[str] = set()

    for sp in summaries:
        ds_name = sp.parent.name
        datasets.append(ds_name)
        obj = load_summary(sp)
        block = _get_summary_block(obj)
        for algo, info in block.items():
            if not _algo_has_results(info):
                continue
            all_algos.add(algo)
            m = info.get("平均指标", {})
            acc = m.get("acc")
            data.setdefault(algo, {})[ds_name] = {
                "acc": acc,
                "rank": info.get("rank"),
                "time": info.get("算法参数网格搜索总耗时秒", info.get("算法约简平均每折耗时秒")),
                "avg_len": info.get("平均约简子集长度"),
            }

    algos = sorted(all_algos)
    if not algos:
        print(f"未找到有效算法列（各 summary 中 候选子集总数 均为 0）: {root}")
        return
    verify_against_json_on_disk(root, data, algos)

    doc_title = (
        args.title
        if args.title
        else f"{root.name} 实验汇总（来自各数据集 `summary.json`）"
    )
    lines: list[str] = []
    lines.append(f"# {doc_title}\n")
    lines.append(
        "指标口径：与 `summary.json` 一致（多模型平均 `acc`、summary 内 `rank`、约简总耗时、平均约简子集长度）。"
        "（主口径为“最优参数组合”）"
        "表格仅包含 **候选子集总数>0** 的算法（未启用或未跑的算法不占列）。"
        "每表末行「平均结果」为各算法列在全部数据集上的算术平均（该列有缺失则仅对非空值平均）。\n"
    )

    def make_table(title: str, key: str, nd: int = 4) -> None:
        lines.append(f"## {title}\n")
        header = "| 数据集 | " + " | ".join(algos) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(algos)) + " |"
        lines.append(header)
        lines.append(sep)
        for ds in datasets:
            row = [ds]
            for a in algos:
                cell = data.get(a, {}).get(ds, {})
                v = cell.get(key)
                if v is None:
                    row.append("—")
                elif key == "rank":
                    row.append(str(int(v)) if v is not None else "—")
                elif key in ("time", "avg_len"):
                    row.append(fmt_float(float(v), nd) if v is not None else "—")
                else:
                    row.append(fmt_float(float(v), nd))
            lines.append("| " + " | ".join(row) + " |")
        # 各列对全部数据集取算术平均（有缺失则仅对非空值平均）
        avg_row: list[str] = ["平均结果"]
        for a in algos:
            vals: list[float] = []
            for ds in datasets:
                cell = data.get(a, {}).get(ds, {})
                v = cell.get(key)
                if v is not None:
                    vals.append(float(v))
            if not vals:
                avg_row.append("—")
            else:
                mean_v = sum(vals) / len(vals)
                if key == "rank":
                    avg_row.append(f"{mean_v:.2f}")
                else:
                    avg_row.append(fmt_float(mean_v, nd))
        lines.append("| " + " | ".join(avg_row) + " |")
        lines.append("")

    make_table("平均准确率 acc", "acc", 4)
    make_table("准确率排名 rank（1 为最好）", "rank", 0)
    make_table("算法参数网格搜索总耗时（秒）", "time", 3)
    make_table("平均约简子集长度", "avg_len", 3)

    if not args.no_analysis:
        lines.extend(build_analysis_section(data, algos, datasets, root))

    lines.append("---\n*由 `diagnose/aggregate_outputs_summary_md.py` 自动生成。*\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
