"""
合并「全算法 baseline」与「patch 目录中指定算法」的 summary，生成汇总_约简算法对比.md。

典型用法：
- 仅 ARPDMF 来自 patch：--pdmf outputs/归一化_pdmf --patch-algos ARPDMF
- ARPDMF + MFNMI 来自 patch：--pdmf outputs/归一化_pdmf_2 --patch-algos ARPDMF,MFNMI

其余算法列始终来自 --baseline（默认 outputs/归一化）。
rank：合并后按 acc 重算（同 results_io）。

用法（仓库根目录）:
  python diagnose/merge_guiyi_pdmf_aggregate_md.py
  python diagnose/merge_guiyi_pdmf_aggregate_md.py --pdmf outputs/归一化_pdmf_2 --patch-algos ARPDMF,MFNMI
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_summary_block(obj: dict) -> dict:
    block = obj.get("算法最优参数效果（10折，多模型）")
    if block is None:
        raise RuntimeError("summary.json 缺少字段：算法最优参数效果（10折，多模型）")
    return block


def fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def extract_cell(info: dict | None) -> dict[str, object | None]:
    if not info:
        return {"acc": None, "time": None, "avg_len": None}
    m = info.get("平均指标") or {}
    return {
        "acc": m.get("acc"),
        "time": info.get("算法参数网格搜索总耗时秒", info.get("算法约简平均每折耗时秒")),
        "avg_len": info.get("平均约简子集长度"),
    }


def recompute_ranks(
    algos: list[str],
    row: dict[str, dict[str, object | None]],
) -> dict[str, int]:
    """与 results_io.save_full_and_splits 一致：(-acc, avg_len 升序)，竞赛名次。"""
    sortable: list[tuple[str, float, float]] = []
    for name in algos:
        cell = row.get(name, {})
        acc = cell.get("acc")
        if acc is None:
            continue
        avg_len = cell.get("avg_len")
        al = float(avg_len) if avg_len is not None else float("inf")
        sortable.append((name, float(acc), al))
    sortable.sort(key=lambda x: (-x[1], x[2]))

    name_to_rank: dict[str, int] = {}
    last_acc: float | None = None
    last_rank = 0
    for idx, (name, acc, _avg_len) in enumerate(sortable, start=1):
        if last_acc is None or acc != last_acc:
            last_rank = idx
            last_acc = acc
        name_to_rank[name] = last_rank
    return name_to_rank


def build_analysis_arpdmf_only(root_pdmf: Path, datasets: list[str]) -> list[str]:
    """ARPDMF 参数表：扫 patch 目录下各 `arpdmf_by_params.json`。"""
    out: list[str] = ["### 2. ARPDMF 参数（δ, k, μ）\n"]
    ap_files = sorted(root_pdmf.glob("*/arpdmf_by_params.json"))
    if not ap_files:
        out.append("未找到 `arpdmf_by_params.json`。\n")
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
        out.append("无有效参数行。\n")
        return out

    ranked: list[tuple[float, int, tuple[float, int, str]]] = []
    for k, vs in bucket.items():
        ranked.append((sum(vs) / len(vs), len(vs), k))
    ranked.sort(key=lambda x: (-x[0], -x[1]))

    n_ds = len(datasets)
    out.append(
        f"数据来自 **`{root_pdmf.name}`** 下各 `arpdmf_by_params.json`（与上表 ARPDMF 列一致）。"
        f"本批共 **{n_ds}** 个数据集。\n"
    )
    out.append("| 排名 | δ | k | μ | 平均 acc | 样本数 |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for i, (mean_acc, cnt, ktuple) in enumerate(ranked[:12], 1):
        d, kk, mu = ktuple
        out.append(f"| {i} | {d:g} | {kk} | {mu!r} | {mean_acc:.4f} | {cnt} |")
    out.append("\n")
    return out


def _parse_patch_algos(s: str) -> list[str]:
    parts = [x.strip() for x in s.replace(";", ",").split(",")]
    return [x for x in parts if x]


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 baseline + patch 目录中指定算法，生成汇总 md")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_REPO_ROOT / "outputs" / "归一化",
        help="含全算法 summary 的目录（默认 outputs/归一化）",
    )
    parser.add_argument(
        "--pdmf",
        type=Path,
        default=_REPO_ROOT / "outputs" / "归一化_pdmf",
        help="重跑部分算法的 summary 目录（默认 outputs/归一化_pdmf）",
    )
    parser.add_argument(
        "--patch-algos",
        type=str,
        default="ARPDMF",
        help="从 --pdmf 取数的算法名，逗号分隔，例如 ARPDMF 或 ARPDMF,MFNMI",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 md（默认 <pdmf>/汇总_约简算法对比.md）",
    )
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()

    base = args.baseline if args.baseline.is_absolute() else (_REPO_ROOT / args.baseline).resolve()
    pdmf = args.pdmf if args.pdmf.is_absolute() else (_REPO_ROOT / args.pdmf).resolve()
    out_path = args.out or (pdmf / "汇总_约简算法对比.md")
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()

    patch_algos = set(_parse_patch_algos(args.patch_algos))
    if not patch_algos:
        print("--patch-algos 不能为空")
        return

    base_sets = {p.parent.name for p in base.glob("*/summary.json")}
    pdmf_sets = {p.parent.name for p in pdmf.glob("*/summary.json")}
    datasets = sorted(base_sets & pdmf_sets)
    if not datasets:
        print("两目录无共同含 summary.json 的数据集子目录")
        return

    # 算法列：以 baseline 中任一份 summary 的键为准
    ref_base = base / datasets[0] / "summary.json"
    algos = sorted(get_summary_block(load_summary(ref_base)).keys())
    for pa in patch_algos:
        if pa not in algos:
            algos.append(pa)
    algos.sort()

    # data[algo][ds] = {acc, time, avg_len, rank}
    data: dict[str, dict[str, dict]] = {a: {} for a in algos}

    for ds in datasets:
        bpath = base / ds / "summary.json"
        ppath = pdmf / ds / "summary.json"
        base_block = get_summary_block(load_summary(bpath))
        pdmf_block = get_summary_block(load_summary(ppath))

        row_flat: dict[str, dict[str, object | None]] = {}
        for algo in algos:
            if algo in patch_algos:
                info = pdmf_block.get(algo) or base_block.get(algo)
            else:
                info = base_block.get(algo)
            cell = extract_cell(info)
            row_flat[algo] = cell

        ranks = recompute_ranks(algos, row_flat)
        for algo in algos:
            c = dict(row_flat[algo])
            c["rank"] = ranks.get(algo)
            data[algo][ds] = c

    patch_list = ", ".join(f"`{x}`" for x in sorted(patch_algos))
    lines: list[str] = []
    lines.append(
        f"# 归一化实验汇总（合并：{patch_list} 来自 `{pdmf.name}`，其余来自 `{base.name}`）\n"
    )
    lines.append(
        f"- **其余算法**：`{base.name}` 下各数据集 `summary.json`（先前全量跑）。\n"
        f"- **{patch_list}**：`{pdmf.name}` 下各数据集 `summary.json`（重跑）。\n"
        "- **rank**：按合并后各算法 **平均 acc** 重算（acc 降序；同 acc 时平均子集长度更短优先；名次规则同主流程「竞赛排名」）。\n"
        "- 指标口径与 `summary.json` 一致（多模型平均 acc 等）。\n"
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
        avg_row = ["平均结果"]
        for a in algos:
            vals: list[float] = []
            for ds in datasets:
                v = data.get(a, {}).get(ds, {}).get(key)
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
    make_table("准确率排名 rank（1 为最好，合并后重算）", "rank", 0)
    make_table("算法约简总耗时（秒）", "time", 3)
    make_table("平均约简子集长度", "avg_len", 3)

    if not args.no_analysis:
        lines.append("## 简要分析\n")
        means: list[tuple[str, float]] = []
        for a in algos:
            vals = [
                float(data[a][ds]["acc"])
                for ds in datasets
                if ds in data.get(a, {}) and data[a][ds].get("acc") is not None
            ]
            if vals:
                means.append((a, sum(vals) / len(vals)))
        means.sort(key=lambda x: -x[1])
        lines.append("### 1. 约简算法（跨数据集平均 acc，合并口径）\n")
        for i, (a, m) in enumerate(means, 1):
            lines.append(f"{i}. **{a}**：平均 acc ≈ {m:.4f}")
        lines.append(
            f"\n说明：**{patch_list}** 来自 `{pdmf.name}` 新结果，其余算法来自 `{base.name}` 旧结果，"
            "平均 acc 仅作横向参考，非同一批次同跑。\n"
        )
        if "ARPDMF" in patch_algos:
            lines.extend(build_analysis_arpdmf_only(pdmf, datasets))

    lines.append("---\n*由 `diagnose/merge_guiyi_pdmf_aggregate_md.py` 自动生成。*\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
