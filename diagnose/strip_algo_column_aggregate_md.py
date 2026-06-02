"""
从「汇总_约简算法对比.md」类文件中剔除指定算法列，并按剩余列重算：
- 准确率排名 rank（同 merge_guiyi：acc 降序，同 acc 则平均子集长度升序，竞赛名次）
- 各表「平均结果」行

用法:
  python diagnose/strip_algo_column_aggregate_md.py \\
    --input outputs/归一化_pdmf_1/汇总_约简算法对比.md \\
    --drop FSFrMI \\
    --out outputs/归一化_pdmf_1/汇总_约简算法对比_无FSFrMI.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_md_table(lines: list[str], start_idx: int) -> tuple[list[str] | None, list[list[str]], int]:
    """
    从 start_idx 起解析一张 pipe 表，返回 (header_cells, data_rows, next_line_index)。
    header 含首列「数据集」。
    """
    if start_idx >= len(lines):
        return None, [], start_idx
    line = lines[start_idx].strip()
    if not line.startswith("|"):
        return None, [], start_idx
    header = [c.strip() for c in line.split("|")[1:-1]]
    if len(header) < 2:
        return None, [], start_idx
    sep_idx = start_idx + 1
    if sep_idx >= len(lines) or "---" not in lines[sep_idx]:
        return None, [], start_idx
    rows: list[list[str]] = []
    i = sep_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.split("|")[1:-1]])
        i += 1
    return header, rows, i


def fmt_float(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def recompute_ranks(
    algos: list[str],
    acc_row: dict[str, float],
    len_row: dict[str, float],
) -> dict[str, int]:
    sortable: list[tuple[str, float, float]] = []
    for name in algos:
        if name not in acc_row:
            continue
        acc = acc_row[name]
        al = len_row.get(name, float("inf"))
        sortable.append((name, float(acc), float(al)))
    sortable.sort(key=lambda x: (-x[1], x[2]))
    name_to_rank: dict[str, int] = {}
    last_acc: float | None = None
    last_rank = 0
    for idx, (name, acc, _al) in enumerate(sortable, start=1):
        if last_acc is None or acc != last_acc:
            last_rank = idx
            last_acc = acc
        name_to_rank[name] = last_rank
    return name_to_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--drop", type=str, default="FSFrMI", help="要剔除的算法列名")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    in_path = args.input if args.input.is_absolute() else (_REPO_ROOT / args.input).resolve()
    out_path = args.out
    if out_path is None:
        out_path = in_path.parent / f"{in_path.stem}_无{args.drop}{in_path.suffix}"
    elif not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()

    text = in_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    acc_data: dict[str, dict[str, float]] = {}
    avg_len_data: dict[str, dict[str, float]] = {}
    time_data: dict[str, dict[str, float]] = {}

    i = 0
    current_metric: str | None = None
    while i < len(lines):
        if "## 平均准确率" in lines[i]:
            current_metric = "acc"
        elif "## 准确率排名" in lines[i]:
            current_metric = "rank"
        elif "## 平均约简子集长度" in lines[i]:
            current_metric = "avg_len"
        elif "## 算法约简总耗时" in lines[i]:
            current_metric = "time"
        elif lines[i].strip().startswith("|") and "数据集" in lines[i] and current_metric in (
            "acc",
            "avg_len",
            "time",
        ):
            hdr, rows, i = parse_md_table(lines, i)
            i -= 1
            if hdr and len(hdr) >= 2:
                algos_h = hdr[1:]
                for row in rows:
                    if len(row) != len(hdr):
                        continue
                    ds = row[0]
                    if ds == "平均结果":
                        continue
                    vals = row[1:]
                    for k, a in enumerate(algos_h):
                        if k >= len(vals):
                            continue
                        try:
                            v = float(vals[k])
                        except ValueError:
                            continue
                        if current_metric == "acc":
                            acc_data.setdefault(ds, {})[a] = v
                        elif current_metric == "avg_len":
                            avg_len_data.setdefault(ds, {})[a] = v
                        elif current_metric == "time":
                            time_data.setdefault(ds, {})[a] = v
        i += 1

    drop = args.drop.strip()
    if not acc_data:
        print("未能解析 acc 表，请检查输入 Markdown 格式")
        return

    algo_order: list[str] | None = None
    for line in lines:
        s = line.strip()
        if s.startswith("|") and "数据集" in s and drop in s:
            hdr = [c.strip() for c in s.split("|")[1:-1]]
            if len(hdr) >= 2:
                algo_order = hdr[1:]
                break
    if not algo_order:
        sample_ds = next(iter(acc_data.keys()))
        algo_order = list(acc_data[sample_ds].keys())
    if drop not in algo_order:
        print(f"列 {drop!r} 不在表头顺序中，现有: {algo_order}")
        return
    algos = [a for a in algo_order if a != drop]
    datasets = sorted(acc_data.keys())

    # 重算 rank
    rank_data: dict[str, dict[str, int]] = {}
    for ds in datasets:
        acc_row = {a: acc_data[ds][a] for a in algos if a in acc_data[ds]}
        len_row = {a: avg_len_data.get(ds, {}).get(a, float("inf")) for a in algos}
        r = recompute_ranks(algos, acc_row, len_row)
        rank_data[ds] = r

    def mean_col(metric: dict[str, dict[str, float]], algo: str) -> float:
        vs = [metric[ds][algo] for ds in datasets if ds in metric and algo in metric[ds]]
        return sum(vs) / len(vs) if vs else float("nan")

    def mean_rank(algo: str) -> float:
        vs = [float(rank_data[ds][algo]) for ds in datasets if ds in rank_data and algo in rank_data[ds]]
        return sum(vs) / len(vs) if vs else float("nan")

    def render_table(title: str, metric_key: str, nd: int) -> list[str]:
        out = [f"## {title}\n", "| 数据集 | " + " | ".join(algos) + " |", "| --- | " + " | ".join(["---"] * len(algos)) + " |"]
        for ds in datasets:
            row = [ds]
            for a in algos:
                if metric_key == "acc":
                    v = acc_data[ds].get(a)
                elif metric_key == "rank":
                    v = rank_data[ds].get(a)
                elif metric_key == "avg_len":
                    v = avg_len_data.get(ds, {}).get(a)
                else:
                    v = time_data.get(ds, {}).get(a)
                if v is None:
                    row.append("—")
                elif metric_key == "rank":
                    row.append(str(int(v)))
                else:
                    row.append(fmt_float(float(v), nd))
            out.append("| " + " | ".join(row) + " |")
        avg_r = ["平均结果"]
        for a in algos:
            if metric_key == "acc":
                avg_r.append(fmt_float(mean_col(acc_data, a), nd))
            elif metric_key == "rank":
                avg_r.append(f"{mean_rank(a):.2f}")
            elif metric_key == "avg_len":
                avg_r.append(fmt_float(mean_col(avg_len_data, a), nd))
            else:
                avg_r.append(fmt_float(mean_col(time_data, a), nd))
        out.append("| " + " | ".join(avg_r) + " |")
        out.append("")
        return out

    # 简要分析：按平均 acc 排序（不含 FSFrMI）
    means = [(a, mean_col(acc_data, a)) for a in algos]
    means.sort(key=lambda x: -x[1])

    new_lines: list[str] = []
    new_lines.append(
        f"# {in_path.stem}（已剔除 **{drop}** 列；rank 与「平均结果」已按剩余算法重算）\n"
    )
    new_lines.append(f"- 源文件：`{in_path.as_posix()}`\n")
    new_lines.append(f"- 剔除列：**{drop}**（表中不再出现，排名按其余算法重算）。\n")
    new_lines.append("")
    new_lines.extend(render_table("平均准确率 acc", "acc", 4))
    new_lines.extend(render_table("准确率排名 rank（1 为最好，剔除后重算）", "rank", 0))
    new_lines.extend(render_table("平均约简子集长度", "avg_len", 3))
    new_lines.extend(render_table("算法约简总耗时（秒）", "time", 3))

    new_lines.append("## 简要分析\n")
    new_lines.append("### 1. 约简算法（跨数据集平均 acc，剔除 FSFrMI 后）\n")
    for i, (a, m) in enumerate(means, 1):
        new_lines.append(f"{i}. **{a}**：平均 acc ≈ {m:.4f}")
    new_lines.append(
        "\n说明：ARPDMF 为 `归一化_pdmf` 新结果，其余算法为 `归一化` 旧结果，平均 acc 仅作横向参考，非同一批次同跑。"
        f" 本表已排除 **{drop}** 并重算 rank；算法排序与上表「平均结果」行一致。\n"
    )

    # 若原文有 ARPDMF 参数小节，整段附在末尾（从 ### 2. ARPDMF 到 --- 前）
    copy_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("### 2. ARPDMF"):
            copy_start = i
            break
    if copy_start is not None:
        copy_end = len(lines)
        for j in range(copy_start, len(lines)):
            if lines[j].strip() == "---":
                copy_end = j
                break
        new_lines.append("")
        new_lines.extend(lines[copy_start:copy_end])
        new_lines.append("")

    new_lines.append("---\n")
    new_lines.append(f"*由 `diagnose/strip_algo_column_aggregate_md.py` 从 `{in_path.name}` 生成。*\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
