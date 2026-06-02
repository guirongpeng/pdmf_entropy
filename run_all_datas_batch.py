"""
批量执行：对 datas 下每个数据集调用 run_full_selection_pipeline.py。

目标列与论文目录「汇总有效数据集\\MFRSE_data_reduction.ipynb」中 DATAS 的 D_name 一致。
输出目录：outputs/<outname>/ ，默认 outname 为 CSV 主文件名（不含扩展名）。

用法（在项目根目录）:
  python run_all_datas_batch.py
  python run_all_datas_batch.py --dry-run
  python run_all_datas_batch.py --only wdbc.csv

当 config.get_enabled_reductions() 含 ARPDMF 时，每个数据集流水线成功后自动生成
arpdmf_param_surface.png（读取同目录 arpdmf_by_params.json）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "run_full_selection_pipeline.py"
PLOT_ARPDMF = ROOT / "plot_arpdmf_surface.py"
AGGREGATE_MD = ROOT / "diagnose" / "aggregate_outputs_summary_md.py"
DATAS_DIR = ROOT / "datas"
# 各数据集写入 outputs/<OUTPUT_PREFIX>/<数据集名>/
DEFAULT_OUTPUT_PREFIX = "test_datas_2"

def _arpdmf_enabled() -> bool:
    from config import get_enabled_reductions

    return "ARPDMF" in get_enabled_reductions()


def _arpdmf_plot_cmd(out_dir: Path, dataset_stem: str) -> list[str]:
    json_path = out_dir / "arpdmf_by_params.json"
    return [
        sys.executable,
        str(PLOT_ARPDMF),
        "--json",
        str(json_path),
        "--title",
        f"ARPDMF ({dataset_stem})",
    ]


def _run_arpdmf_plot(out_dir: Path, dataset_stem: str) -> int:
    """流水线成功后生成 ARPDMF 三维图；依赖 arpdmf_by_params.json。"""
    json_path = out_dir / "arpdmf_by_params.json"
    if not json_path.is_file():
        print(f"[跳过绘图] 未找到: {json_path}", file=sys.stderr)
        return 0
    cmd = _arpdmf_plot_cmd(out_dir, dataset_stem)
    print(f"ARPDMF 绘图: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


# 文件名 -> 目标列（Y）
DATASET_TARGETS: dict[str, str] = {

    "Toxicity.csv":"class",
    "Gas_sensor_array_under_flow_modulation.csv": "class",
    "Period-Changer.csv":"class",
    "DBWorld_e_mails.csv": "class",
    "MicroMass_pure.csv": "class",

}


def main() -> None:
    parser = argparse.ArgumentParser(description="对 datas 中各数据集批量跑完整选优流程")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不真正运行")
    parser.add_argument("--only", type=str, default="", help="只处理某一个文件名，例如 wdbc.csv")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"输出目录前缀，结果在 outputs/<前缀>/<数据集名>/（默认 {DEFAULT_OUTPUT_PREFIX}）",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="批量结束后不运行 aggregate_outputs_summary_md.py",
    )
    args = parser.parse_args()

    if not PIPELINE.is_file():
        print(f"未找到脚本: {PIPELINE}", file=sys.stderr)
        sys.exit(1)
    plot_arpdmf = _arpdmf_enabled()
    if plot_arpdmf and not PLOT_ARPDMF.is_file():
        print(f"未找到绘图脚本: {PLOT_ARPDMF}", file=sys.stderr)
        sys.exit(1)
    if not args.no_aggregate and not args.dry_run and not AGGREGATE_MD.is_file():
        print(f"未找到汇总脚本: {AGGREGATE_MD}", file=sys.stderr)
        sys.exit(1)

    items = sorted(DATASET_TARGETS.items(), key=lambda x: x[0])
    if args.only.strip():
        only = args.only.strip()
        if only not in DATASET_TARGETS:
            print(f"未知数据集: {only}，可选: {list(DATASET_TARGETS.keys())}", file=sys.stderr)
            sys.exit(1)
        items = [(only, DATASET_TARGETS[only])]

    failures: list[tuple[str, int]] = []
    plot_failures: list[tuple[str, int]] = []

    for fname, target in items:
        csv_path = DATAS_DIR / fname
        if not csv_path.is_file():
            print(f"[跳过] 文件不存在: {csv_path}")
            continue

        dataset_stem = Path(fname).stem
        outname = f"{args.output_prefix.strip('/')}/{dataset_stem}"
        out_dir = ROOT / "outputs" / outname

        cmd = [
            sys.executable,
            str(PIPELINE),
            "--csv",
            str(csv_path),
            "--target",
            target,
            "--outname",
            outname,
        ]

        print("=" * 60)
        print(f"数据集: {fname}")
        print(f"目标列: {target}")
        print(f"命令: {' '.join(cmd)}")
        if plot_arpdmf:
            print(f"[dry-run] ARPDMF 绘图: {' '.join(_arpdmf_plot_cmd(out_dir, dataset_stem))}")
        if args.dry_run:
            continue

        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            failures.append((fname, r.returncode))
            print(f"[失败] {fname} 退出码 {r.returncode}", file=sys.stderr)
            continue

        if plot_arpdmf:
            r_plot = _run_arpdmf_plot(out_dir, dataset_stem)
            if r_plot != 0:
                plot_failures.append((fname, r_plot))
                print(f"[失败] ARPDMF 绘图 {fname} 退出码 {r_plot}", file=sys.stderr)

    if args.dry_run:
        if not args.no_aggregate:
            output_root = ROOT / "outputs" / args.output_prefix.strip("/")
            agg_out = output_root / "汇总_约简算法对比.md"
            agg_cmd = [
                sys.executable,
                str(AGGREGATE_MD),
                "--root",
                str(output_root),
                "--out",
                str(agg_out),
            ]
            print("=" * 60)
            print(f"[dry-run] 汇总命令: {' '.join(agg_cmd)}")
        return

    print("=" * 60)
    if failures:
        print(f"数据集流程完成，失败 {len(failures)} 个: {failures}", file=sys.stderr)
    else:
        print("全部数据集流程成功。")
    if plot_failures:
        print(f"ARPDMF 绘图失败 {len(plot_failures)} 个: {plot_failures}", file=sys.stderr)

    if not args.no_aggregate:
        output_root = ROOT / "outputs" / args.output_prefix.strip("/")
        agg_out = output_root / "汇总_约简算法对比.md"
        agg_cmd = [
            sys.executable,
            str(AGGREGATE_MD),
            "--root",
            str(output_root),
            "--out",
            str(agg_out),
        ]
        print("=" * 60)
        print("生成汇总 Markdown …")
        print(f"命令: {' '.join(agg_cmd)}")
        r_agg = subprocess.run(agg_cmd, cwd=str(ROOT))
        if r_agg.returncode != 0:
            print(f"[失败] 汇总脚本退出码 {r_agg.returncode}", file=sys.stderr)
            sys.exit(r_agg.returncode if not failures else 1)
        # print(f"已写入: {agg_out}")

    if failures or plot_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()



# 1.	python run_all_datas_batch.py
#       a)	跑全部数据集：output/xxx多个数据集
# 2.	python run_full_selection_pipeline.py --csv datas\flag.csv --target column_7
#       a)	跑指定数据集：output/xxx数据集
# 3.	结果汇总（批量结束后自动执行，也可用 --no-aggregate 跳过）
#       a)	python diagnose/aggregate_outputs_summary_md.py --root outputs/test_datas --out outputs/test_datas/汇总_约简算法对比.md

# python diagnose/aggregate_outputs_summary_md.py --root outputs/test_datas_all_1_opt --out outputs/test_datas_all_1_opt/汇总_约简算法对比.md
# python diagnose/merge_guiyi_pdmf_aggregate_md.py --baseline outputs/test_datas_2 --pdmf outputs/test_datas_2_归一化 --patch-algos ARPDMF --out outputs/test_datas_2_归一化/汇总_约简算法对比_合并baseline.md
