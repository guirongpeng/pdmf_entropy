# diagnose

统计、诊断、一次性实验脚本，与主流程（`run_full_selection_pipeline.py`、`config.py` 等）解耦；在仓库根目录执行时请加前缀 `diagnose/`。

| 脚本 | 作用 |
|------|------|
| `aggregate_outputs_summary_md.py` | 汇总 `outputs/.../summary.json` 为 Markdown（四表 + 简要分析 + ARPDMF 参数 Top）；仅含 `候选子集总数>0` 的算法列；`--root outputs/归一化` 等；`--no-analysis` 仅输出表格；生成前二次读盘校验 |
| `merge_guiyi_pdmf_aggregate_md.py` | 合并 baseline（默认 `outputs/归一化`）与 patch 目录中指定算法（如 `--patch-algos ARPDMF,MFNMI`），生成 `汇总_约简算法对比.md` |
| `analyze_arpdmf_by_params.py` | 跨数据集分析 `arpdmf_by_params.json` 参数效果 |
| `diagnose_constant_features.py` | 折内常数特征诊断 |
| `run_all_reductions_demo.py` | 约简算法演示 |
