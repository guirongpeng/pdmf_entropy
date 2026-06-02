"""Demo script: run all attribute reduction algorithms in `frs.attribute_reduction`.

Usage (PowerShell):
python .\frs\run_all_reductions_demo.py

Optional:
python .\frs\run_all_reductions_demo.py --csv ".\frs\datas\Absenteeism_two.csv" --target "Absenteeism time in hours"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    # Run as package module from parent directory: `python -m frs.run_all_reductions_demo`
    from frs.attribute_reduction import (
        prepare_dataframe,
        prepare_mfren_fold,
        prepare_fsfrmi_fold,
        reduce_mfren,
        reduce_fsfrmi,
        reduce_fnrs,
        reduce_frar,
        reduce_iarfcie,
        reduce_mfigi,
    )
except ModuleNotFoundError:
    # Run directly inside this directory: `python .\run_all_reductions_demo.py`
    from attribute_reduction import (
        prepare_dataframe,
        prepare_mfren_fold,
        prepare_fsfrmi_fold,
        reduce_mfren,
        reduce_fsfrmi,
        reduce_fnrs,
        reduce_frar,
        reduce_iarfcie,
        reduce_mfigi,
    )


def _idx_to_names(indices, columns):
    return [columns[i] for i in indices]


def run_all(csv_path: Path, target_name: str):
    # 1) Load / normalize data
    data = prepare_dataframe(file_path=str(csv_path), target_name=target_name)
    train_index = list(range(data.shape[0]))

    # 2) MFREN
    _, _, mfren_fsr_obj, mfren_metric, columns = prepare_mfren_fold(
        data=data,
        train_index=train_index,
        target_name=target_name,
        scaler=1,
        columns_nominal=[],
    )
    mfren_red = reduce_mfren(columns, mfren_metric, mfren_fsr_obj)

    # 3) FSFrMI
    _, _, fsfrmi_fsr_obj, fsfrmi_metric, columns2 = prepare_fsfrmi_fold(
        data=data,
        train_index=train_index,
        target_name=target_name,
        scaler=1,
        columns_nominal=[],
    )
    fsfrmi_red = reduce_fsfrmi(columns2, fsfrmi_metric, fsfrmi_fsr_obj)

    # 4) FNRS (returns index subsets)
    fnrs_red_idx = reduce_fnrs(data=data, train_index=train_index, target_name=target_name)
    fnrs_red_idx = fnrs_red_idx[0] if fnrs_red_idx else []
    fnrs_red = _idx_to_names(fnrs_red_idx, columns)

    # 5) FRAR (returns one index subset wrapped by list)
    frar_red_idx = reduce_frar(data=data, train_index=train_index, target_name=target_name)
    frar_red_idx = frar_red_idx[0] if frar_red_idx else []
    frar_red = _idx_to_names(frar_red_idx, columns)

    # 6) IARFCIE
    iarfcie_red = reduce_iarfcie(data=data, target_name=target_name, threshold=0.2)

    # 7) MFIGI
    mfigi_red = reduce_mfigi(data=data, target_name=target_name, w=0.005)

    result = {
        "dataset": str(csv_path),
        "target_name": target_name,
        "n_samples": int(data.shape[0]),
        "n_features": int(len(columns)),
        "reductions": {
            "FNRS": fnrs_red,
            "FRAR": frar_red,
            "FSFrMI": fsfrmi_red,
            "IARFCIE": iarfcie_red,
            "MFIGI": mfigi_red,
            # "MFREN": mfren_red,
        },
    }
    return result


def main():
    default_csv = Path(__file__).resolve().parent / "datas" / "Absenteeism_two.csv"
    parser = argparse.ArgumentParser(description="Run all attribute reduction algorithms in @frs.")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="CSV path.")
    parser.add_argument(
        "--target",
        type=str,
        default="Absenteeism time in hours",
        help="Target column name.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output json path. If omitted, only print.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    result = run_all(csv_path=csv_path, target_name=args.target)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

# base
# python .\run_all_reductions_demo.py --out .\datas\reduction_result.json