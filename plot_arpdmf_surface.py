"""
从 arpdmf_by_params.json 绘制 ARPDMF 参数面：delta × k → 指标（默认 acc）。

情况 A：μ 固定（单值或 --mu 指定），Z 轴为 平均指标（默认 ×100 显示为百分比，如 0.968→96.80）。

用法（项目根目录）:
  python plot_arpdmf_surface.py --json outputs/test_datas_2_guiyi/breast-cancer-wisconsin/arpdmf_by_params.json
  python plot_arpdmf_surface.py --json outputs/.../arpdmf_by_params.json --metric f1_macro --mu B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
from matplotlib.tri import Triangulation


def _load_arpdmf_surface(
    json_path: Path,
    metric: str = "acc",
    mu_method: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    解析 arpdmf_by_params.json → DataFrame(delta, k, mu_method, value)。
    返回 (df, 实际使用的 mu_method)。
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = data.get("按参数组合")
    if not rows:
        raise ValueError(f"{json_path} 中「按参数组合」为空或缺失")

    records: list[dict] = []
    for row in rows:
        p = row.get("参数") or {}
        m = row.get("平均指标") or {}
        if metric not in m:
            continue
        records.append(
            {
                "delta": float(p["delta"]),
                "k": int(p["k"]),
                "mu_method": str(p.get("mu_method", "")),
                "value": float(m[metric]),
            }
        )

    if not records:
        raise ValueError(f"{json_path} 的「平均指标」中未找到指标: {metric}")

    df = pd.DataFrame(records)
    mus = sorted(df["mu_method"].unique())
    if mu_method is None:
        if len(mus) == 1:
            mu_method = mus[0]
        else:
            raise ValueError(
                f"存在多种 mu_method: {mus}，请用 --mu 指定其一（情况 A：固定 μ）"
            )
    else:
        mu_method = str(mu_method)
        if mu_method not in mus:
            raise ValueError(f"未找到 mu_method={mu_method!r}，可选: {mus}")

    df = df[df["mu_method"] == mu_method].copy()
    if df.empty:
        raise ValueError(f"过滤 mu_method={mu_method!r} 后无数据")

    return df, mu_method


def _is_full_grid(df: pd.DataFrame) -> bool:
    deltas = np.sort(df["delta"].unique())
    ks = np.sort(df["k"].unique())
    return len(deltas) * len(ks) == len(df)


def _build_mesh_from_full_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        df.pivot(index="delta", columns="k", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    x = pivot.index.values.astype(float)
    y = pivot.columns.values.astype(float)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = pivot.values.astype(float)
    return X, Y, Z


def _z_axis_label(metric: str, as_percent: bool) -> str:
    if as_percent:
        return f"{metric} (%)"
    return metric


def _format_axis(ax, *, as_percent: bool = True) -> None:
    """刻度用普通小数；关闭 3D 轴 offset 文本（避免出现 1e-5+9.68e-1 一类标注）。"""
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.3f}"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _pos: f"{int(v)}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
        )
    )
    if as_percent:
        ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.2f}"))
    else:
        ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.4f}"))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "offsetText"):
            axis.offsetText.set_visible(False)
        fmt = axis.get_major_formatter()
        if hasattr(fmt, "set_useOffset"):
            fmt.set_useOffset(False)
        if hasattr(fmt, "set_scientific"):
            fmt.set_scientific(False)


def plot_surface_from_json(
    json_path: Path,
    out_dir: Path,
    metric: str = "acc",
    mu_method: str | None = None,
    title: str = "",
    cmap_name: str = "viridis",
    dpi: int = 300,
    elev: float = 35.0,
    azim: float = -60.0,
    include_contour: bool = True,
    out_name: str = "arpdmf_param_surface.png",
    as_percent: bool = True,
) -> Path:
    """从 arpdmf_by_params.json 生成 delta–k–metric 三维曲面图（PNG）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / out_name

    df, mu_used = _load_arpdmf_surface(json_path, metric=metric, mu_method=mu_method)
    if as_percent:
        df = df.copy()
        df["value"] = df["value"] * 100.0

    if not title:
        dataset_hint = json_path.parent.name
        title = f"ARPDMF ({dataset_hint}, μ={mu_used})"

    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(cmap_name)
    fig = plt.figure(figsize=(7.5, 5.8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if _is_full_grid(df):
        X, Y, Z = _build_mesh_from_full_grid(df)
        surf = ax.plot_surface(
            X, Y, Z, cmap=cmap, edgecolor="none", antialiased=True, linewidth=0
        )
        if include_contour:
            z_span = float(Z.max() - Z.min()) or 1.0
            ax.contour(
                X,
                Y,
                Z,
                zdir="z",
                offset=float(Z.min()) - z_span * 0.15,
                cmap=cmap,
                levels=12,
                linewidths=0.6,
                antialiased=True,
            )
    else:
        x = df["delta"].to_numpy(dtype=float)
        y = df["k"].to_numpy(dtype=float)
        z = df["value"].to_numpy(dtype=float)
        tri = Triangulation(x, y)
        surf = ax.plot_trisurf(tri, z, cmap=cmap, linewidth=0.2, antialiased=True)
        if include_contour:
            z_span = float(z.max() - z.min()) or 1.0
            ax.tricontour(
                tri,
                z,
                zdir="z",
                offset=float(z.min()) - z_span * 0.15,
                cmap=cmap,
                levels=12,
                linewidths=0.6,
            )

    cb = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=18, pad=0.08)
    # 色条仅表示颜色深浅，不设 acc 等文字标签；刻度用普通小数
    cbar_fmt = ScalarFormatter(useOffset=False)
    cbar_fmt.set_scientific(False)
    cb.formatter = cbar_fmt
    cb.update_ticks()
    if hasattr(cb.ax, "yaxis") and hasattr(cb.ax.yaxis, "offsetText"):
        cb.ax.yaxis.offsetText.set_visible(False)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("delta", labelpad=8)
    ax.set_ylabel("k", labelpad=8)
    ax.set_zlabel(_z_axis_label(metric, as_percent), labelpad=8)
    if title:
        ax.set_title(title, pad=12)
    _format_axis(ax, as_percent=as_percent)

    plt.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ARPDMF (delta, k) -> metric 3D surface from arpdmf_by_params.json"
    )
    default_json = (
        Path(__file__).resolve().parent
        / "outputs"
        / "test_datas_2_guiyi"
        / "breast-cancer-wisconsin"
        / "arpdmf_by_params.json"
    )
    parser.add_argument("--json", type=str, default=str(default_json), help="arpdmf_by_params.json 路径")
    parser.add_argument("--outdir", type=str, default="", help="输出目录（默认与 JSON 同目录）")
    parser.add_argument("--metric", type=str, default="acc", help="平均指标键名，如 acc、f1_macro")
    parser.add_argument("--mu", type=str, default="", help="固定 mu_method；多种 μ 时必须指定")
    parser.add_argument("--title", type=str, default="", help="图标题")
    parser.add_argument("--out-name", type=str, default="arpdmf_param_surface.png", help="输出 PNG 文件名")
    parser.add_argument("--dpi", type=int, default=300, help="图片 DPI")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib 色图名")
    parser.add_argument("--no-contour", action="store_true", help="不画底部投影等高线")
    parser.add_argument(
        "--no-percent",
        action="store_true",
        help="Z 轴与色条保持 0–1 小数（默认将指标×100 显示为百分比，如 0.968→96.80）",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="",
        help="视角 elev,azim，例如 35,-60",
    )
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件: {json_path}")
    out_dir = Path(args.outdir).resolve() if args.outdir else json_path.parent

    elev, azim = 35.0, -60.0
    if args.view:
        try:
            parts = [float(s.strip()) for s in args.view.split(",")]
            if len(parts) == 2:
                elev, azim = parts[0], parts[1]
        except Exception:
            pass

    mu_arg = args.mu.strip() or None
    png_path = plot_surface_from_json(
        json_path=json_path,
        out_dir=out_dir,
        metric=args.metric.strip(),
        mu_method=mu_arg,
        title=args.title.strip(),
        cmap_name=args.cmap,
        dpi=args.dpi,
        elev=elev,
        azim=azim,
        include_contour=not args.no_contour,
        out_name=args.out_name,
        as_percent=not args.no_percent,
    )
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()

'''
# 指定数据与输出目录
python plot_arpdmf_surface.py \
  --json outputs/test_datas_2_guiyi/breast-cancer-wisconsin/arpdmf_by_params.json


# 其他指标 / 视角 / 色图
python plot_arpdmf_surface.py \
  --json outputs/test_datas_all_1_opt/spect-0/arpdmf_by_params.json \
  --metric f1_macro \
  --view "35,-60" \
  --cmap viridis \
  --dpi 300
'''