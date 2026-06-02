from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.tri import Triangulation


def _load_param_surface(json_path: Path) -> pd.DataFrame:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    if not {"delta", "theta", "acc_mean"}.issubset(df.columns):
        raise ValueError("mfnmi_param_surface.json 缺少必要字段：delta、theta、acc_mean")
    df["delta"] = df["delta"].astype(float)
    df["theta"] = df["theta"].astype(float)
    df["acc_mean"] = df["acc_mean"].astype(float)
    return df


def _is_full_grid(df: pd.DataFrame) -> bool:
    deltas = np.sort(df["delta"].unique())
    thetas = np.sort(df["theta"].unique())
    return len(deltas) * len(thetas) == len(df)


def _build_mesh_from_full_grid(df: pd.DataFrame):
    pivot = df.pivot(index="delta", columns="theta", values="acc_mean").sort_index().sort_index(axis=1)
    x = pivot.index.values
    y = pivot.columns.values
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = pivot.values
    return X, Y, Z


def _format_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.3f}"))


def plot_surface_from_json(
    json_path: Path,
    out_dir: Path,
    title: str = "MFNMI Parameter Surface",
    cmap_name: str = "viridis",
    dpi: int = 300,
    elev: float = 35.0,
    azim: float = -60.0,
    include_contour: bool = True,
) -> Path:
    """
    从 mfnmi_param_surface.json 生成三维图，仅保存 PNG，颜色条标题为 acc_mean，不标注最大值红点。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "mfnmi_param_surface.png"

    df = _load_param_surface(Path(json_path))

    cmap = cm.get_cmap(cmap_name)
    fig = plt.figure(figsize=(7.5, 5.8), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if _is_full_grid(df):
        X, Y, Z = _build_mesh_from_full_grid(df)
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", antialiased=True, linewidth=0)
        if include_contour:
            ax.contour(
                X, Y, Z,
                zdir="z",
                offset=Z.min() - (Z.max() - Z.min()) * 0.15,
                cmap=cmap, levels=12, linewidths=0.6, antialiased=True,
            )
    else:
        x = df["delta"].to_numpy()
        y = df["theta"].to_numpy()
        z = df["acc_mean"].to_numpy()
        tri = Triangulation(x, y)
        surf = ax.plot_trisurf(tri, z, cmap=cmap, linewidth=0.2, antialiased=True)
        if include_contour:
            ax.tricontour(
                tri, z,
                zdir="z",
                offset=z.min() - (z.max() - z.min()) * 0.15,
                cmap=cmap, levels=12, linewidths=0.6,
            )

    cb = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=18, pad=0.08)
    cb.set_label("acc_mean", fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("delta", labelpad=8)
    ax.set_ylabel("theta", labelpad=8)
    ax.set_zlabel("acc_mean", labelpad=8)
    ax.set_title(title, pad=12)
    _format_axis(ax)

    plt.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(description="Plot MFNMI (delta, theta) -> acc_mean 3D surface (PNG only)")
    default_json = Path(__file__).resolve().parents[1] / "outputs" / "abs_3" / "sub_info" / "mfnmi_param_surface.json"
    parser.add_argument("--json", type=str, default=str(default_json), help="mfnmi_param_surface.json 路径")
    parser.add_argument("--outdir", type=str, default="", help="输出目录（默认与 JSON 同目录）")
    parser.add_argument("--title", type=str, default="MFNMI Parameter Surface", help="图标题")
    parser.add_argument("--dpi", type=int, default=300, help="图片DPI")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib 色图名")
    parser.add_argument("--no-contour", action="store_true", help="不画投影等高线")
    parser.add_argument("--view", type=str, default="", help="视角，格式为 elev,azim 例如 35,-60")
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"未找到JSON文件: {json_path}")
    out_dir = Path(args.outdir).resolve() if args.outdir else json_path.parent

    elev, azim = 35.0, -60.0
    if args.view:
        try:
            parts = [float(s.strip()) for s in args.view.split(",")]
            if len(parts) == 2:
                elev, azim = parts[0], parts[1]
        except Exception:
            pass

    png_path = plot_surface_from_json(
        json_path=json_path,
        out_dir=out_dir,
        title=args.title,
        cmap_name=args.cmap,
        dpi=args.dpi,
        elev=elev,
        azim=azim,
        include_contour=(not args.no_contour),
    )
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()


#   python plot_mfnmi_surface.py --json "outputs/abs_3/sub_info/mfnmi_param_surface.json" --outdir "outputs/abs_3/sub_info" --view "35,-60" --cmap "viridis" --dpi 300 --no-contour
'''
- **--view "elev,azim"**: 三维视角（单位：度）
  - **elev**（俯仰角）：相机相对水平面的仰角。常用 20–60；0 为平视，90 为顶视。
  - **azim**（方位角）：绕垂直轴旋转角。范围约 -180～180；负值顺时针、正值逆时针。
  - 论文常用：`"30,-60"`、`"35,-60"`、`"40,-45"`。

- **--cmap "viridis"**: 颜色映射方案（色图）
  - 连续型常用：`viridis`（默认，感知均匀）、`plasma`、`inferno`、`magma`、`cividis`、`turbo`。
  - 若有明确“高/低”对比强调，`inferno`/`plasma`对高值更醒目；色盲友好推荐 `viridis`/`cividis`。
  - 建议论文：`viridis` 或 `cividis`。

- **--dpi 300**: 输出图片分辨率（每英寸点数）
  - 网页/预览：96–150
  - 论文/打印：300（常用）、600（更清晰但文件更大）

- **--no-contour**: 关闭投影等高线
  - 加上此开关=不画等高线；不加则默认会在底部投影一层等高线辅助读图。
  - 若曲面较平滑、颜色梯度清晰，可用 `--no-contour` 保持简洁；否则建议保留等高线提升可读性。
'''