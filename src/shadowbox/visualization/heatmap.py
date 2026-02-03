"""深度ヒートマップ可視化モジュール。

このモジュールは、深度マップをカラーヒートマップとして
可視化する機能を提供します。
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def create_depth_heatmap(
    depth_map: NDArray[np.float32],
    original_image: NDArray[np.uint8] | None = None,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (12, 5),
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """深度マップをヒートマップとして可視化。

    オリジナル画像が指定された場合は、元画像と深度マップを
    並べて表示します。

    Args:
        depth_map: 深度マップ。shape (H, W) のfloat32配列。
            値は0.0（近い）から1.0（遠い）。
        original_image: オリジナル画像（オプション）。
            shape (H, W, 3) のuint8配列。
        cmap: matplotlibのカラーマップ名。
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'など。
        figsize: 図のサイズ (幅, 高さ)。
        title: 図のタイトル（オプション）。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, ax = create_depth_heatmap(depth_map, original_image)
        >>> plt.show()
    """
    # オリジナル画像がある場合は2つ並べて表示
    if original_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_image, ax_depth = axes

        # オリジナル画像を表示
        ax_image.imshow(original_image)
        ax_image.set_title("元画像")
        ax_image.axis("off")

        # 深度マップを表示
        im = ax_depth.imshow(depth_map, cmap=cmap)
        ax_depth.set_title("深度マップ")
        ax_depth.axis("off")

        # カラーバーを追加
        fig.colorbar(im, ax=ax_depth, label="深度 (0=近い, 1=遠い)")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig, axes

    # 深度マップのみの場合
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))

    im = ax.imshow(depth_map, cmap=cmap)
    ax.set_title(title if title else "深度マップ")
    ax.axis("off")

    fig.colorbar(im, ax=ax, label="深度 (0=近い, 1=遠い)")
    fig.tight_layout()

    return fig, ax


def create_depth_overlay(
    original_image: NDArray[np.uint8],
    depth_map: NDArray[np.float32],
    alpha: float = 0.5,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (8, 6),
) -> tuple[Figure, Axes]:
    """深度マップを元画像に重ねて表示。

    元画像の上に深度マップを半透明で重ねて表示します。

    Args:
        original_image: オリジナル画像。shape (H, W, 3) のuint8配列。
        depth_map: 深度マップ。shape (H, W) のfloat32配列。
        alpha: 深度マップの透明度（0.0〜1.0）。
        cmap: matplotlibのカラーマップ名。
        figsize: 図のサイズ。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, ax = create_depth_overlay(image, depth_map, alpha=0.6)
        >>> plt.show()
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 元画像を表示
    ax.imshow(original_image)

    # 深度マップを半透明で重ねる
    im = ax.imshow(depth_map, cmap=cmap, alpha=alpha)
    ax.axis("off")
    ax.set_title("深度オーバーレイ")

    fig.colorbar(im, ax=ax, label="深度 (0=近い, 1=遠い)")
    fig.tight_layout()

    return fig, ax


def create_depth_histogram(
    depth_map: NDArray[np.float32],
    bins: int = 50,
    figsize: tuple[int, int] = (10, 4),
) -> tuple[Figure, Axes]:
    """深度値のヒストグラムを作成。

    深度マップの深度値分布を可視化します。

    Args:
        depth_map: 深度マップ。shape (H, W) のfloat32配列。
        bins: ヒストグラムのビン数。
        figsize: 図のサイズ。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, ax = create_depth_histogram(depth_map)
        >>> plt.show()
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 深度値をフラットにしてヒストグラム作成
    depth_values = depth_map.flatten()

    ax.hist(depth_values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("深度値")
    ax.set_ylabel("ピクセル数")
    ax.set_title("深度分布ヒストグラム")
    ax.set_xlim(0, 1)

    # 統計情報を追加
    mean_depth = np.mean(depth_values)
    std_depth = np.std(depth_values)
    ax.axvline(mean_depth, color="red", linestyle="--", label=f"平均: {mean_depth:.3f}")
    ax.legend()

    # テキストで統計情報を表示
    stats_text = f"平均: {mean_depth:.3f}\n標準偏差: {std_depth:.3f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()
    return fig, ax


def create_depth_contour(
    depth_map: NDArray[np.float32],
    levels: int = 10,
    original_image: NDArray[np.uint8] | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[Figure, Axes]:
    """深度マップの等高線図を作成。

    深度マップを等高線で可視化します。オリジナル画像が
    指定された場合は、その上に等高線を重ねます。

    Args:
        depth_map: 深度マップ。shape (H, W) のfloat32配列。
        levels: 等高線の数。
        original_image: オリジナル画像（オプション）。
        figsize: 図のサイズ。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, ax = create_depth_contour(depth_map, levels=15)
        >>> plt.show()
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 背景画像がある場合は先に表示
    if original_image is not None:
        ax.imshow(original_image)

    # 等高線を描画
    contour_colors = "white" if original_image is not None else None
    contour = ax.contour(depth_map, levels=levels, colors=contour_colors)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    if original_image is None:
        # 背景画像がない場合は深度マップを塗りつぶしで表示
        ax.contourf(depth_map, levels=levels, cmap="viridis", alpha=0.7)

    ax.set_title("深度等高線")
    ax.axis("off")

    fig.tight_layout()
    return fig, ax
