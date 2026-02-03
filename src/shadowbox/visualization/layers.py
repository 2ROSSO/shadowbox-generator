"""レイヤープレビュー可視化モジュール。

このモジュールは、クラスタリング結果のレイヤーを
可視化する機能を提供します。
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def create_layer_preview(
    image: NDArray[np.uint8],
    labels: NDArray[np.int32],
    centroids: NDArray[np.float32],
    figsize: tuple[int, int] | None = None,
    columns: int = 4,
) -> tuple[Figure, NDArray[Axes]]:
    """各レイヤーの画像プレビューを作成。

    クラスタリング結果の各レイヤーに属するピクセルのみを
    表示した画像を並べて表示します。

    Args:
        image: オリジナル画像。shape (H, W, 3) のuint8配列。
        labels: ピクセルごとのレイヤーラベル。shape (H, W) のint32配列。
        centroids: 各レイヤーの深度中心値。shape (k,) のfloat32配列。
        figsize: 図のサイズ。Noneの場合は自動計算。
        columns: 列数。

    Returns:
        matplotlibのFigureとAxes配列のタプル。

    Example:
        >>> fig, axes = create_layer_preview(image, labels, centroids)
        >>> plt.show()
    """
    k = len(centroids)
    rows = (k + columns - 1) // columns

    # figsize自動計算
    if figsize is None:
        figsize = (columns * 3, rows * 3)

    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    # squeeze=Falseにより常に2次元配列が返される

    # 深度でソート（手前から奥へ）
    sorted_indices = np.argsort(centroids)

    for idx, layer_idx in enumerate(sorted_indices):
        row = idx // columns
        col = idx % columns
        ax = axes[row, col]

        # このレイヤーに属するピクセルのみを抽出
        mask = labels == layer_idx
        layer_image = np.zeros_like(image)
        layer_image[mask] = image[mask]

        # 背景をグレーに
        layer_image[~mask] = [200, 200, 200]

        ax.imshow(layer_image)
        ax.set_title(f"レイヤー {idx}\n(深度: {centroids[layer_idx]:.2f})")
        ax.axis("off")

    # 余分なサブプロットを非表示
    for idx in range(k, rows * columns):
        row = idx // columns
        col = idx % columns
        axes[row, col].axis("off")

    fig.suptitle("レイヤー分解プレビュー（手前→奥）")
    fig.tight_layout()

    return fig, axes


def create_layer_mask_preview(
    labels: NDArray[np.int32],
    centroids: NDArray[np.float32],
    figsize: tuple[int, int] | None = None,
    columns: int = 4,
) -> tuple[Figure, NDArray[Axes]]:
    """各レイヤーのマスクプレビューを作成。

    各レイヤーに属するピクセルの位置を白黒マスクとして
    表示します。

    Args:
        labels: ピクセルごとのレイヤーラベル。shape (H, W) のint32配列。
        centroids: 各レイヤーの深度中心値。shape (k,) のfloat32配列。
        figsize: 図のサイズ。Noneの場合は自動計算。
        columns: 列数。

    Returns:
        matplotlibのFigureとAxes配列のタプル。

    Example:
        >>> fig, axes = create_layer_mask_preview(labels, centroids)
        >>> plt.show()
    """
    k = len(centroids)
    rows = (k + columns - 1) // columns

    if figsize is None:
        figsize = (columns * 3, rows * 3)

    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    # squeeze=Falseにより常に2次元配列が返される

    # 深度でソート
    sorted_indices = np.argsort(centroids)

    for idx, layer_idx in enumerate(sorted_indices):
        row = idx // columns
        col = idx % columns
        ax = axes[row, col]

        # マスク作成
        mask = (labels == layer_idx).astype(np.uint8) * 255

        ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        pixel_count = np.sum(labels == layer_idx)
        ax.set_title(f"レイヤー {idx}\n({pixel_count:,} px)")
        ax.axis("off")

    # 余分なサブプロットを非表示
    for idx in range(k, rows * columns):
        row = idx // columns
        col = idx % columns
        axes[row, col].axis("off")

    fig.suptitle("レイヤーマスク（白=該当ピクセル）")
    fig.tight_layout()

    return fig, axes


def create_labeled_image(
    image: NDArray[np.uint8],
    labels: NDArray[np.int32],
    centroids: NDArray[np.float32],
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.4,
) -> tuple[Figure, Axes]:
    """レイヤーラベルを色分けして元画像に重ねて表示。

    各レイヤーを異なる色で塗り分け、元画像に半透明で
    重ねて表示します。

    Args:
        image: オリジナル画像。shape (H, W, 3) のuint8配列。
        labels: ピクセルごとのレイヤーラベル。shape (H, W) のint32配列。
        centroids: 各レイヤーの深度中心値。shape (k,) のfloat32配列。
        figsize: 図のサイズ。
        alpha: レイヤー色の透明度。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, ax = create_labeled_image(image, labels, centroids)
        >>> plt.show()
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 元画像を表示
    ax.imshow(image)

    # レイヤーラベルをカラーマップで表示
    k = len(centroids)
    cmap = plt.colormaps.get_cmap("tab10").resampled(k)

    # 深度でソートしたラベルに変換
    sorted_indices = np.argsort(centroids)
    index_map = {old: new for new, old in enumerate(sorted_indices)}

    # ラベルを再マッピング
    sorted_labels = np.vectorize(index_map.get)(labels)

    im = ax.imshow(sorted_labels, cmap=cmap, alpha=alpha, vmin=0, vmax=k - 1)

    ax.set_title("レイヤー分類結果")
    ax.axis("off")

    # カラーバーにレイヤー番号を表示
    cbar = fig.colorbar(im, ax=ax, ticks=range(k))
    cbar.ax.set_yticklabels([f"レイヤー {i}" for i in range(k)])

    fig.tight_layout()
    return fig, ax


def create_depth_layer_comparison(
    depth_map: NDArray[np.float32],
    labels: NDArray[np.int32],
    centroids: NDArray[np.float32],
    figsize: tuple[int, int] = (14, 5),
) -> tuple[Figure, Axes]:
    """深度マップとレイヤー分類の比較表示。

    元の深度マップと、クラスタリングによる離散化された
    深度マップを並べて表示します。

    Args:
        depth_map: 深度マップ。shape (H, W) のfloat32配列。
        labels: ピクセルごとのレイヤーラベル。shape (H, W) のint32配列。
        centroids: 各レイヤーの深度中心値。shape (k,) のfloat32配列。
        figsize: 図のサイズ。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, axes = create_depth_layer_comparison(depth_map, labels, centroids)
        >>> plt.show()
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 元の深度マップ
    im0 = axes[0].imshow(depth_map, cmap="viridis")
    axes[0].set_title("元の深度マップ")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], label="深度")

    # 離散化された深度マップ（各ピクセルをcentroid値で置き換え）
    discretized_depth = centroids[labels]
    im1 = axes[1].imshow(discretized_depth, cmap="viridis")
    axes[1].set_title(f"離散化深度（{len(centroids)}レイヤー）")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], label="深度")

    # レイヤーラベル
    k = len(centroids)
    sorted_indices = np.argsort(centroids)
    index_map = {old: new for new, old in enumerate(sorted_indices)}
    sorted_labels = np.vectorize(index_map.get)(labels)

    cmap = plt.colormaps.get_cmap("tab10").resampled(k)
    im2 = axes[2].imshow(sorted_labels, cmap=cmap, vmin=0, vmax=k - 1)
    axes[2].set_title("レイヤー番号")
    axes[2].axis("off")
    cbar = fig.colorbar(im2, ax=axes[2], ticks=range(k))
    cbar.ax.set_yticklabels([f"L{i}" for i in range(k)])

    fig.suptitle("深度とレイヤーの比較")
    fig.tight_layout()

    return fig, axes


def create_stacked_layer_view(
    image: NDArray[np.uint8],
    labels: NDArray[np.int32],
    centroids: NDArray[np.float32],
    figsize: tuple[int, int] = (12, 8),
    offset: float = 0.1,
) -> tuple[Figure, Axes]:
    """レイヤーを疑似3D的に積み重ねた表示。

    各レイヤーを少しずつずらして重ねて表示することで、
    3D的な階層構造を2Dで表現します。

    Args:
        image: オリジナル画像。shape (H, W, 3) のuint8配列。
        labels: ピクセルごとのレイヤーラベル。shape (H, W) のint32配列。
        centroids: 各レイヤーの深度中心値。shape (k,) のfloat32配列。
        figsize: 図のサイズ。
        offset: レイヤー間のオフセット比率。

    Returns:
        matplotlibのFigureとAxesのタプル。

    Example:
        >>> fig, ax = create_stacked_layer_view(image, labels, centroids)
        >>> plt.show()
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    k = len(centroids)
    h, w = labels.shape

    # 深度でソート（奥から手前の順）
    sorted_indices = np.argsort(centroids)[::-1]

    # キャンバスを作成
    canvas_h = int(h * (1 + offset * (k - 1)))
    canvas_w = int(w * (1 + offset * (k - 1)))
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240  # 薄いグレー背景

    for idx, layer_idx in enumerate(sorted_indices):
        # オフセット計算（奥のレイヤーほど右下にずれる）
        y_offset = int(idx * h * offset)
        x_offset = int(idx * w * offset)

        # マスク作成
        mask = labels == layer_idx

        # レイヤー画像を作成（マスク外は透明扱い）
        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    canvas[y + y_offset, x + x_offset] = image[y, x]

    ax.imshow(canvas)
    ax.set_title("レイヤー積み重ね表示（奥→手前）")
    ax.axis("off")

    fig.tight_layout()
    return fig, ax


def show_clustering_summary(
    image: NDArray[np.uint8],
    depth_map: NDArray[np.float32],
    labels: NDArray[np.int32],
    centroids: NDArray[np.float32],
    figsize: tuple[int, int] = (16, 10),
) -> tuple[Figure, NDArray[Axes]]:
    """クラスタリング結果の総合サマリーを表示。

    元画像、深度マップ、レイヤー分類、各レイヤーのプレビューを
    まとめて表示します。

    Args:
        image: オリジナル画像。shape (H, W, 3) のuint8配列。
        depth_map: 深度マップ。shape (H, W) のfloat32配列。
        labels: ピクセルごとのレイヤーラベル。shape (H, W) のint32配列。
        centroids: 各レイヤーの深度中心値。shape (k,) のfloat32配列。
        figsize: 図のサイズ。

    Returns:
        matplotlibのFigureとAxes配列のタプル。

    Example:
        >>> fig, axes = show_clustering_summary(image, depth_map, labels, centroids)
        >>> plt.show()
    """
    k = len(centroids)

    # レイアウト: 上段に元画像・深度・ラベル、下段にレイヤープレビュー
    fig = plt.figure(figsize=figsize)

    # 上段のグリッド
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)

    # 元画像
    ax1.imshow(image)
    ax1.set_title("元画像")
    ax1.axis("off")

    # 深度マップ
    im2 = ax2.imshow(depth_map, cmap="viridis")
    ax2.set_title("深度マップ")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # レイヤー分類
    sorted_indices = np.argsort(centroids)
    index_map = {old: new for new, old in enumerate(sorted_indices)}
    # カードフレーム用ラベル(-1)を最前面(k)として扱う
    index_map[-1] = k
    sorted_labels = np.vectorize(index_map.get)(labels).astype(np.int32)

    # カードフレームがある場合は色数を+1
    has_frame_label = -1 in labels
    num_colors = k + 1 if has_frame_label else k
    cmap = plt.colormaps.get_cmap("tab10").resampled(num_colors)
    ax3.imshow(sorted_labels, cmap=cmap, vmin=0, vmax=num_colors - 1)
    title = f"レイヤー分類（k={k}）"
    if has_frame_label:
        title += " + フレーム"
    ax3.set_title(title)
    ax3.axis("off")

    # 下段にレイヤープレビュー
    layer_cols = min(k, 6)
    for idx, layer_idx in enumerate(sorted_indices[:layer_cols]):
        ax = fig.add_subplot(2, layer_cols, layer_cols + idx + 1)

        mask = labels == layer_idx
        layer_image = np.zeros_like(image)
        layer_image[mask] = image[mask]
        layer_image[~mask] = [200, 200, 200]

        ax.imshow(layer_image)
        ax.set_title(f"L{idx} (d={centroids[layer_idx]:.2f})")
        ax.axis("off")

    fig.suptitle("シャドーボックス レイヤー分解サマリー", fontsize=14)
    fig.tight_layout()

    return fig, fig.axes
