"""深度レイヤーのクラスタリングモジュール。

このモジュールは、深度マップを離散的なレイヤーに分割するための
クラスタリング機能を提供します。最適なクラスタ数の自動探索や、
深度値の階層化を行います。
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from shadowbox.config.settings import ClusteringSettings


class LayerClustererProtocol(Protocol):
    """レイヤークラスタラーのプロトコル（DIインターフェース）。

    このプロトコルを実装することで、新しいクラスタリング手法を
    追加できます。
    """

    def find_optimal_k(self, depth_map: NDArray[np.float32]) -> int:
        """最適なクラスタ数を探索。

        Args:
            depth_map: 深度マップ (shape: H, W)。

        Returns:
            最適なクラスタ数 k。
        """
        ...

    def cluster(
        self,
        depth_map: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """深度マップをk個のレイヤーにクラスタリング。

        Args:
            depth_map: 深度マップ (shape: H, W)。
            k: クラスタ数（レイヤー数）。

        Returns:
            (labels, centroids) のタプル:
            - labels: 各ピクセルのレイヤーインデックス (shape: H, W)
            - centroids: ソート済みセントロイド (shape: k,)
        """
        ...


@dataclass
class ClusteringResult:
    """クラスタリング結果を格納するデータクラス。

    Attributes:
        labels: 各ピクセルのレイヤーインデックス。
            0が最も手前、k-1が最も奥。
        centroids: 各レイヤーのセントロイド（深度値）。
            前から後ろへソート済み。
        k: レイヤー数。
    """

    labels: NDArray[np.int32]
    centroids: NDArray[np.float32]
    k: int


class KMeansLayerClusterer:
    """K-Meansベースの深度レイヤークラスタラー。

    深度値をK-Meansでクラスタリングし、シャドーボックスの
    レイヤーを生成します。最適なk値はシルエット分析または
    エルボー法で自動探索できます。

    Attributes:
        settings: クラスタリングの設定。

    Example:
        >>> settings = ClusteringSettings(min_k=3, max_k=8)
        >>> clusterer = KMeansLayerClusterer(settings)
        >>> k = clusterer.find_optimal_k(depth_map)
        >>> labels, centroids = clusterer.cluster(depth_map, k)
    """

    def __init__(self, settings: ClusteringSettings) -> None:
        """クラスタラーを初期化。

        Args:
            settings: クラスタリングの設定。
        """
        self._settings = settings

    def find_optimal_k(self, depth_map: NDArray[np.float32]) -> int:
        """最適なクラスタ数を探索。

        設定に基づいてシルエット分析またはエルボー法で
        最適なk値を探索します。

        Args:
            depth_map: 深度マップ (shape: H, W)。

        Returns:
            最適なクラスタ数 k。
        """
        # 1次元に平坦化
        flat_depth = depth_map.flatten().reshape(-1, 1)

        # サンプリング（大きな画像の場合）
        flat_depth = self._subsample_if_needed(flat_depth)

        if self._settings.method == "silhouette":
            return self._find_optimal_k_silhouette(flat_depth)
        else:
            return self._find_optimal_k_elbow(flat_depth)

    def _subsample_if_needed(
        self,
        data: NDArray[np.float32],
        max_samples: int = 10000,
    ) -> NDArray[np.float32]:
        """必要に応じてデータをサブサンプリング。

        計算効率のため、大きなデータセットの場合は
        ランダムサンプリングを行います。

        Args:
            data: サンプリング対象データ。
            max_samples: 最大サンプル数。

        Returns:
            サンプリング後のデータ。
        """
        if len(data) <= max_samples:
            return data

        rng = np.random.default_rng(self._settings.random_state)
        indices = rng.choice(len(data), max_samples, replace=False)
        return data[indices]

    def _find_optimal_k_silhouette(self, data: NDArray[np.float32]) -> int:
        """シルエット分析で最適なkを探索。

        シルエットスコアが最大となるk値を返します。

        Args:
            data: 1次元にリシェイプされた深度データ。

        Returns:
            最適なk値。
        """
        k_range = range(self._settings.min_k, self._settings.max_k + 1)
        scores = []

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self._settings.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)

        # 最大スコアのインデックスを取得
        best_idx = int(np.argmax(scores))
        return k_range[best_idx]

    def _find_optimal_k_elbow(self, data: NDArray[np.float32]) -> int:
        """エルボー法で最適なkを探索。

        イナーシャの変化率が最も大きく変わる点（エルボー）を
        検出します。

        Args:
            data: 1次元にリシェイプされた深度データ。

        Returns:
            最適なk値。
        """
        k_range = range(self._settings.min_k, self._settings.max_k + 1)
        inertias = []

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self._settings.random_state,
                n_init=10,
            )
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        # 2階差分でエルボーポイントを検出
        inertias_arr = np.array(inertias)
        diffs = np.diff(inertias_arr)
        diffs2 = np.diff(diffs)

        # 2階差分が最大となる点がエルボー
        if len(diffs2) == 0:
            return self._settings.min_k

        elbow_idx = int(np.argmax(diffs2)) + 1  # diffによるオフセット補正
        return k_range[elbow_idx]

    def cluster(
        self,
        depth_map: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """深度マップをk個のレイヤーにクラスタリング。

        Args:
            depth_map: 深度マップ (shape: H, W)。
            k: クラスタ数（レイヤー数）。

        Returns:
            (labels, centroids) のタプル:
            - labels: レイヤーインデックス (shape: H, W)。
              0が最も手前、k-1が最も奥。
            - centroids: ソート済みセントロイド (shape: k,)。
        """
        original_shape = depth_map.shape
        flat_depth = depth_map.flatten().reshape(-1, 1)

        # K-Meansでクラスタリング
        kmeans = KMeans(
            n_clusters=k,
            random_state=self._settings.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(flat_depth)
        centroids = kmeans.cluster_centers_.flatten()

        # セントロイドを深度順（前→後）にソート
        # 深度値が小さい = 手前
        sorted_indices = np.argsort(centroids)
        sorted_centroids = centroids[sorted_indices]

        # ラベルを再マッピング（0が最も手前になるように）
        label_mapping = {old: new for new, old in enumerate(sorted_indices)}
        remapped_labels = np.vectorize(label_mapping.get)(labels)

        return (
            remapped_labels.reshape(original_shape).astype(np.int32),
            sorted_centroids.astype(np.float32),
        )

    def cluster_with_result(
        self,
        depth_map: NDArray[np.float32],
        k: int | None = None,
    ) -> ClusteringResult:
        """深度マップをクラスタリングし、結果オブジェクトを返す。

        kがNoneの場合は最適なkを自動探索します。

        Args:
            depth_map: 深度マップ (shape: H, W)。
            k: クラスタ数。Noneの場合は自動探索。

        Returns:
            ClusteringResultオブジェクト。
        """
        if k is None:
            k = self.find_optimal_k(depth_map)

        labels, centroids = self.cluster(depth_map, k)

        return ClusteringResult(
            labels=labels,
            centroids=centroids,
            k=k,
        )
