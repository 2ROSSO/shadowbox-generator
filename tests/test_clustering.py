"""クラスタリングモジュールのテスト。

このモジュールは、KMeansLayerClustererと関連機能の
ユニットテストを提供します。
"""

import numpy as np
import pytest

from shadowbox.config import ClusteringSettings
from shadowbox.core.clustering import ClusteringResult, KMeansLayerClusterer


class TestKMeansLayerClusterer:
    """KMeansLayerClustererのテスト。"""

    def test_init(self) -> None:
        """初期化をテスト。"""
        settings = ClusteringSettings()
        clusterer = KMeansLayerClusterer(settings)

        assert clusterer._settings == settings

    def test_cluster_basic(self, sample_depth_map: np.ndarray) -> None:
        """基本的なクラスタリングをテスト。"""
        settings = ClusteringSettings(min_k=3, max_k=5, random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        k = 3
        labels, centroids = clusterer.cluster(sample_depth_map, k)

        # 形状の確認
        assert labels.shape == sample_depth_map.shape
        assert len(centroids) == k

        # 型の確認
        assert labels.dtype == np.int32
        assert centroids.dtype == np.float32

        # ラベルの範囲確認
        assert labels.min() >= 0
        assert labels.max() < k

        # セントロイドがソートされていることを確認
        assert np.all(np.diff(centroids) >= 0)

    def test_cluster_labels_consistent(self) -> None:
        """ラベルがセントロイドと一貫していることをテスト。"""
        # 明確に分離された3つの領域を持つ深度マップ
        depth_map = np.zeros((30, 30), dtype=np.float32)
        depth_map[0:10, :] = 0.1   # 手前（近い）
        depth_map[10:20, :] = 0.5  # 中間
        depth_map[20:30, :] = 0.9  # 奥（遠い）

        settings = ClusteringSettings(random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        labels, centroids = clusterer.cluster(depth_map, k=3)

        # 手前の領域（深度値が小さい）はラベル0
        assert labels[5, 15] == 0  # 上部の中央
        # 奥の領域（深度値が大きい）はラベル2
        assert labels[25, 15] == 2  # 下部の中央

    def test_find_optimal_k_silhouette(self, sample_depth_map: np.ndarray) -> None:
        """シルエット法による最適k探索をテスト。"""
        settings = ClusteringSettings(
            min_k=2,
            max_k=5,
            method="silhouette",
            random_state=42,
        )
        clusterer = KMeansLayerClusterer(settings)

        k = clusterer.find_optimal_k(sample_depth_map)

        # kが範囲内にあることを確認
        assert settings.min_k <= k <= settings.max_k

    def test_find_optimal_k_elbow(self, sample_depth_map: np.ndarray) -> None:
        """エルボー法による最適k探索をテスト。"""
        settings = ClusteringSettings(
            min_k=2,
            max_k=5,
            method="elbow",
            random_state=42,
        )
        clusterer = KMeansLayerClusterer(settings)

        k = clusterer.find_optimal_k(sample_depth_map)

        # kが範囲内にあることを確認
        assert settings.min_k <= k <= settings.max_k

    def test_subsample_small_data(self) -> None:
        """小さいデータはサブサンプリングされないことをテスト。"""
        settings = ClusteringSettings()
        clusterer = KMeansLayerClusterer(settings)

        small_data = np.random.rand(100, 1).astype(np.float32)
        result = clusterer._subsample_if_needed(small_data, max_samples=1000)

        # データがそのまま返される
        assert len(result) == 100

    def test_subsample_large_data(self) -> None:
        """大きいデータがサブサンプリングされることをテスト。"""
        settings = ClusteringSettings(random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        large_data = np.random.rand(50000, 1).astype(np.float32)
        result = clusterer._subsample_if_needed(large_data, max_samples=10000)

        # サンプリングされてサイズが減少
        assert len(result) == 10000

    def test_cluster_with_result(self, sample_depth_map: np.ndarray) -> None:
        """cluster_with_resultメソッドをテスト。"""
        settings = ClusteringSettings(min_k=3, max_k=5, random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        # 明示的にk=3を指定
        result = clusterer.cluster_with_result(sample_depth_map, k=3)

        assert isinstance(result, ClusteringResult)
        assert result.k == 3
        assert result.labels.shape == sample_depth_map.shape
        assert len(result.centroids) == 3

    def test_cluster_with_result_auto_k(self, sample_depth_map: np.ndarray) -> None:
        """cluster_with_resultの自動k探索をテスト。"""
        settings = ClusteringSettings(min_k=2, max_k=5, random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        # k=Noneで自動探索
        result = clusterer.cluster_with_result(sample_depth_map, k=None)

        assert isinstance(result, ClusteringResult)
        assert settings.min_k <= result.k <= settings.max_k


class TestClusteringResult:
    """ClusteringResultデータクラスのテスト。"""

    def test_create(self) -> None:
        """ClusteringResultの作成をテスト。"""
        labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
        centroids = np.array([0.1, 0.5, 0.9], dtype=np.float32)

        result = ClusteringResult(labels=labels, centroids=centroids, k=3)

        assert result.k == 3
        assert np.array_equal(result.labels, labels)
        assert np.array_equal(result.centroids, centroids)


class TestClusteringEdgeCases:
    """クラスタリングのエッジケーステスト。"""

    def test_uniform_depth_map(self) -> None:
        """均一な深度マップのクラスタリングをテスト。"""
        # 全て同じ値
        uniform_depth = np.full((50, 50), 0.5, dtype=np.float32)

        settings = ClusteringSettings(min_k=2, max_k=3, random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        # クラスタリングは実行可能（ただし意味のある分離はない）
        labels, centroids = clusterer.cluster(uniform_depth, k=2)

        assert labels.shape == uniform_depth.shape
        assert len(centroids) == 2

    def test_two_layer_depth_map(self) -> None:
        """2層の明確な深度マップをテスト。"""
        depth_map = np.zeros((20, 20), dtype=np.float32)
        depth_map[:10, :] = 0.2  # 前半分
        depth_map[10:, :] = 0.8  # 後半分

        settings = ClusteringSettings(random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        labels, centroids = clusterer.cluster(depth_map, k=2)

        # 2つの明確に分離されたクラスタ
        assert len(np.unique(labels)) == 2

        # セントロイドは約0.2と0.8
        assert centroids[0] == pytest.approx(0.2, abs=0.1)
        assert centroids[1] == pytest.approx(0.8, abs=0.1)

    def test_gradient_depth_map(self) -> None:
        """グラデーション深度マップのクラスタリングをテスト。"""
        # 上から下へのグラデーション
        height = 100
        gradient = np.linspace(0, 1, height, dtype=np.float32)
        depth_map = np.tile(gradient[:, np.newaxis], (1, 50))

        settings = ClusteringSettings(min_k=3, max_k=5, random_state=42)
        clusterer = KMeansLayerClusterer(settings)

        k = 4
        labels, centroids = clusterer.cluster(depth_map, k)

        # 4つのレイヤーに分割
        assert len(np.unique(labels)) == k

        # セントロイドは均等に分布（約0.125, 0.375, 0.625, 0.875）
        expected = np.array([0.125, 0.375, 0.625, 0.875])
        for actual, exp in zip(centroids, expected):
            assert actual == pytest.approx(exp, abs=0.15)
