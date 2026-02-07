"""設定モジュールのテスト。

このモジュールは、BoundingBox、CardTemplate、各種Settings、
YAMLConfigLoaderのユニットテストを提供します。
"""

import tempfile
from pathlib import Path

import pytest

from shadowbox.config import (
    BoundingBox,
    CardTemplate,
    ClusteringSettings,
    DepthEstimationSettings,
    RenderSettings,
    ShadowboxSettings,
    YAMLConfigLoader,
)


class TestBoundingBox:
    """BoundingBoxデータクラスのテスト。"""

    def test_create_valid_bbox(self) -> None:
        """有効なバウンディングボックスの作成をテスト。"""
        bbox = BoundingBox(x=10, y=20, width=100, height=200)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 200

    def test_bbox_right_bottom(self) -> None:
        """rightとbottomプロパティをテスト。"""
        bbox = BoundingBox(x=10, y=20, width=100, height=200)

        assert bbox.right == 110
        assert bbox.bottom == 220

    def test_bbox_to_tuple(self) -> None:
        """to_tupleメソッドをテスト。"""
        bbox = BoundingBox(x=10, y=20, width=100, height=200)

        assert bbox.to_tuple() == (10, 20, 100, 200)

    def test_bbox_to_crop_box(self) -> None:
        """to_crop_boxメソッド（PIL用）をテスト。"""
        bbox = BoundingBox(x=10, y=20, width=100, height=200)

        assert bbox.to_crop_box() == (10, 20, 110, 220)

    def test_bbox_invalid_width(self) -> None:
        """負の幅でエラーが発生することをテスト。"""
        with pytest.raises(ValueError, match="幅は正の値"):
            BoundingBox(x=10, y=20, width=-100, height=200)

    def test_bbox_invalid_height(self) -> None:
        """ゼロの高さでエラーが発生することをテスト。"""
        with pytest.raises(ValueError, match="高さは正の値"):
            BoundingBox(x=10, y=20, width=100, height=0)

    def test_bbox_invalid_x(self) -> None:
        """負のxでエラーが発生することをテスト。"""
        with pytest.raises(ValueError, match="xは0以上"):
            BoundingBox(x=-10, y=20, width=100, height=200)

    def test_bbox_invalid_y(self) -> None:
        """負のyでエラーが発生することをテスト。"""
        with pytest.raises(ValueError, match="yは0以上"):
            BoundingBox(x=10, y=-20, width=100, height=200)


class TestCardTemplate:
    """CardTemplateデータクラスのテスト。"""

    def test_create_valid_template(self) -> None:
        """有効なテンプレートの作成をテスト。"""
        bbox = BoundingBox(x=42, y=72, width=660, height=488)
        template = CardTemplate(
            name="pokemon_standard",
            game="pokemon",
            illustration_area=bbox,
            card_width=744,
            card_height=1039,
        )

        assert template.name == "pokemon_standard"
        assert template.game == "pokemon"
        assert template.frame_margin == 10  # デフォルト値

    def test_template_with_description(self) -> None:
        """説明付きテンプレートをテスト。"""
        bbox = BoundingBox(x=42, y=72, width=660, height=488)
        template = CardTemplate(
            name="pokemon_standard",
            game="pokemon",
            illustration_area=bbox,
            card_width=744,
            card_height=1039,
            description="標準的なポケモンカードのテンプレート",
        )

        assert template.description == "標準的なポケモンカードのテンプレート"

    def test_template_invalid_card_width(self) -> None:
        """無効なカード幅でエラーが発生することをテスト。"""
        bbox = BoundingBox(x=42, y=72, width=660, height=488)

        with pytest.raises(ValueError, match="card_widthは正の値"):
            CardTemplate(
                name="test",
                game="test",
                illustration_area=bbox,
                card_width=0,
                card_height=1039,
            )

    def test_template_illustration_exceeds_card(self) -> None:
        """イラスト領域がカードを超える場合にエラーが発生することをテスト。"""
        bbox = BoundingBox(x=42, y=72, width=800, height=488)  # 幅が大きすぎる

        with pytest.raises(ValueError, match="イラスト領域がカードの幅を超えて"):
            CardTemplate(
                name="test",
                game="test",
                illustration_area=bbox,
                card_width=744,
                card_height=1039,
            )

    def test_template_to_dict(self) -> None:
        """辞書へのシリアライズをテスト。"""
        bbox = BoundingBox(x=42, y=72, width=660, height=488)
        template = CardTemplate(
            name="pokemon_standard",
            game="pokemon",
            illustration_area=bbox,
            card_width=744,
            card_height=1039,
        )

        data = template.to_dict()

        assert data["name"] == "pokemon_standard"
        assert data["illustration_area"]["x"] == 42

    def test_template_from_dict(self) -> None:
        """辞書からのデシリアライズをテスト。"""
        data = {
            "name": "pokemon_standard",
            "game": "pokemon",
            "card_width": 744,
            "card_height": 1039,
            "illustration_area": {
                "x": 42,
                "y": 72,
                "width": 660,
                "height": 488,
            },
        }

        template = CardTemplate.from_dict(data)

        assert template.name == "pokemon_standard"
        assert template.illustration_area.x == 42


class TestSettings:
    """設定データクラスのテスト。"""

    def test_depth_estimation_defaults(self) -> None:
        """DepthEstimationSettingsのデフォルト値をテスト。"""
        settings = DepthEstimationSettings()

        assert settings.model_type == "depth_anything"
        assert "Depth-Anything-V2-Small" in settings.model_name
        assert settings.device == "auto"

    def test_clustering_defaults(self) -> None:
        """ClusteringSettingsのデフォルト値をテスト。"""
        settings = ClusteringSettings()

        assert settings.min_k == 3
        assert settings.max_k == 10
        assert settings.method == "silhouette"

    def test_render_defaults(self) -> None:
        """RenderSettingsのデフォルト値をテスト。"""
        settings = RenderSettings()

        assert settings.layer_thickness == 0.1
        assert settings.frame_z == 0.0

    def test_shadowbox_settings_defaults(self) -> None:
        """ShadowboxSettingsのデフォルト値をテスト。"""
        settings = ShadowboxSettings()

        assert isinstance(settings.depth, DepthEstimationSettings)
        assert isinstance(settings.clustering, ClusteringSettings)
        assert isinstance(settings.render, RenderSettings)
        assert settings.templates_dir == Path("data/templates")

    def test_shadowbox_settings_string_path(self) -> None:
        """文字列パスがPathに変換されることをテスト。"""
        settings = ShadowboxSettings(templates_dir="custom/path")  # type: ignore

        assert isinstance(settings.templates_dir, Path)
        assert settings.templates_dir == Path("custom/path")


class TestYAMLConfigLoader:
    """YAMLConfigLoaderのテスト。"""

    def test_list_templates_empty(self) -> None:
        """空のディレクトリでテンプレート一覧を取得するテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = YAMLConfigLoader(Path(tmpdir))

            assert loader.list_templates() == []

    def test_save_and_load_template(self) -> None:
        """テンプレートの保存と読み込みをテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = YAMLConfigLoader(Path(tmpdir))

            # テンプレートを作成して保存
            bbox = BoundingBox(x=42, y=72, width=660, height=488)
            template = CardTemplate(
                name="test_template",
                game="pokemon",
                illustration_area=bbox,
                card_width=744,
                card_height=1039,
            )
            loader.save_template(template)

            # 読み込んで検証
            loaded = loader.load_template("test_template")

            assert loaded.name == "test_template"
            assert loaded.illustration_area.x == 42

    def test_list_templates(self) -> None:
        """保存済みテンプレートの一覧取得をテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = YAMLConfigLoader(Path(tmpdir))

            # 2つのテンプレートを保存
            bbox = BoundingBox(x=0, y=0, width=100, height=100)
            for name in ["template1", "template2"]:
                template = CardTemplate(
                    name=name,
                    game="test",
                    illustration_area=bbox,
                    card_width=200,
                    card_height=300,
                )
                loader.save_template(template)

            templates = loader.list_templates()

            assert len(templates) == 2
            assert "template1" in templates
            assert "template2" in templates

    def test_load_nonexistent_template(self) -> None:
        """存在しないテンプレートの読み込みでエラーが発生することをテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = YAMLConfigLoader(Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                loader.load_template("nonexistent")

    def test_template_exists(self) -> None:
        """template_existsメソッドをテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = YAMLConfigLoader(Path(tmpdir))

            assert not loader.template_exists("test")

            bbox = BoundingBox(x=0, y=0, width=100, height=100)
            template = CardTemplate(
                name="test",
                game="test",
                illustration_area=bbox,
                card_width=200,
                card_height=300,
            )
            loader.save_template(template)

            assert loader.template_exists("test")

    def test_delete_template(self) -> None:
        """テンプレートの削除をテスト。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = YAMLConfigLoader(Path(tmpdir))

            bbox = BoundingBox(x=0, y=0, width=100, height=100)
            template = CardTemplate(
                name="to_delete",
                game="test",
                illustration_area=bbox,
                card_width=200,
                card_height=300,
            )
            loader.save_template(template)

            assert loader.template_exists("to_delete")
            assert loader.delete_template("to_delete")
            assert not loader.template_exists("to_delete")

            # 存在しないテンプレートの削除はFalseを返す
            assert not loader.delete_template("to_delete")
