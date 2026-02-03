"""TripoSR 3Dメッシュ生成モジュール。

Stability AIのTripoSRを使用して、単一画像から
3Dメッシュを直接生成する機能を提供します。

Note:
    このモジュールを使用するには、triposr依存関係が必要です:
    pip install shadowbox[triposr]
"""

from typing import TYPE_CHECKING, Optional

from shadowbox.triposr.settings import TripoSRSettings

if TYPE_CHECKING:
    from shadowbox.config.settings import RenderSettings

__all__ = ["TripoSRSettings", "create_triposr_generator", "create_triposr_pipeline"]


def create_triposr_generator(settings: TripoSRSettings):
    """TripoSR生成器を作成。

    Args:
        settings: TripoSR設定。

    Returns:
        TripoSRGenerator: 設定済みの生成器。

    Raises:
        ImportError: TripoSR依存関係がインストールされていない場合。
    """
    from shadowbox.triposr.generator import TripoSRGenerator
    return TripoSRGenerator(settings)


def create_triposr_pipeline(
    settings: TripoSRSettings,
    render_settings: Optional["RenderSettings"] = None,
):
    """TripoSRパイプラインを作成。

    Args:
        settings: TripoSR設定。
        render_settings: レンダリング設定（フレーム生成に使用）。

    Returns:
        TripoSRPipeline: 設定済みのパイプライン。

    Raises:
        ImportError: TripoSR依存関係がインストールされていない場合。
    """
    from shadowbox.triposr.pipeline import TripoSRPipeline
    return TripoSRPipeline(settings, render_settings)
