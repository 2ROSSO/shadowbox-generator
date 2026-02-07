"""国際化 (i18n) モジュール。

UIテキストの英語/日本語切り替えを提供します。

Usage:
    from shadowbox.gui.i18n import tr, set_language
    label.setText(tr("tab.original"))
    set_language("ja")
"""

from __future__ import annotations

import json
from pathlib import Path

_current_language: str = "en"

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ---- Menu ----
    "menu.file": {"en": "File", "ja": "ファイル"},
    "menu.open": {"en": "Open Image...", "ja": "画像を開く..."},
    "menu.exit": {"en": "Exit", "ja": "終了"},
    "menu.language": {"en": "Language", "ja": "言語"},
    "menu.help": {"en": "Help", "ja": "ヘルプ"},
    "menu.about": {"en": "About", "ja": "About"},
    # ---- Status bar ----
    "status.load_image": {"en": "Load an image to begin", "ja": "画像を読み込んでください"},
    "status.loaded": {"en": "Loaded: {name}", "ja": "読み込み完了: {name}"},
    "status.region_selected": {
        "en": "Region selected: ({x}, {y}) {w}x{h}",
        "ja": "領域選択: ({x}, {y}) {w}x{h}",
    },
    "status.region_cleared": {"en": "Region selection cleared", "ja": "領域選択をクリア"},
    "status.done": {
        "en": "Done: {layers} layers, {vertices} vertices",
        "ja": "処理完了: {layers}レイヤー, {vertices}頂点",
    },
    "status.error": {"en": "Processing error", "ja": "処理エラー"},
    "status.exported": {"en": "Exported: {path}", "ja": "エクスポート完了: {path}"},
    "status.8dir_done": {
        "en": "8-direction images saved to {path}",
        "ja": "8方向画像を保存しました: {path}",
    },
    # ---- Dialogs ----
    "dialog.open_image": {"en": "Open Image", "ja": "画像を開く"},
    "dialog.image_filter": {
        "en": "Image files (*.png *.jpg *.jpeg *.gif *.bmp);;All files (*.*)",
        "ja": "画像ファイル (*.png *.jpg *.jpeg *.gif *.bmp);;すべてのファイル (*.*)",
    },
    "dialog.error": {"en": "Error", "ja": "エラー"},
    "dialog.load_failed": {
        "en": "Failed to load image:\n{error}",
        "ja": "画像の読み込みに失敗しました:\n{error}",
    },
    "dialog.process_failed": {
        "en": "Processing failed:\n{error}",
        "ja": "処理に失敗しました:\n{error}",
    },
    "dialog.view3d_failed": {
        "en": "3D view failed:\n{error}",
        "ja": "3D表示に失敗しました:\n{error}",
    },
    "dialog.export_failed": {
        "en": "Export failed:\n{error}",
        "ja": "エクスポートに失敗しました:\n{error}",
    },
    "dialog.export": {"en": "Export", "ja": "エクスポート"},
    "dialog.save_settings": {"en": "Save Settings", "ja": "設定の保存"},
    "dialog.save_settings_q": {
        "en": "Save current settings as defaults?",
        "ja": "初期値を保存しますか？",
    },
    "dialog.select_dir": {"en": "Select Output Directory", "ja": "出力フォルダを選択"},
    "dialog.about_text": {
        "en": (
            "TCG Shadowbox Generator\n\n"
            "Transform TCG card illustrations into layered 3D shadowboxes\n"
            "using AI-powered depth estimation and clustering.\n\n"
            "https://github.com/2ROSSO/shadowbox-generator"
        ),
        "ja": (
            "TCG Shadowbox Generator\n\n"
            "TCGカードのイラストを深度推定とクラスタリングで階層化し、\n"
            "インタラクティブな3Dシャドーボックスとして表示するツール。\n\n"
            "https://github.com/2ROSSO/shadowbox-generator"
        ),
    },
    "dialog.pick_bg_color": {"en": "Choose Background Color", "ja": "背景色を選択"},
    # ---- Tabs (image preview) ----
    "tab.original": {"en": "Original", "ja": "元画像"},
    "tab.depth": {"en": "Depth", "ja": "深度"},
    "tab.layers": {"en": "Layers", "ja": "レイヤー"},
    "tab.no_image": {"en": "No image", "ja": "画像なし"},
    "tab.region_select": {"en": "Region Select", "ja": "領域選択"},
    "tab.region_reset": {"en": "Region Reset", "ja": "領域リセット"},
    # ---- Buttons ----
    "btn.generate": {"en": "Generate Shadowbox", "ja": "シャドーボックス生成"},
    "btn.view_3d": {"en": "Open 3D View", "ja": "3Dビューを開く"},
    "btn.export_3d": {"en": "3D Data Export", "ja": "3Dデータ出力"},
    "btn.export_8dir": {"en": "8-Direction Image Export", "ja": "8方向画像出力"},
    # ---- Settings tabs ----
    "settings.processing": {"en": "Processing", "ja": "処理"},
    "settings.layers": {"en": "Layers", "ja": "レイヤー"},
    "settings.frame": {"en": "Frame", "ja": "フレーム"},
    "settings.rendering": {"en": "3D View", "ja": "3D表示"},
    # ---- Processing tab ----
    "proc.model_mode": {"en": "Model mode:", "ja": "モデルモード:"},
    "proc.detection": {"en": "Detection:", "ja": "検出方法:"},
    "proc.mock_depth": {"en": "Mock depth (testing)", "ja": "モック深度推定（テスト用）"},
    "proc.raw_depth": {"en": "Raw depth mode", "ja": "生深度モード"},
    "proc.depth_scale": {"en": "Depth scale:", "ja": "深度スケール:"},
    "proc.num_layers": {"en": "Layers:", "ja": "レイヤー数:"},
    "proc.auto": {"en": "Auto", "ja": "自動"},
    "proc.max_resolution": {"en": "Max resolution:", "ja": "最大解像度:"},
    "proc.unlimited": {"en": "Unlimited", "ja": "無制限"},
    # ---- Layers tab ----
    "layer.cumulative": {"en": "Cumulative layers", "ja": "累積レイヤー"},
    "layer.back_panel": {"en": "Back panel", "ja": "背面パネル"},
    "layer.interpolation": {"en": "Layer interpolation:", "ja": "レイヤー補間:"},
    "layer.pop_out": {"en": "Pop-out amount:", "ja": "飛び出し量:"},
    "layer.spacing_mode": {"en": "Spacing mode:", "ja": "間隔モード:"},
    "layer.mask_mode": {"en": "Mask mode:", "ja": "マスクモード:"},
    "layer.thickness": {"en": "Layer thickness:", "ja": "レイヤー厚み:"},
    "layer.gap": {"en": "Layer gap:", "ja": "レイヤー隙間:"},
    # ---- Frame tab ----
    "frame.include": {"en": "Include frame", "ja": "フレームを含める"},
    "frame.include_card": {"en": "Include card frame", "ja": "カードフレームを含める"},
    "frame.depth": {"en": "Frame depth:", "ja": "フレーム厚み:"},
    "frame.wall_mode": {"en": "Wall mode:", "ja": "壁モード:"},
    # ---- Rendering tab ----
    "render.mode": {"en": "Render mode:", "ja": "描画モード:"},
    "render.point_size": {"en": "Point size:", "ja": "ポイントサイズ:"},
    "render.mesh_size": {"en": "Mesh size:", "ja": "メッシュサイズ:"},
    "render.show_axes": {"en": "Show axes", "ja": "軸を表示"},
    "render.show_frame": {"en": "Show frame", "ja": "フレームを表示"},
    "render.opacity": {"en": "Opacity:", "ja": "不透明度:"},
    "render.bg_color": {"en": "Background:", "ja": "背景色:"},
    # ---- Processing thread ----
    "progress.creating": {"en": "Creating pipeline...", "ja": "パイプラインを作成中..."},
    "progress.processing": {"en": "Processing...", "ja": "処理中..."},
    "progress.detecting": {
        "en": "Detecting illustration region...",
        "ja": "イラスト領域を検出中...",
    },
}


def tr(key: str, **kwargs: object) -> str:
    """翻訳済み文字列を返す。

    Args:
        key: 翻訳キー。
        **kwargs: 文字列フォーマット引数。

    Returns:
        現在の言語での翻訳文字列。キーが見つからない場合はキー自体を返す。
    """
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(_current_language, entry.get("en", key))
    if kwargs:
        text = text.format(**kwargs)
    return text


def set_language(lang: str) -> None:
    """表示言語を変更する。

    Args:
        lang: 言語コード ("en" または "ja")。
    """
    global _current_language
    _current_language = lang


def get_language() -> str:
    """現在の表示言語を返す。"""
    return _current_language


def _prefs_path() -> Path:
    """言語設定ファイルのパスを返す。"""
    return Path.home() / ".shadowbox" / "language.json"


def save_language_preference() -> None:
    """現在の言語設定をファイルに保存する。"""
    path = _prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"language": _current_language}), encoding="utf-8")


def load_language_preference() -> None:
    """ファイルから言語設定を読み込む。"""
    global _current_language
    path = _prefs_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            lang = data.get("language", "en")
            if lang in ("en", "ja"):
                _current_language = lang
        except (json.JSONDecodeError, OSError):
            pass
