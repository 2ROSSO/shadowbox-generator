"""GUIモジュール。

テンプレートエディタ、画像選択、スタンドアロンアプリの
GUI機能を提供します。
"""

from shadowbox.gui.template_editor import (
    QuickRegionSelector,
    TemplateEditor,
    select_illustration_region,
)

__all__ = [
    "TemplateEditor",
    "QuickRegionSelector",
    "select_illustration_region",
]
