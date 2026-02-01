"""GUIモジュール。

テンプレートエディタ、画像選択、スタンドアロンアプリの
GUI機能を提供します。
"""

from shadowbox.gui.image_selector import (
    ImageSelector,
    URLImageLoader,
    create_image_selector,
    load_from_url,
)
from shadowbox.gui.template_editor import (
    QuickRegionSelector,
    TemplateEditor,
    select_illustration_region,
)

__all__ = [
    # template_editor
    "TemplateEditor",
    "QuickRegionSelector",
    "select_illustration_region",
    # image_selector
    "ImageSelector",
    "URLImageLoader",
    "create_image_selector",
    "load_from_url",
]


def run_app():
    """GUIアプリケーションを起動。

    Example:
        >>> from shadowbox.gui import run_app
        >>> run_app()
    """
    from shadowbox.gui.app import main

    main()
