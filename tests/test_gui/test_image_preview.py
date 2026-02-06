"""プレビューテスト。"""

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.gui


class TestImagePreview:
    def test_initial_state(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        assert preview._current_tab == "original"
        assert len(preview._pixmaps) == 0

    def test_set_image(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        preview.set_image(img)
        assert "original" in preview._pixmaps
        assert preview._current_image_size == (100, 100)

    def test_set_depth_map(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        depth = np.random.rand(50, 50).astype(np.float32)
        preview.set_depth_map(depth)
        assert "depth" in preview._pixmaps

    def test_set_labels(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[25:, :] = 1
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        preview.set_labels(labels, image)
        assert "layers" in preview._pixmaps

    def test_clear(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        preview.set_image(img)
        preview.clear()
        assert len(preview._pixmaps) == 0

    def test_tab_switch(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        preview._switch_tab("depth")
        assert preview._current_tab == "depth"
        assert preview._stack.currentIndex() == 1

    def test_region_toggle_on_switches_to_original(self, qtbot):
        from shadowbox.gui.widgets.image_preview import ImagePreview

        preview = ImagePreview()
        qtbot.addWidget(preview)
        preview._switch_tab("depth")
        assert preview._current_tab == "depth"
        # Simulating region toggle ON
        preview._on_region_toggle(True)
        assert preview._current_tab == "original"
        assert preview._stack.currentIndex() == 0
