"""領域選択テスト。"""

import pytest
from PyQt6.QtCore import QPoint, QRect

pytestmark = pytest.mark.gui


class TestRegionSelector:
    def test_initial_state(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        assert sel.get_selection() is None

    def test_set_active(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel.set_active(True)
        assert sel._active

    def test_clear_selection(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel._selection = QRect(10, 10, 50, 50)
        sel.clear_selection()
        assert sel._selection is None

    def test_set_image_rect(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel.set_image_rect(QRect(0, 0, 200, 200), (400, 400))
        assert sel._image_size == (400, 400)

    def test_clamp_to_image(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel.set_image_rect(QRect(10, 10, 100, 100), (100, 100))
        clamped = sel._clamp_to_image(QPoint(200, 200))
        assert clamped.x() <= 110
        assert clamped.y() <= 110

    def test_set_selection_sets_rect(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel.set_image_rect(QRect(0, 0, 200, 200), (200, 200))
        sel.set_selection(10, 20, 50, 60)
        assert sel._selection is not None
        assert isinstance(sel._selection, QRect)

    def test_set_selection_roundtrip(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel.set_image_rect(QRect(0, 0, 200, 200), (200, 200))
        sel.set_selection(10, 20, 50, 60)
        result = sel.get_selection()
        assert result is not None
        assert result == (10, 20, 50, 60)

    def test_set_selection_no_signal(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        sel.set_image_rect(QRect(0, 0, 200, 200), (200, 200))
        with qtbot.assertNotEmitted(sel.region_selected):
            sel.set_selection(10, 20, 50, 60)

    def test_set_selection_empty_image_rect_noop(self, qtbot):
        from shadowbox.gui.region_selector import RegionSelector

        sel = RegionSelector()
        qtbot.addWidget(sel)
        # image_rect is empty by default
        sel.set_selection(10, 20, 50, 60)
        assert sel._selection is None
