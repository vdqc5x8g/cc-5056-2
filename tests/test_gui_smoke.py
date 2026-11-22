"""Offscreen GUI smoke tests: the canvas really paints, solves and renders.

These drive the actual Qt event handlers (mouse press/move/release) and the
real ``main()`` entry point, so a broken stroke callback or a missing layer
fails loudly instead of being masked by script-only checks.

Run directly:  python3 tests/test_gui_smoke.py
Requires PySide6; the test forces QT_QPA_PLATFORM=offscreen.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from hairmatting.gui.workbench import MainWindow, main as gui_main
    HAVE_QT = True
except Exception:                                   # pragma: no cover
    HAVE_QT = False

from hairmatting.solver.imageio import TRIMAP_BG  # noqa: E402
from hairmatting.solver.synthetic import make_hair_scene  # noqa: E402


def _mouse(widget, etype, img_yx, button, buttons):
    canvas = widget
    x = img_yx[1] * canvas._scale + canvas._offset.x()
    y = img_yx[0] * canvas._scale + canvas._offset.y()
    ev = QtGui.QMouseEvent(etype, QtCore.QPointF(x, y), button, buttons,
                           QtCore.Qt.NoModifier)
    if etype == QtCore.QEvent.Type.MouseButtonPress:
        canvas.mousePressEvent(ev)
    elif etype == QtCore.QEvent.Type.MouseMove:
        canvas.mouseMoveEvent(ev)
    else:
        canvas.mouseReleaseEvent(ev)


@unittest.skipUnless(HAVE_QT, "PySide6 unavailable")
class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            [])

    def _window_with_scene(self):
        win = MainWindow()
        win.resize(1100, 700)
        scene = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)
        win.set_scene(scene.image, scene.trimap)
        win.show()
        self.app.processEvents()
        return win, scene

    def test_main_launches_demo_and_screenshots_alpha_and_composite(self):
        with tempfile.TemporaryDirectory() as d:
            alpha_png = os.path.join(d, "alpha.png")
            comp_png = os.path.join(d, "composite.png")
            rc = gui_main(["--demo", "--solve", "--layer", "alpha",
                           "--screenshot", alpha_png])
            self.assertEqual(rc, 0)
            rc = gui_main(["--demo", "--solve", "--layer", "black",
                           "--screenshot", comp_png])
            self.assertEqual(rc, 0)
            from PIL import Image
            for path in (alpha_png, comp_png):
                arr = np.asarray(Image.open(path).convert("L"),
                                 dtype=np.float64)
                # A rendered view, not a blank window.
                self.assertGreater(arr.std(), 1.0)

    def test_mouse_drag_paints_one_atomic_stroke(self):
        win, scene = self._window_with_scene()
        before = win.editor.trimap.copy()
        n_history_before = len(win.editor._history)
        press = _mouse
        press(win.canvas, QtCore.QEvent.Type.MouseButtonPress, (20, 20),
              QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
        for p in ((30, 45), (40, 70), (50, 95)):
            _mouse(win.canvas, QtCore.QEvent.Type.MouseMove, p,
                   QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
        _mouse(win.canvas, QtCore.QEvent.Type.MouseButtonRelease, (50, 95),
               QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
        self.app.processEvents()
        # The drag painted background and is ONE atomic history entry.
        self.assertFalse(np.array_equal(win.editor.trimap, before))
        self.assertEqual(len(win.editor._history), n_history_before + 1)
        win.editor.undo()
        self.assertTrue(np.array_equal(win.editor.trimap, before))

    def test_solve_populates_layers_and_status(self):
        win, scene = self._window_with_scene()
        win._solve(False)
        self.assertIsNotNone(win.result)
        for layer in ("alpha", "foreground", "black", "white", "evidence"):
            self.assertIn(layer, win.canvas.layers)
        self.assertEqual(win.result.fixed_violation, 0.0)
        self.assertIn("solve done", win.status.text().lower())
        win._solve(True)
        self.assertIsNotNone(win.result.block_info)
        self.assertEqual(win.result.fixed_violation, 0.0)

    def test_solver_failure_feeds_back_to_status(self):
        win, scene = self._window_with_scene()
        # A trimap without any definite label is ill-posed; the workbench
        # must surface the failure instead of crashing.
        win.editor = type(win.editor)(
            np.full(scene.trimap.shape, 128, dtype=np.uint8))
        win._solve(False)
        self.assertIn("solve failed", win.status.text().lower())

    def test_solve_without_scene_is_rejected_gracefully(self):
        win = MainWindow()
        win._solve(False)
        self.assertIn("load both", win.status.text().lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
