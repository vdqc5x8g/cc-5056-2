"""PySide6 graphical workbench for hair matting.

Features
--------
* load an RGB image and a trimap; the source image is never modified;
* hard definite-fg / definite-bg / erase brushes with full undo/redo;
* zoom (wheel) and pan (middle mouse or shift+drag);
* per-channel inspection of image, alpha and foreground (R/G/B/grayscale);
* one-click global or tiled solve with seam diagnostics;
* black/white background composites side by side, plus uncertainty overlay;
* export of alpha, RGBA foreground, composites, evidence and uncertainty.

The whole Qt surface is defined only when PySide6 (and its system libraries)
import successfully, so the package and CLI work on headless hosts.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ..solver.imageio import (
    TRIMAP_BG, TRIMAP_FG, save_alpha, save_rgb, save_rgba, save_trimap,
    load_rgb, load_trimap,
)
from ..solver.pipeline import run_pipeline
from ..solver.trimap_editor import TrimapEditor, line_mask

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception:  # ImportError or missing system libs (e.g. libEGL) headless
    QtCore = QtGui = QtWidgets = None


if QtWidgets is not None:

    def _to_qimage(rgb: np.ndarray) -> "QtGui.QImage":
        arr = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        h, w, _ = arr.shape
        return QtGui.QImage(arr.copy(), w, h, 3 * w,
                            QtGui.QImage.Format_RGB888)

    def _gray_qimage(gray: np.ndarray) -> "QtGui.QImage":
        arr = np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8)
        h, w = arr.shape
        return QtGui.QImage(arr.copy(), w, h, w,
                            QtGui.QImage.Format_Grayscale8)

    class ImageCanvas(QtWidgets.QWidget):
        """Zoomable/pannable canvas for image, trimap overlay and results."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(480, 360)
            self.setMouseTracking(True)
            self._scale = 1.0
            self._offset = QtCore.QPoint(0, 0)
            self._panning = False
            self._last_pan = None
            self.layers = {}
            self.active_layer = "image"
            self.trimap_overlay = None
            self.uncertain_overlay = None
            self.show_trimap = True
            self.show_uncertain = False
            self.channel = "RGB"
            self.brush_radius = 8
            self.on_stroke = None
            self.stroke_label = TRIMAP_BG
            self._stroke_start = None
            self.base_rgb = None
            self.result = None

        def set_base_image(self, rgb: np.ndarray):
            self.base_rgb = rgb
            self._refresh_layers()
            self.fit_to_view()

        def _select_channel(self, rgb):
            if rgb.ndim == 3:
                if self.channel == "RGB":
                    return rgb
                idx = {"R": 0, "G": 1, "B": 2}[self.channel]
                return np.repeat(rgb[..., idx][..., None], 3, axis=2)
            return rgb

        def set_result(self, result):
            self.result = result
            self._refresh_layers()

        def set_trimap(self, trimap: np.ndarray):
            ov = np.zeros((*trimap.shape, 4), dtype=np.uint8)
            ov[trimap == TRIMAP_FG] = (0, 200, 0, 90)
            ov[trimap == TRIMAP_BG] = (255, 60, 60, 90)
            ov[trimap == 128] = (255, 255, 0, 40)
            h, w = trimap.shape
            self.trimap_overlay = QtGui.QImage(
                ov, w, h, 4 * w, QtGui.QImage.Format_RGBA8888).copy()
            self.update()

        def set_uncertain(self, mask: np.ndarray, score: np.ndarray):
            ov = np.zeros((*mask.shape, 4), dtype=np.uint8)
            ov[..., 0] = np.clip(score * 255, 0, 255).astype(np.uint8)
            ov[..., 3] = np.where(mask, 180, 0).astype(np.uint8)
            h, w = mask.shape
            self.uncertain_overlay = QtGui.QImage(
                ov, w, h, 4 * w, QtGui.QImage.Format_RGBA8888).copy()
            self.update()

        def _refresh_layers(self):
            if self.base_rgb is None:
                return
            self.layers = {
                "image": _to_qimage(self._select_channel(self.base_rgb))}
            if self.result is not None:
                r = self.result
                self.layers["alpha"] = _gray_qimage(r.alpha)
                self.layers["foreground"] = _to_qimage(
                    self._select_channel(r.foreground))
                self.layers["black"] = _to_qimage(
                    self._select_channel(r.composite_black))
                self.layers["white"] = _to_qimage(
                    self._select_channel(r.composite_white))
                self.layers["evidence"] = _gray_qimage(r.evidence)
            self.update()

        def fit_to_view(self):
            qimg = self.layers.get("image")
            if qimg is None:
                return
            self._scale = min(self.width() / qimg.width(),
                              self.height() / qimg.height())
            self._offset = QtCore.QPoint(0, 0)

        def wheelEvent(self, event):
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self._scale = float(np.clip(self._scale * factor, 0.1, 16.0))
            self.update()

        def _image_pos(self, pos):
            if self.base_rgb is None:
                return None
            p = (pos - self._offset) / self._scale
            h, w = self.base_rgb.shape[:2]
            x, y = int(p.x()), int(p.y())
            if 0 <= x < w and 0 <= y < h:
                return y, x
            return None

        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.MiddleButton or (
                event.button() == QtCore.Qt.LeftButton
                and event.modifiers() & QtCore.Qt.ShiftModifier
            ):
                self._panning = True
                self._last_pan = event.position().toPoint()
                return
            if event.button() == QtCore.Qt.LeftButton and self.on_stroke:
                p = self._image_pos(event.position().toPoint())
                if p is not None:
                    self._stroke_start = p
                    self.on_stroke(line_mask(
                        self.base_rgb.shape[:2], p, p, self.brush_radius))

        def mouseMoveEvent(self, event):
            if self._panning:
                now = event.position().toPoint()
                self._offset += now - self._last_pan
                self._last_pan = now
                self.update()
                return
            if event.buttons() & QtCore.Qt.LeftButton and self.on_stroke:
                p = self._image_pos(event.position().toPoint())
                if p is not None and self._stroke_start is not None:
                    self.on_stroke(line_mask(
                        self.base_rgb.shape[:2], self._stroke_start, p,
                        self.brush_radius))
                    self._stroke_start = p

        def mouseReleaseEvent(self, event):
            self._panning = False
            self._stroke_start = None

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
            qimg = self.layers.get(self.active_layer)
            if qimg is None:
                return
            target = QtCore.QRect(
                self._offset,
                QtCore.QSize(int(qimg.width() * self._scale),
                             int(qimg.height() * self._scale)))
            painter.drawImage(target, qimg)
            if self.show_trimap and self.trimap_overlay is not None:
                painter.drawImage(target, self.trimap_overlay)
            if self.show_uncertain and self.uncertain_overlay is not None:
                painter.drawImage(target, self.uncertain_overlay)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Hair Matting Workbench")
            self.editor = None
            self.image = None
            self.result = None

            self.canvas = ImageCanvas()
            self.canvas.on_stroke = self._apply_stroke
            self.setCentralWidget(self.canvas)

            dock = QtWidgets.QDockWidget("Controls", self)
            panel = QtWidgets.QWidget()
            form = QtWidgets.QVBoxLayout(panel)

            load_row = QtWidgets.QHBoxLayout()
            self.btn_image = QtWidgets.QPushButton("Load image")
            self.btn_trimap = QtWidgets.QPushButton("Load trimap")
            self.btn_image.clicked.connect(self._load_image)
            self.btn_trimap.clicked.connect(self._load_trimap)
            load_row.addWidget(self.btn_image)
            load_row.addWidget(self.btn_trimap)
            form.addLayout(load_row)

            form.addWidget(QtWidgets.QLabel("Brush label"))
            self.label_combo = QtWidgets.QComboBox()
            self.label_combo.addItems([
                "Definite background (0)",
                "Definite foreground (255)",
                "Erase to unknown (128)"])
            form.addWidget(self.label_combo)

            radius_row = QtWidgets.QHBoxLayout()
            radius_row.addWidget(QtWidgets.QLabel("Brush radius"))
            self.radius_spin = QtWidgets.QSpinBox()
            self.radius_spin.setRange(1, 60)
            self.radius_spin.setValue(8)
            self.radius_spin.valueChanged.connect(
                lambda v: setattr(self.canvas, "brush_radius", v))
            radius_row.addWidget(self.radius_spin)
            form.addLayout(radius_row)

            undo_row = QtWidgets.QHBoxLayout()
            self.btn_undo = QtWidgets.QPushButton("Undo")
            self.btn_redo = QtWidgets.QPushButton("Redo")
            self.btn_reset = QtWidgets.QPushButton("Reset trimap")
            self.btn_undo.clicked.connect(self._undo)
            self.btn_redo.clicked.connect(self._redo)
            self.btn_reset.clicked.connect(self._reset)
            undo_row.addWidget(self.btn_undo)
            undo_row.addWidget(self.btn_redo)
            undo_row.addWidget(self.btn_reset)
            form.addLayout(undo_row)

            form.addWidget(QtWidgets.QLabel("View layer"))
            self.layer_combo = QtWidgets.QComboBox()
            self.layer_combo.addItems(["image", "alpha", "foreground",
                                       "black", "white", "evidence"])
            self.layer_combo.currentTextChanged.connect(self._set_layer)
            form.addWidget(self.layer_combo)

            form.addWidget(QtWidgets.QLabel("Channel"))
            self.channel_combo = QtWidgets.QComboBox()
            self.channel_combo.addItems(["RGB", "R", "G", "B"])
            self.channel_combo.currentTextChanged.connect(self._set_channel)
            form.addWidget(self.channel_combo)

            self.chk_trimap = QtWidgets.QCheckBox("Show trimap overlay")
            self.chk_trimap.setChecked(True)
            self.chk_trimap.toggled.connect(self._toggle_overlays)
            form.addWidget(self.chk_trimap)
            self.chk_unc = QtWidgets.QCheckBox("Show uncertainty overlay")
            self.chk_unc.toggled.connect(self._toggle_overlays)
            form.addWidget(self.chk_unc)

            solve_row = QtWidgets.QHBoxLayout()
            self.btn_solve = QtWidgets.QPushButton("Solve global")
            self.btn_solve_blocks = QtWidgets.QPushButton("Solve tiled")
            self.btn_solve.clicked.connect(lambda: self._solve(False))
            self.btn_solve_blocks.clicked.connect(lambda: self._solve(True))
            solve_row.addWidget(self.btn_solve)
            solve_row.addWidget(self.btn_solve_blocks)
            form.addLayout(solve_row)

            self.btn_export = QtWidgets.QPushButton("Export results...")
            self.btn_export.clicked.connect(self._export)
            form.addWidget(self.btn_export)

            self.status = QtWidgets.QLabel("Load an image and trimap to begin.")
            self.status.setWordWrap(True)
            form.addWidget(self.status)
            form.addStretch(1)

            dock.setWidget(panel)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            self.resize(1280, 800)

        def _load_image(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image")
            if path:
                self.image = load_rgb(path)
                self.canvas.set_base_image(self.image)
                self.status.setText(
                    f"Loaded image {os.path.basename(path)}")

        def _load_trimap(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open trimap")
            if path:
                self.editor = TrimapEditor(load_trimap(path))
                self.canvas.set_trimap(self.editor.trimap)
                self.status.setText(
                    f"Loaded trimap {os.path.basename(path)}")

        def _apply_stroke(self, mask, _ignored):
            if self.editor is None or self.image is None:
                return
            choice = self.label_combo.currentIndex()
            if choice == 0:
                self.editor.paint(mask, TRIMAP_BG)
            elif choice == 1:
                self.editor.paint(mask, TRIMAP_FG)
            else:
                self.editor.erase_to_unknown(mask)
            self.canvas.set_trimap(self.editor.trimap)

        def _undo(self):
            if self.editor:
                self.editor.undo()
                self.canvas.set_trimap(self.editor.trimap)

        def _redo(self):
            if self.editor:
                self.editor.redo()
                self.canvas.set_trimap(self.editor.trimap)

        def _reset(self):
            if self.editor:
                self.editor.reset()
                self.canvas.set_trimap(self.editor.trimap)

        def _set_layer(self, name):
            self.canvas.active_layer = name
            self.canvas.update()

        def _set_channel(self, name):
            self.canvas.channel = name
            self.canvas._refresh_layers()

        def _toggle_overlays(self):
            self.canvas.show_trimap = self.chk_trimap.isChecked()
            self.canvas.show_uncertain = self.chk_unc.isChecked()
            self.canvas.update()

        def _solve(self, blocks: bool):
            if self.editor is None or self.image is None:
                self.status.setText("Load both image and trimap first.")
                return
            self.result = run_pipeline(
                self.image, self.editor.trimap, use_blocks=blocks)
            self.canvas.set_result(self.result)
            self.canvas.set_uncertain(self.result.uncertain_mask,
                                      self.result.uncertainty_score)
            msg = (f"{'Tiled' if blocks else 'Global'} solve done. "
                   f"fixed violation={self.result.fixed_violation:.2e}, "
                   f"uncertain fraction="
                   f"{self.result.uncertain_mask.mean():.3f}")
            if self.result.block_info:
                msg += (", seam mean diff="
                        f"{self.result.block_info['seam_residual_mean']:.4f}")
            self.status.setText(msg)

        def _export(self):
            if self.result is None:
                self.status.setText("Run a solve before exporting.")
                return
            outdir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Export to")
            if not outdir:
                return
            r = self.result
            save_alpha(os.path.join(outdir, "alpha.png"), r.alpha)
            save_rgba(os.path.join(outdir, "foreground_rgba.png"),
                      r.foreground, r.alpha)
            save_rgb(os.path.join(outdir, "composite_black.png"),
                     r.composite_black)
            save_rgb(os.path.join(outdir, "composite_white.png"),
                     r.composite_white)
            save_rgb(os.path.join(outdir, "evidence.png"),
                     np.repeat(r.evidence[..., None], 3, axis=2))
            unc_vis = np.zeros((*r.uncertainty_score.shape, 3))
            unc_vis[..., 0] = r.uncertainty_score
            save_rgb(os.path.join(outdir, "uncertainty.png"), unc_vis)
            if self.editor:
                save_trimap(os.path.join(outdir, "trimap_edited.png"),
                            self.editor.trimap)
            self.status.setText(
                f"Exported alpha, RGBA, composites, maps to {outdir}")


def main() -> int:
    if QtWidgets is None:
        print(
            "PySide6 (or its system libraries) is unavailable in this "
            "environment; use 'python -m hairmatting.cli' or the headless "
            "report script instead.", file=sys.stderr)
        return 1
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
