"""Trimap editing with an undo stack.

The original RGB image is never modified.  Edits produce *new* trimap arrays;
the history holds immutable copies so every stroke can be reverted.  Markers
are hard labels (definite bg / definite fg); unknown pixels remain unknown
until the solver runs, and the solver's output is never written back into the
trimap.
"""
from __future__ import annotations

import numpy as np

from .imageio import TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN


class TrimapEditor:
    def __init__(self, trimap: np.ndarray, max_history: int = 50):
        self._initial = trimap.copy()
        self._history: list[np.ndarray] = [trimap.copy()]
        self._cursor = 0
        self._max_history = max_history

    @property
    def trimap(self) -> np.ndarray:
        return self._history[self._cursor]

    def paint(self, mask: np.ndarray, label: int) -> np.ndarray:
        """Apply a hard marker (TRIMAP_BG / TRIMAP_FG) on the given pixel mask."""
        if label not in (TRIMAP_BG, TRIMAP_FG):
            raise ValueError("hard markers must be definite bg (0) or fg (255)")
        if mask.shape != self.trimap.shape or mask.dtype != bool:
            raise ValueError("paint mask must be a boolean array of trimap shape")
        new_map = self.trimap.copy()
        new_map[mask] = label
        # Any redo branch is discarded when a new stroke is made.
        del self._history[self._cursor + 1:]
        self._history.append(new_map)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._cursor = len(self._history) - 1
        return self.trimap

    def erase_to_unknown(self, mask: np.ndarray) -> np.ndarray:
        new_map = self.trimap.copy()
        new_map[mask] = TRIMAP_UNKNOWN
        del self._history[self._cursor + 1:]
        self._history.append(new_map)
        self._cursor = len(self._history) - 1
        return self.trimap

    def undo(self) -> np.ndarray:
        if self._cursor > 0:
            self._cursor -= 1
        return self.trimap

    def redo(self) -> np.ndarray:
        if self._cursor < len(self._history) - 1:
            self._cursor += 1
        return self.trimap

    def reset(self) -> np.ndarray:
        self._history = [self._initial.copy()]
        self._cursor = 0
        return self.trimap

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._history) - 1


def disk_mask(shape: tuple[int, int], center: tuple[int, int],
              radius: int) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    return (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2


def line_mask(shape: tuple[int, int], p0: tuple[int, int],
              p1: tuple[int, int], radius: int) -> np.ndarray:
    """Rasterised thick segment used for brush strokes between pointer points."""
    h, w = shape
    y0, x0 = p0
    y1, x1 = p1
    n = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    ys = np.linspace(y0, y1, n).round().astype(int)
    xs = np.linspace(x0, x1, n).round().astype(int)
    mask = np.zeros((h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    for cy, cx in zip(ys, xs):
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask
