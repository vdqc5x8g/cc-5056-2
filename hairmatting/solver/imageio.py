"""Image / trimap IO and compositing utilities.

Conventions
-----------
RGB images: float64 arrays of shape (H, W, 3), range [0, 1], channel order RGB.
Trimap:    uint8 / int array with values {0, 128, 255}
           (0 = definite background, 255 = definite foreground, 128 = unknown).
Alpha:     float64 array of shape (H, W), range [0, 1].
"""
from __future__ import annotations

import numpy as np
from PIL import Image

TRIMAP_BG = 0
TRIMAP_UNKNOWN = 128
TRIMAP_FG = 255


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def load_trimap(path: str) -> np.ndarray:
    """Load a trimap and snap arbitrary gray levels to the three canonical labels."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.int16)
    out = np.full(arr.shape, TRIMAP_UNKNOWN, dtype=np.uint8)
    out[arr <= 64] = TRIMAP_BG
    out[arr >= 192] = TRIMAP_FG
    return out


def trimap_masks(trimap: np.ndarray):
    bg = trimap == TRIMAP_BG
    fg = trimap == TRIMAP_FG
    unk = trimap == TRIMAP_UNKNOWN
    return bg, fg, unk


def save_rgb(path: str, rgb: np.ndarray) -> None:
    arr = _to_uint8(rgb)
    Image.fromarray(arr, mode="RGB").save(path)


def save_alpha(path: str, alpha: np.ndarray) -> None:
    Image.fromarray(_to_uint8(np.clip(alpha, 0.0, 1.0)), mode="L").save(path)


def save_trimap(path: str, trimap: np.ndarray) -> None:
    Image.fromarray(trimap.astype(np.uint8), mode="L").save(path)


def save_rgba(path: str, rgb: np.ndarray, alpha: np.ndarray) -> None:
    rgba = np.dstack([_to_uint8(rgb), _to_uint8(np.clip(alpha, 0.0, 1.0))])
    Image.fromarray(rgba, mode="RGBA").save(path)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def composite(foreground_rgb: np.ndarray, alpha: np.ndarray,
              background_rgb) -> np.ndarray:
    """Standard Porter-Duff over-compositing.

    background_rgb may be an RGB image or a scalar / 3-vector constant.
    """
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    bg = np.asarray(background_rgb, dtype=np.float64)
    if bg.ndim == 0:
        bg = np.broadcast_to(bg, foreground_rgb.shape)
    elif bg.ndim == 1:
        bg = np.broadcast_to(bg[None, None, :], foreground_rgb.shape)
    return foreground_rgb * a + bg * (1.0 - a)


def split_channels(rgb: np.ndarray):
    return [rgb[..., c] for c in range(3)]
