"""Synthetic hair scenes with known ground-truth alpha and foreground colour.

Scene layout (the exact configuration requested for evaluation):

* a vertical *fixed background line* splits the frame;
* left of the line: a textured curtain (multi-frequency procedural texture);
* right of the line: a flat region whose colour is deliberately close to the
  hair colour (low fg/bg contrast, the hard matting case);
* anti-aliased hair strands with sub-pixel widths sweep from the curtain side
  into the flat side, crossing both the background line and (optionally) the
  solver's tile seams;
* ground-truth alpha and foreground colour are retained for error metrics;
* an optional JPEG round-trip injects realistic compression/blocking noise.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SyntheticScene:
    image: np.ndarray            # observed RGB, possibly compression-noised
    image_clean: np.ndarray      # noise-free composite
    alpha_gt: np.ndarray         # ground-truth alpha
    foreground_gt: np.ndarray    # ground-truth hair colour
    background_gt: np.ndarray
    trimap: np.ndarray
    bg_line_x: int               # fixed background line column


def _curtain(h: int, w: int, split: int) -> np.ndarray:
    """Textured curtain: low-frequency folds plus fine weave texture."""
    bg = np.zeros((h, w, 3), dtype=np.float64)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    folds = 0.5 + 0.5 * np.sin(x / 14.0 + 0.7 * np.sin(y / 31.0))
    weave = 0.5 + 0.5 * np.sin(x * 1.7) * np.sin(y * 1.3)
    rng = np.random.default_rng(7)
    grain = rng.normal(0.0, 0.03, size=(h, w))
    lum = np.clip(0.35 + 0.30 * folds + 0.10 * weave + grain, 0, 1)
    bg[..., 0] = lum * 0.85
    bg[..., 1] = lum * 0.55
    bg[..., 2] = lum * 0.35
    bg[:, split:, :] = 0.0      # overwritten by caller with flat colour
    return bg


def _bezier(p0, p1, p2, p3, n=400):
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    pts = (mt ** 3 * np.array(p0)
           + 3 * mt ** 2 * t * np.array(p1)
           + 3 * mt * t ** 2 * np.array(p2)
           + t ** 3 * np.array(p3))
    return pts


def _draw_strand(alpha: np.ndarray, fg: np.ndarray, path: np.ndarray,
                 width: float, hair_color: np.ndarray, color_drift: float,
                 rng: np.random.Generator) -> None:
    """Anti-aliased strand: alpha falls off smoothly across the strand width."""
    h, w = alpha.shape
    # Bounding box of the path with margin, clamped to the frame. Skip strands
    # whose path never enters the image.
    ymin = max(int(path[:, 0].min() - width - 2), 0)
    ymax = min(int(path[:, 0].max() + width + 3), h)
    xmin = max(int(path[:, 1].min() - width - 2), 0)
    xmax = min(int(path[:, 1].max() + width + 3), w)
    if ymax <= ymin or xmax <= xmin:
        return
    yy, xx = np.mgrid[ymin:ymax, xmin:xmax].astype(np.float64)
    d2 = np.full(yy.shape, np.inf, dtype=np.float64)
    nearest_t = np.zeros(yy.shape, dtype=np.float64)
    for k in range(len(path) - 1):
        a = path[k]
        b = path[k + 1]
        v = b - a
        seg_len2 = (v ** 2).sum()
        if seg_len2 < 1e-12:
            continue
        t = np.clip(((yy - a[0]) * v[0] + (xx - a[1]) * v[1]) / seg_len2, 0, 1)
        py = a[0] + t * v[0]
        px = a[1] + t * v[1]
        dist2 = (yy - py) ** 2 + (xx - px) ** 2
        closer = dist2 < d2
        d2[closer] = dist2[closer]
        nearest_t[closer] = (k + t[closer]) / (len(path) - 1)
    dist = np.sqrt(d2)
    # Smooth alpha ramp: fully opaque inside width*0.5, zero beyond width.
    strand_alpha = np.clip((width - dist) / (width * 0.5 + 1e-9), 0.0, 1.0)
    strand_alpha = strand_alpha * strand_alpha * (3 - 2 * strand_alpha)

    local_fg = hair_color[None, None, :] * (
        1.0 + color_drift * rng.normal(0, 1, size=(*strand_alpha.shape, 1))
    )
    local_fg = np.clip(local_fg, 0.0, 1.0)

    region = alpha[ymin:ymax, xmin:xmax]
    a_old = region[..., None]
    a_new = strand_alpha[..., None]
    # Porter-Duff over for overlapping strands.
    out_alpha = a_new + a_old * (1.0 - a_new)
    out_fg = (local_fg * a_new + fg[ymin:ymax, xmin:xmax] * a_old
              * (1.0 - a_new)) / np.maximum(out_alpha, 1e-12)
    alpha[ymin:ymax, xmin:xmax] = out_alpha[..., 0]
    fg[ymin:ymax, xmin:xmax] = np.where(out_alpha > 1e-9, out_fg,
                                        fg[ymin:ymax, xmin:xmax])


def make_hair_scene(h: int = 320, w: int = 480,
                    hair_color=(0.20, 0.11, 0.07),
                    flat_bg_color=(0.24, 0.15, 0.10),
                    jpeg_quality: int | None = None,
                    seed: int = 0) -> SyntheticScene:
    """Build the curtain-to-flat-region hair scene with full ground truth."""
    rng = np.random.default_rng(seed)
    split = w // 2
    background = _curtain(h, w, split)
    flat = np.array(flat_bg_color, dtype=np.float64)
    background[:, split:, :] = flat[None, None, :]

    alpha = np.zeros((h, w), dtype=np.float64)
    fg = np.zeros((h, w, 3), dtype=np.float64)
    hair = np.array(hair_color, dtype=np.float64)

    # Thick lock plus fine filaments; each crosses the background line at x=split
    # and is placed so it also crosses a 160 px tile seam at x=160/320.
    strands = [
        ((-20, 60), (90, 120), (180, 200), (h + 20, 300), 3.2),
        ((-10, 30), (80, 110), (200, 250), (h + 10, 430), 1.4),
        ((h + 10, 40), (220, 130), (90, 220), (-20, 330), 1.1),
        ((20, 0), (120, 90), (210, 260), (300, w + 10), 0.9),
        ((160, 0), (170, 110), (150, 250), (170, w + 10), 0.7),
        ((260, 0), (240, 100), (250, 240), (250, w + 10), 1.6),
    ]
    for p0, p1, p2, p3, width in strands:
        path = _bezier(p0, p1, p2, p3)
        _draw_strand(alpha, fg, path, width, hair, color_drift=0.08, rng=rng)

    clean = fg * alpha[..., None] + background * (1.0 - alpha[..., None])
    observed = clean.copy()
    if jpeg_quality is not None:
        observed = _jpeg_roundtrip(observed, jpeg_quality)

    trimap = _trimap_from_alpha(alpha, band=5)
    return SyntheticScene(
        image=observed,
        image_clean=clean,
        alpha_gt=alpha,
        foreground_gt=fg,
        background_gt=background,
        trimap=trimap,
        bg_line_x=split,
    )


def _trimap_from_alpha(alpha: np.ndarray, band: int = 5) -> np.ndarray:
    """Unknown band around hair; fixed labels only where evidence is certain."""
    trimap = np.full(alpha.shape, 128, dtype=np.uint8)
    trimap[alpha >= 0.95] = 255
    trimap[alpha == 0.0] = 0
    # Erode the fixed labels so the unknown region is a genuine band and no
    # distance-field fill can reproduce the answer.
    import scipy.ndimage as ndi
    fg = ndi.binary_erosion(trimap == 255, iterations=1)
    bg = ndi.binary_erosion(trimap == 0, iterations=band)
    trimap[:] = 128
    trimap[bg] = 0
    trimap[fg] = 255
    return trimap


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    arr = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0
    return out
