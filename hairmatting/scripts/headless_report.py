"""Headless report generator (PIL only): cross-background visual validation.

Produces a labelled montage containing, side by side:

* observed image, ground-truth and estimated alpha, their absolute difference;
* black- and white-background composites, fixed-point vs. unknown segments;
* per-channel views (R/G/B) of the estimated foreground;
* uncertainty overlay and evidence map;
* tiled-vs-global alpha with the tile seam lines drawn in.

This is the verifiable substitute for a single "looks good on one backdrop"
screenshot: every panel is a real solver output or ground-truth comparison.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from hairmatting.solver.imageio import (  # noqa: E402
    composite, load_rgb, load_trimap, save_rgb,
)
from hairmatting.solver.metrics import (  # noqa: E402
    alpha_metrics, color_residue, fixed_point_violation, region_masks_for_scene,
    seam_metrics,
)
from hairmatting.solver.pipeline import run_pipeline  # noqa: E402
from hairmatting.solver.synthetic import make_hair_scene  # noqa: E402

LABEL_H = 18
PADDING = 6


def _u8(a):
    return np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)


def _panel(rgb: np.ndarray, label: str) -> Image.Image:
    if rgb.ndim == 2:
        img = Image.fromarray(_u8(rgb), mode="L").convert("RGB")
    else:
        img = Image.fromarray(_u8(rgb), mode="RGB")
    canvas = Image.new("RGB",
                       (img.width, img.height + LABEL_H), (25, 25, 25))
    canvas.paste(img, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    d.text((2, 3), label, fill=(230, 230, 230))
    return canvas


def _row(panels: list[Image.Image]) -> Image.Image:
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + PADDING * (len(panels) - 1)
    row = Image.new("RGB", (w, h), (12, 12, 12))
    x = 0
    for p in panels:
        row.paste(p, (x, 0))
        x += p.width + PADDING
    return row


def _stack(rows: list[Image.Image]) -> Image.Image:
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + PADDING * (len(rows) - 1)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + PADDING
    return out


def _heat_overlay(base: np.ndarray, mask: np.ndarray,
                  score: np.ndarray) -> np.ndarray:
    # Only pixels that cross the uncertainty threshold are overlaid; intensity
    # encodes the score.  Low-score background stays untouched so a confident
    # zero-alpha region is never visually mistaken for an uncertainty band.
    out = base.copy()
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    weight = np.clip(score, 0, 1)[..., None] * 0.7
    out = np.where(mask[..., None], out * (1 - weight) + red * weight, out)
    return out


def _draw_seams(rgb: np.ndarray, v_edges, h_edges) -> np.ndarray:
    out = rgb.copy() if rgb.ndim == 3 else np.repeat(rgb[..., None], 3, 2)
    for r in v_edges:
        if 0 <= r < out.shape[0]:
            out[r, :] = [0.0, 1.0, 1.0]
    for c in h_edges:
        if 0 <= c < out.shape[1]:
            out[:, c] = [0.0, 1.0, 1.0]
    return out


def build_report(scene, result, out_path: str) -> dict:
    r = result
    alpha_diff = np.abs(r.alpha - scene.alpha_gt)
    regions = region_masks_for_scene(scene.alpha_gt.shape, scene.bg_line_x,
                                     scene.trimap)
    metrics = alpha_metrics(r.alpha, scene.alpha_gt, regions)
    fixed = fixed_point_violation(r.alpha, scene.trimap)
    residue = color_residue(
        r.foreground, r.alpha, scene.foreground_gt, scene.alpha_gt,
        {"black": 0.0, "white": 1.0},
    )

    rows = []
    rows.append(_row([
        _panel(scene.image, "observed image"),
        _panel(scene.alpha_gt, "alpha ground truth"),
        _panel(r.alpha, "alpha estimated"),
        _panel(alpha_diff, "abs alpha error"),
    ]))

    # Fixed vs. unknown segment visual check.
    fixed_vis = scene.image.copy()
    fixed_vis[scene.trimap == 0] *= 0.4
    fixed_vis[scene.trimap == 255] = (
        0.6 * fixed_vis[scene.trimap == 255] + 0.4
        * np.array([0.0, 1.0, 0.0])
    )
    rows.append(_row([
        _panel(fixed_vis, "fixed bg/fg points (red/green dim)"),
        _panel(r.composite_black, "composite on BLACK"),
        _panel(r.composite_white, "composite on WHITE"),
        _panel(_heat_overlay(scene.image, r.uncertain_mask,
                             r.uncertainty_score),
               "uncertain pixels only (red)"),
        _panel(r.uncertainty_score, "uncertainty score (continuous)"),
    ]))

    rows.append(_row([
        _panel(r.foreground[..., 0], "foreground R"),
        _panel(r.foreground[..., 1], "foreground G"),
        _panel(r.foreground[..., 2], "foreground B"),
        _panel(r.evidence, "local colour evidence"),
    ]))

    if r.alpha_global is not None and r.block_info is not None:
        v_edges, h_edges = r.block_info["tile_edges"]
        seam = seam_metrics(r.alpha, r.alpha_global, v_edges, h_edges)
        rows.append(_row([
            _panel(_draw_seams(r.alpha, v_edges, h_edges),
                   "tiled alpha + seam lines"),
            _panel(r.alpha_global, "global alpha (constraint reference)"),
            _panel(np.abs(r.alpha - r.alpha_global), "tiled - global"),
        ]))
    else:
        seam = None

    montage = _stack(rows)
    montage.save(out_path)

    report = {
        "alpha": metrics.as_dict(),
        "fixed_points": fixed,
        "color_residue": residue,
        "uncertain_fraction": float(r.uncertain_mask.mean()),
        "seam": seam,
        "montage": out_path,
    }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build headless validation montage")
    p.add_argument("--outdir", required=True)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=360)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--blocks", action="store_true")
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    scene = make_hair_scene(h=args.height, w=args.width,
                            jpeg_quality=args.jpeg_quality)
    result = run_pipeline(scene.image, scene.trimap, use_blocks=args.blocks)
    montage_path = os.path.join(args.outdir, "report_montage.png")
    import json
    report = build_report(scene, result, montage_path)
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
