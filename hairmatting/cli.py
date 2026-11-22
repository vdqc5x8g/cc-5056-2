"""Command-line workbench: real solving, cross-background validation, export.

Subcommands
-----------
solve      Run the matting pipeline on an image + trimap and export alpha,
           foreground RGBA, black/white composites and the uncertainty map.
synthetic  Generate the curtain-to-flat-region hair scene (with optional JPEG
           noise) and write ground truth + trimap for offline inspection.
validate   End-to-end quantitative validation on synthetic scenes: alpha error
           split by region, fixed-point violations, tiled-vs-global seam
           metrics, background-swap colour residue, trimap immutability and
           undo/redo behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hairmatting.solver.imageio import (  # noqa: E402
    composite, load_rgb, load_trimap, save_alpha, save_rgb, save_rgba,
    save_trimap,
)
from hairmatting.solver.metrics import (  # noqa: E402
    alpha_metrics, color_residue, fixed_point_violation, region_masks_for_scene,
    seam_metrics,
)
from hairmatting.solver.pipeline import run_pipeline  # noqa: E402
from hairmatting.solver.compare import compare_runs  # noqa: E402
from hairmatting.solver.synthetic import make_hair_scene  # noqa: E402
from hairmatting.solver.trimap_editor import (  # noqa: E402
    TrimapEditor, disk_mask,
)


def _save_uncertainty(path: str, score: np.ndarray,
                      mask: np.ndarray) -> None:
    # Red overlay semantics: intensity = uncertainty score; mask shown at 255.
    vis = np.zeros((*score.shape, 3), dtype=np.float64)
    vis[..., 0] = score
    vis[mask, 0] = 1.0
    save_rgb(path, vis)


def cmd_solve(args) -> None:
    image = load_rgb(args.image)
    trimap = load_trimap(args.trimap)
    if image.shape[:2] != trimap.shape:
        raise SystemExit("image and trimap dimensions differ")
    result = run_pipeline(
        image, trimap, use_blocks=args.blocks, block=args.block,
        overlap=args.overlap,
    )
    os.makedirs(args.outdir, exist_ok=True)
    save_alpha(os.path.join(args.outdir, "alpha.png"), result.alpha)
    save_rgba(os.path.join(args.outdir, "foreground_rgba.png"),
              result.foreground, result.alpha)
    save_rgb(os.path.join(args.outdir, "composite_black.png"),
             result.composite_black)
    save_rgb(os.path.join(args.outdir, "composite_white.png"),
             result.composite_white)
    _save_uncertainty(os.path.join(args.outdir, "uncertainty.png"),
                      result.uncertainty_score, result.uncertain_mask)
    save_rgb(os.path.join(args.outdir, "evidence.png"),
             np.repeat(result.evidence[..., None], 3, axis=2))

    report = {
        "fixed_point_violation": result.fixed_violation,
        "uncertain_fraction": float(result.uncertain_mask.mean()),
        "alpha_mean_unknown": float(
            result.alpha[trimap == 128].mean() if (trimap == 128).any() else 0.0
        ),
    }
    if result.block_info:
        report["block"] = result.block_info
    with open(os.path.join(args.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def cmd_synthetic(args) -> None:
    scene = make_hair_scene(h=args.height, w=args.width,
                            jpeg_quality=args.jpeg_quality, seed=args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    save_rgb(os.path.join(args.outdir, "image.png"), scene.image)
    save_rgb(os.path.join(args.outdir, "image_clean.png"), scene.image_clean)
    save_alpha(os.path.join(args.outdir, "alpha_gt.png"), scene.alpha_gt)
    save_rgb(os.path.join(args.outdir, "foreground_gt.png"), scene.foreground_gt)
    save_rgb(os.path.join(args.outdir, "background_gt.png"), scene.background_gt)
    save_trimap(os.path.join(args.outdir, "trimap.png"), scene.trimap)
    print(f"scene written to {args.outdir} (bg line x={scene.bg_line_x})")


def cmd_validate(args) -> None:
    report = {}
    for quality in args.jpeg_qualities:
        scene = make_hair_scene(h=args.height, w=args.width,
                                jpeg_quality=quality, seed=args.seed)
        regions = region_masks_for_scene(scene.alpha_gt.shape,
                                         scene.bg_line_x, scene.trimap)
        result = run_pipeline(scene.image, scene.trimap,
                              use_blocks=args.blocks, block=args.block,
                              overlap=args.overlap)
        metrics = alpha_metrics(result.alpha, scene.alpha_gt, regions)
        fixed = fixed_point_violation(result.alpha, scene.trimap)
        residue = color_residue(
            result.foreground, result.alpha,
            scene.foreground_gt, scene.alpha_gt,
            {"black": 0.0, "white": 1.0},
        )
        entry = {
            "alpha": metrics.as_dict(),
            "fixed_points": fixed,
            "color_residue": residue,
            "uncertain_fraction": float(result.uncertain_mask.mean()),
        }
        if args.blocks:
            v_edges, h_edges = result.block_info["tile_edges"]
            entry["seam"] = seam_metrics(
                result.alpha, result.alpha_global, v_edges, h_edges
            )

        # Trimap immutability: solving must never write guesses into the map.
        before = scene.trimap.copy()
        _ = run_pipeline(scene.image, scene.trimap)
        entry["trimap_mutated_by_solver"] = bool(
            not np.array_equal(before, scene.trimap)
        )

        # Hard background stroke with full undo round-trip.
        editor = TrimapEditor(scene.trimap.copy())
        h, w = scene.trimap.shape
        stroke = disk_mask((h, w), (h // 2, w // 4), radius=6)
        editor.paint(stroke, 0)
        entry["stroke_changes_pixels"] = int(
            (editor.trimap != scene.trimap).sum()
        )
        editor.undo()
        entry["undo_restores_trimap"] = bool(
            np.array_equal(editor.trimap, scene.trimap)
        )

        report[f"jpeg_q{quality}"] = entry

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, "validation.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"validation report written to {path}")
    print(json.dumps(report, indent=2))


def cmd_compare(args) -> None:
    """Re-solve after a local trimap edit and quantify the difference."""
    image = load_rgb(args.image)
    before = load_trimap(args.trimap_before)
    after = load_trimap(args.trimap_after)
    if image.shape[:2] != before.shape or before.shape != after.shape:
        raise SystemExit("image and trimaps must share dimensions")
    comparison = compare_runs(image, before, after, use_blocks=args.blocks,
                              block=args.block, overlap=args.overlap)
    r = comparison.result_after
    report = {
        "alpha_diff_mean": comparison.alpha_diff_mean,
        "alpha_diff_max": comparison.alpha_diff_max,
        "changed_pixels": comparison.changed_pixels,
        "unknown_region_changed_fraction":
            comparison.region_changed_fraction,
        "pixels_changed_outside_edit": comparison.outside_edit_changed,
        "fixed_violation_before": comparison.fixed_violation_before,
        "fixed_violation_after": comparison.fixed_violation_after,
        "uncertain_fraction_after": float(r.uncertain_mask.mean()),
    }
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        save_alpha(os.path.join(args.outdir, "alpha_after.png"), r.alpha)
        save_alpha(os.path.join(args.outdir, "alpha_diff.png"),
                   comparison.alpha_diff)
        save_rgb(os.path.join(args.outdir, "composite_black_after.png"),
                 r.composite_black)
        save_rgb(os.path.join(args.outdir, "composite_white_after.png"),
                 r.composite_white)
        with open(os.path.join(args.outdir, "compare.json"), "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hair matting workbench")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("solve", help="solve an image/trimap pair")
    ps.add_argument("--image", required=True)
    ps.add_argument("--trimap", required=True)
    ps.add_argument("--outdir", required=True)
    ps.add_argument("--blocks", action="store_true")
    ps.add_argument("--block", type=int, default=160)
    ps.add_argument("--overlap", type=int, default=32)
    ps.set_defaults(func=cmd_solve)

    pg = sub.add_parser("synthetic", help="generate synthetic hair scene")
    pg.add_argument("--outdir", required=True)
    pg.add_argument("--height", type=int, default=320)
    pg.add_argument("--width", type=int, default=480)
    pg.add_argument("--jpeg-quality", type=int, default=None)
    pg.add_argument("--seed", type=int, default=0)
    pg.set_defaults(func=cmd_synthetic)

    pv = sub.add_parser("validate", help="quantitative validation suite")
    pv.add_argument("--outdir", default=None)
    pv.add_argument("--height", type=int, default=200)
    pv.add_argument("--width", type=int, default=300)
    pv.add_argument("--jpeg-qualities", type=int, nargs="+",
                    default=[None, 90, 70, 50])
    pv.add_argument("--blocks", action="store_true")
    pv.add_argument("--block", type=int, default=120)
    pv.add_argument("--overlap", type=int, default=24)
    pv.add_argument("--seed", type=int, default=0)
    pv.set_defaults(func=cmd_validate)

    pc = sub.add_parser("compare",
                        help="re-solve and compare after a trimap edit")
    pc.add_argument("--image", required=True)
    pc.add_argument("--trimap-before", required=True)
    pc.add_argument("--trimap-after", required=True)
    pc.add_argument("--outdir", default=None)
    pc.add_argument("--blocks", action="store_true")
    pc.add_argument("--block", type=int, default=160)
    pc.add_argument("--overlap", type=int, default=32)
    pc.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
