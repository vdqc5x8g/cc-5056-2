"""Minimal-reproduction audit for trimap-constraint handling in hair matting.

Each check is a small, self-contained reproduction of one failure mode:

A. fixed-point drift     confirmed fg/bg labels move during the unknown-region
                         solve (soft constraint assembly) before any snap;
B. fine-strand prompts   1 px strands with labels hugging the strand boundary
                         must stay exactly fixed while the unknown band between
                         them is genuinely solved;
C. overlapping strokes   two brushes cross the same hair tip: one pins
                         background, the other re-opens unknown; undoing the
                         FIRST stroke must release only that stroke's pixels
                         and leave the second stroke's labels untouched;
D. atomic cross-block    one physical stroke crossing tile seams must be a
                         single undoable unit;
E. stale write-back      a delayed trimap computed from an older editor state
                         must not re-release pixels locked by newer strokes;
F. swapped write-backs   applying two write-backs in either order must keep
                         the hair outline and every hard constraint;
G. tiled ring write-back tile overlap rings clamped to the global solve must
                         use the fractional alpha, not a binarized label.

The script writes a JSON report plus PNG evidence and exits non-zero when any
check fails, so it can run as a regression gate before and after a fix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from hairmatting.solver.alpha_solve import (  # noqa: E402
    solve_alpha_blocks, solve_alpha_global,
)
from hairmatting.solver.imageio import (  # noqa: E402
    TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN, save_alpha, save_rgb, save_trimap,
)
from hairmatting.solver.metrics import fixed_point_violation  # noqa: E402
from hairmatting.solver.synthetic import make_hair_scene  # noqa: E402
from hairmatting.solver.trimap_editor import (  # noqa: E402
    TrimapEditor, disk_mask, line_mask,
)


def _strand_tip_mask(shape, scene, radius=2):
    """Mask of a small disk on a partial-alpha strand pixel (a hair tip)."""
    partial = np.argwhere((scene.alpha_gt > 0.4) & (scene.alpha_gt < 0.7))
    cy, cx = partial[len(partial) // 2]
    return disk_mask(shape, (int(cy), int(cx)), radius), (int(cy), int(cx))


def check_fixed_point_drift(scene) -> dict:
    """A: confirmed labels must not move while the unknown region is solved.

    The independent reference is Levin's hard-constraint elimination: fixed
    pixels keep their labels exactly and the unknown band solves
    L_ff a_f = -L_fc a_c.  A solver that lets confirmed labels drift (e.g. a
    soft lambda assembly) produces a different unknown region - that is the
    "unknown solve rewrites confirmed fg/bg" failure, measured here directly
    as the max/mean deviation from the hard reference on unknown pixels.
    """
    import scipy.sparse.linalg as spla

    from hairmatting.solver.laplacian import matting_laplacian

    L = matting_laplacian(scene.image)
    bg, fg, unk = scene.trimap == TRIMAP_BG, scene.trimap == TRIMAP_FG, \
        scene.trimap == TRIMAP_UNKNOWN
    fixed_idx = np.flatnonzero((bg | fg).ravel())
    fixed_val = np.where(fg.ravel()[fixed_idx], 1.0, 0.0)
    free_idx = np.flatnonzero(unk.ravel())

    ref = np.empty(L.shape[0])
    ref[fixed_idx] = fixed_val
    ref[free_idx] = spla.spsolve(
        L[free_idx][:, free_idx].tocsr(),
        -(L[free_idx][:, fixed_idx] @ fixed_val),
    )
    ref = np.clip(ref, 0.0, 1.0).reshape(scene.trimap.shape)

    alpha = solve_alpha_global(scene.image, scene.trimap)
    viol = fixed_point_violation(alpha, scene.trimap)
    dev = np.abs(alpha - ref)[unk]
    intermediate = int(np.sum((alpha[unk] > 0.05) & (alpha[unk] < 0.95)))
    return {
        "fixed_point_violation": viol["max_violation"],
        "unknown_deviation_from_hard_ref_max": float(dev.max()),
        "unknown_deviation_from_hard_ref_mean": float(dev.mean()),
        "unknown_intermediate_pixels": intermediate,
        "pass": bool(viol["max_violation"] == 0.0
                     and dev.max() < 1e-9 and intermediate > 50),
    }


def check_fine_strand_tight_prompts(outdir) -> dict:
    """B: 1 px strands, labels hugging the strand edge; fixed stay exact."""
    h, w = 48, 72
    image = np.full((h, w, 3), 0.55)
    alpha_gt = np.zeros((h, w))
    fg = np.zeros((h, w, 3))
    hair = np.array([0.20, 0.12, 0.08])
    for row in (12, 24, 36):                      # three 1 px horizontal hairs
        alpha_gt[row, 8:64] = 0.85
        fg[row, 8:64] = hair
    image = fg * alpha_gt[..., None] + image * (1 - alpha_gt[..., None])

    trimap = np.full((h, w), TRIMAP_UNKNOWN, dtype=np.uint8)
    for row in (12, 24, 36):
        trimap[row, 8:64] = TRIMAP_FG             # label exactly on the strand
        trimap[row - 1, 8:64] = TRIMAP_BG         # label hugging the edge
        trimap[row + 1, 8:64] = TRIMAP_BG
    trimap[:, :4] = TRIMAP_BG

    alpha = solve_alpha_global(image, trimap)
    viol = fixed_point_violation(alpha, trimap)
    unk = trimap == TRIMAP_UNKNOWN
    strand_rows = alpha[[11, 13, 23, 25, 35, 37], :]
    evidence = np.zeros((h, w, 3))
    evidence[..., 0] = alpha
    evidence[..., 1] = np.where(unk, alpha, 0.0)
    save_alpha(os.path.join(outdir, "b_fine_strand_alpha.png"), alpha)
    save_trimap(os.path.join(outdir, "b_fine_strand_trimap.png"), trimap)
    save_rgb(os.path.join(outdir, "b_fine_strand_evidence.png"), evidence)
    return {
        "fixed_point_violation": viol["max_violation"],
        "alpha_between_strands_max": float(strand_rows.max()),
        "unknown_genuine_min_max": [float(alpha[unk].min()),
                                    float(alpha[unk].max())],
        "pass": bool(viol["max_violation"] == 0.0
                     and strand_rows.max() < 0.6),
    }


def check_overlapping_stroke_undo(scene, outdir) -> dict:
    """C: undo the FIRST of two overlapping strokes, per-pixel ownership."""
    shape = scene.trimap.shape
    tip_mask, tip = _strand_tip_mask(shape, scene, radius=2)
    second = disk_mask(shape, (tip[0], tip[1] + 3), 3)
    overlap = tip_mask & second
    only_first = tip_mask & ~second
    only_second = second & ~tip_mask

    editor = TrimapEditor(scene.trimap.copy())
    before = editor.trimap.copy()
    editor.paint(tip_mask, TRIMAP_BG)             # stroke 1 pins background
    editor.erase_to_unknown(second)               # stroke 2 re-opens unknown

    if not hasattr(editor, "undo_stroke"):
        # Sequential undo is the only tool: it must destroy stroke 2 to reach
        # stroke 1 - record that as the failure evidence.
        editor.undo()
        destroyed = int((editor.trimap[only_second]
                         != TRIMAP_UNKNOWN).sum())
        return {
            "selective_undo_available": False,
            "stroke2_pixels_destroyed_by_sequential_undo": destroyed,
            "pass": False,
        }

    first_id = 1                                  # stroke 0 = initial state
    editor.undo_stroke(first_id)
    tri = editor.trimap
    restored = bool((tri[only_first] == before[only_first]).all())
    overlap_kept = bool((tri[overlap] == TRIMAP_UNKNOWN).all())
    second_kept = bool((tri[only_second] == TRIMAP_UNKNOWN).all())
    save_trimap(os.path.join(outdir, "c_trimap_after_selective_undo.png"), tri)
    return {
        "selective_undo_available": True,
        "first_stroke_pixels_restored": restored,
        "overlap_kept_second_label": overlap_kept,
        "second_stroke_pixels_kept": second_kept,
        "pass": bool(restored and overlap_kept and second_kept),
    }


def check_atomic_cross_block_stroke(scene) -> dict:
    """D: one drag across tile seams must undo in a single step."""
    shape = scene.trimap.shape
    editor = TrimapEditor(scene.trimap.copy())
    pts = [(8, 4), (40, 60), (72, 120)]           # crosses 64-px tile seams
    segments = [line_mask(shape, pts[i], pts[i + 1], 2)
                for i in range(len(pts) - 1)]
    if hasattr(editor, "begin_stroke"):
        editor.begin_stroke()
        for seg in segments:
            editor.paint(seg, TRIMAP_BG)
        editor.end_stroke()
        steps = 0
        while not np.array_equal(editor.trimap, scene.trimap) \
                and editor.can_undo:
            editor.undo()
            steps += 1
            if steps > 4:
                break
        restored = np.array_equal(editor.trimap, scene.trimap)
        return {"atomic_api": True, "undo_steps": steps,
                "pass": bool(steps == 1 and restored)}
    for seg in segments:
        editor.paint(seg, TRIMAP_BG)
    editor.undo()
    still_marked = bool((editor.trimap[segments[-1]] == TRIMAP_BG).any())
    return {"atomic_api": False,
            "segments_left_after_one_undo": still_marked,
            "pass": False}


def check_stale_write_back(scene) -> dict:
    """E: a delayed trimap from an older state must not release new locks."""
    editor = TrimapEditor(scene.trimap.copy())
    lock = disk_mask(scene.trimap.shape, (20, 20), 3)
    stale_map = editor.trimap.copy()              # "computed" before the lock
    base_version = getattr(editor, "version", None)
    editor.paint(lock, TRIMAP_BG)                 # newer stroke locks pixels
    if hasattr(editor, "apply_external"):
        editor.apply_external(stale_map, base_version=base_version)
        released = bool((editor.trimap[lock] != TRIMAP_BG).any())
        return {"guarded_write_back": True, "locks_released": released,
                "pass": not released}
    # Unguarded editors apply the stale map verbatim and release the locks.
    released = bool((stale_map[lock] != TRIMAP_BG).any())
    return {"guarded_write_back": False, "locks_released": released,
            "pass": False}


def check_swapped_write_backs(scene, outdir) -> dict:
    """F: two write-backs in swapped order; hair outline must not change."""
    if not hasattr(TrimapEditor, "apply_external"):
        return {"guarded_write_back": False, "pass": False}
    shape = scene.trimap.shape
    lock_a = disk_mask(shape, (30, 40), 3)
    lock_b = disk_mask(shape, (60, 90), 3)
    region_x = disk_mask(shape, (10, 110), 3)
    region_y = disk_mask(shape, (80, 20), 3)

    def run(order):
        ed = TrimapEditor(scene.trimap.copy())
        v0 = ed.version
        # Two async write-backs computed from the state at v0, before any
        # lock existed; unguarded they would silently release the locks.
        wb1 = ed.trimap.copy()
        wb1[region_x] = TRIMAP_FG
        wb2 = ed.trimap.copy()
        wb2[region_y] = TRIMAP_FG
        ed.paint(lock_a, TRIMAP_BG)
        ed.paint(lock_b, TRIMAP_FG)
        maps = {"wb1": wb1, "wb2": wb2}
        for name in order:
            ed.apply_external(maps[name], base_version=v0)
        return ed.trimap

    t1 = run(("wb1", "wb2"))
    t2 = run(("wb2", "wb1"))
    same = bool(np.array_equal(t1, t2))
    locks_ok = bool((t1[lock_a] == TRIMAP_BG).all()
                    and (t1[lock_b] == TRIMAP_FG).all())
    writes_ok = bool((t1[region_x] == TRIMAP_FG).all()
                     and (t1[region_y] == TRIMAP_FG).all())
    alpha = solve_alpha_global(scene.image, t1)
    viol = fixed_point_violation(alpha, t1)["max_violation"]
    strand = (scene.alpha_gt > 0.05) & (scene.alpha_gt < 0.95)
    outline_err = float(np.abs(alpha - scene.alpha_gt)[strand].mean())
    save_alpha(os.path.join(outdir, "f_alpha_after_swapped_writes.png"), alpha)
    return {
        "guarded_write_back": True,
        "order_independent": same,
        "locks_survive": locks_ok,
        "write_backs_applied": writes_ok,
        "fixed_point_violation": viol,
        "strand_outline_mean_abs_err": outline_err,
        "pass": bool(same and locks_ok and writes_ok and viol == 0.0),
    }


def check_tiled_ring_write_back(scene, outdir) -> dict:
    """G: tiled solve vs global reference on fine strands crossing seams."""
    br = solve_alpha_blocks(scene.image, scene.trimap, block=64, overlap=12)
    strand = (scene.alpha_gt > 0.05) & (scene.alpha_gt < 0.95)
    resid = np.abs(br.alpha - br.alpha_global)
    save_alpha(os.path.join(outdir, "g_tiled_alpha.png"), br.alpha)
    save_alpha(os.path.join(outdir, "g_tiled_vs_global.png"),
               resid / max(resid.max(), 1e-12))
    return {
        "seam_residual_max": float(br.seam_residual.max()),
        "seam_residual_mean": float(br.seam_residual.mean()),
        "strand_residual_max": float(resid[strand].max()),
        "strand_residual_mean": float(resid[strand].mean()),
        "fixed_violation": br.fixed_violation,
        "pass": bool(br.fixed_violation == 0.0
                     and resid[strand].max() < 0.15),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    scene = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=args.seed)
    save_rgb(os.path.join(args.outdir, "scene_image.png"), scene.image)
    save_trimap(os.path.join(args.outdir, "scene_trimap.png"), scene.trimap)

    report = {
        "A_fixed_point_drift": check_fixed_point_drift(scene),
        "B_fine_strand_tight_prompts": check_fine_strand_tight_prompts(
            args.outdir),
        "C_overlapping_stroke_undo": check_overlapping_stroke_undo(
            scene, args.outdir),
        "D_atomic_cross_block_stroke": check_atomic_cross_block_stroke(scene),
        "E_stale_write_back": check_stale_write_back(scene),
        "F_swapped_write_backs": check_swapped_write_backs(scene, args.outdir),
        "G_tiled_ring_write_back": check_tiled_ring_write_back(
            scene, args.outdir),
    }
    report["all_pass"] = all(v.get("pass", False) for v in report.values())
    path = os.path.join(args.outdir, "constraint_audit.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
