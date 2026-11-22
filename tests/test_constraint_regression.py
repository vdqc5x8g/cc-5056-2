"""Hair-constraint regression tests.

Covers the failure modes reproduced in
``hairmatting/scripts/constraint_audit.py``:

* the unknown-region solve must never move confirmed fg/bg labels (hard
  constraint assembly, matching Levin's elimination form exactly);
* two brushes crossing the same hair tip: selective undo of the FIRST stroke
  releases only that stroke's pixels, per-pixel, and re-threads later
  records instead of resurrecting released constraints;
* a drag crossing tile blocks is one atomic undo step;
* a delayed trimap write-back cannot re-release pixels locked by newer
  strokes, and swapping two write-backs does not change the hair outline;
* tiled solves clamp overlap rings to fractional global alpha, keeping fine
  strands continuous across seams.

Run directly:  python3 tests/test_constraint_regression.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hairmatting.solver.alpha_solve import (  # noqa: E402
    solve_alpha_blocks, solve_alpha_global,
)
from hairmatting.solver.imageio import (  # noqa: E402
    TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN,
)
from hairmatting.solver.laplacian import matting_laplacian  # noqa: E402
from hairmatting.solver.metrics import fixed_point_violation  # noqa: E402
from hairmatting.solver.synthetic import make_hair_scene  # noqa: E402
from hairmatting.solver.trimap_editor import (  # noqa: E402
    TrimapEditor, disk_mask, line_mask,
)


def _hard_reference(image, trimap):
    """Independent hard-constraint reference: L_ff a_f = -L_fc a_c."""
    import scipy.sparse.linalg as spla
    L = matting_laplacian(image)
    bg, fg = trimap == TRIMAP_BG, trimap == TRIMAP_FG
    fixed_idx = np.flatnonzero((bg | fg).ravel())
    fixed_val = np.where(fg.ravel()[fixed_idx], 1.0, 0.0)
    free_idx = np.flatnonzero((trimap == TRIMAP_UNKNOWN).ravel())
    alpha = np.empty(L.shape[0])
    alpha[fixed_idx] = fixed_val
    alpha[free_idx] = spla.spsolve(
        L[free_idx][:, free_idx].tocsr(),
        -(L[free_idx][:, fixed_idx] @ fixed_val))
    return np.clip(alpha, 0.0, 1.0).reshape(trimap.shape)


class HardConstraintAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)

    def test_unknown_solve_matches_hard_reference_exactly(self):
        sc = self.scene
        alpha = solve_alpha_global(sc.image, sc.trimap)
        ref = _hard_reference(sc.image, sc.trimap)
        unk = sc.trimap == TRIMAP_UNKNOWN
        # A soft (lambda) assembly lets confirmed labels drift and the unknown
        # band absorbs the error; the shipped solver must match the hard
        # reference to solver precision.
        self.assertLess(np.abs(alpha - ref)[unk].max(), 1e-9)
        viol = fixed_point_violation(alpha, sc.trimap)
        self.assertEqual(viol["max_violation"], 0.0)
        # The unknown band is genuinely solved, not snapped to labels.
        intermediate = np.sum((alpha[unk] > 0.05) & (alpha[unk] < 0.95))
        self.assertGreater(intermediate, 50)

    def test_no_labels_is_an_honest_failure(self):
        sc = self.scene
        trimap = np.full(sc.trimap.shape, TRIMAP_UNKNOWN, dtype=np.uint8)
        with self.assertRaises(ValueError):
            solve_alpha_global(sc.image, trimap)

    def test_fine_strands_with_boundary_hugging_prompts(self):
        # 1 px strands with fg labels exactly on the strand and bg labels
        # hugging both edges: fixed pixels stay exact and the narrow unknown
        # gaps are solved, not filled.
        h, w = 48, 72
        image = np.full((h, w, 3), 0.55)
        alpha_gt = np.zeros((h, w))
        fg_col = np.zeros((h, w, 3))
        hair = np.array([0.20, 0.12, 0.08])
        for row in (12, 24, 36):
            alpha_gt[row, 8:64] = 0.85
            fg_col[row, 8:64] = hair
        image = fg_col * alpha_gt[..., None] + image * (1 - alpha_gt[..., None])
        trimap = np.full((h, w), TRIMAP_UNKNOWN, dtype=np.uint8)
        for row in (12, 24, 36):
            trimap[row, 8:64] = TRIMAP_FG
            trimap[row - 1, 8:64] = TRIMAP_BG
            trimap[row + 1, 8:64] = TRIMAP_BG
        trimap[:, :4] = TRIMAP_BG

        alpha = solve_alpha_global(image, trimap)
        viol = fixed_point_violation(alpha, trimap)
        self.assertEqual(viol["max_violation"], 0.0)
        # Unknown rows between the strands stay at the backdrop value.
        self.assertLess(alpha[[11, 13, 23, 25, 35, 37], :].max(), 0.6)


class SelectiveUndoTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)
        self.shape = self.scene.trimap.shape
        partial = np.argwhere((self.scene.alpha_gt > 0.4)
                              & (self.scene.alpha_gt < 0.7))
        cy, cx = partial[len(partial) // 2]
        self.tip = (int(cy), int(cx))

    def _two_overlapping_strokes(self):
        """Brush 1 pins background on a hair tip; brush 2 re-opens unknown."""
        editor = TrimapEditor(self.scene.trimap.copy())
        first = disk_mask(self.shape, self.tip, 2)
        second = disk_mask(self.shape, (self.tip[0], self.tip[1] + 3), 3)
        editor.paint(first, TRIMAP_BG)
        editor.erase_to_unknown(second)
        return editor, first, second

    def test_undo_first_stroke_releases_only_its_own_pixels(self):
        editor, first, second = self._two_overlapping_strokes()
        before = self.scene.trimap
        overlap = first & second
        only_first = first & ~second
        only_second = second & ~first
        self.assertTrue(overlap.any() and only_first.any()
                        and only_second.any())

        first_id = editor.stroke_ids[0]
        editor.undo_stroke(first_id)
        tri = editor.trimap
        # Non-overlapping pixels of stroke 1 are fully restored ...
        self.assertTrue((tri[only_first] == before[only_first]).all())
        # ... the overlap keeps stroke 2's label (unknown) ...
        self.assertTrue((tri[overlap] == TRIMAP_UNKNOWN).all())
        # ... and stroke 2's own pixels are untouched.
        self.assertTrue((tri[only_second] == TRIMAP_UNKNOWN).all())

    def test_second_undo_does_not_resurrect_released_constraint(self):
        editor, first, second = self._two_overlapping_strokes()
        before = self.scene.trimap
        first_id, second_id = editor.stroke_ids[:2]
        editor.undo_stroke(first_id)
        editor.undo_stroke(second_id)
        # Both strokes released: every pixel is back to its pre-stroke label;
        # in particular the overlap must NOT resurrect stroke 1's background.
        self.assertTrue(np.array_equal(editor.trimap, before))

    def test_selective_undo_is_itself_undoable(self):
        editor, first, second = self._two_overlapping_strokes()
        snapshot = editor.trimap.copy()
        editor.undo_stroke(editor.stroke_ids[0])
        editor.undo()
        self.assertTrue(np.array_equal(editor.trimap, snapshot))

    def test_unknown_stroke_id_raises(self):
        editor, _, _ = self._two_overlapping_strokes()
        with self.assertRaises(ValueError):
            editor.undo_stroke(999)


class AtomicStrokeTests(unittest.TestCase):
    def test_cross_block_drag_is_a_single_undo_step(self):
        scene = make_hair_scene(h=96, w=144, seed=4)
        editor = TrimapEditor(scene.trimap.copy())
        pts = [(8, 4), (40, 60), (72, 120)]       # crosses 64-px tile seams
        editor.begin_stroke()
        for i in range(len(pts) - 1):
            editor.paint(line_mask(scene.trimap.shape, pts[i], pts[i + 1], 2),
                         TRIMAP_BG)
        editor.end_stroke()
        self.assertFalse(np.array_equal(editor.trimap, scene.trimap))
        editor.undo()
        self.assertTrue(np.array_equal(editor.trimap, scene.trimap))
        self.assertFalse(editor.can_undo)

    def test_grouped_stroke_keeps_live_view_and_final_labels(self):
        scene = make_hair_scene(h=48, w=48, seed=5)
        editor = TrimapEditor(scene.trimap.copy())
        m1 = disk_mask(scene.trimap.shape, (10, 10), 3)
        m2 = disk_mask(scene.trimap.shape, (30, 30), 3)
        editor.begin_stroke()
        editor.paint(m1, TRIMAP_BG)
        editor.erase_to_unknown(m2)
        # Live view during the drag already shows both segments.
        self.assertTrue((editor.trimap[m1] == TRIMAP_BG).all())
        self.assertTrue((editor.trimap[m2] == TRIMAP_UNKNOWN).all())
        editor.end_stroke()
        self.assertTrue((editor.trimap[m1] == TRIMAP_BG).all())
        editor.undo()
        self.assertTrue(np.array_equal(editor.trimap, scene.trimap))

    def test_operations_during_open_stroke_fail_honestly(self):
        scene = make_hair_scene(h=32, w=32, seed=6)
        editor = TrimapEditor(scene.trimap.copy())
        editor.begin_stroke()
        with self.assertRaises(RuntimeError):
            editor.undo()
        with self.assertRaises(RuntimeError):
            editor.begin_stroke()
        editor.end_stroke()


class GuardedWriteBackTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)

    def test_stale_write_back_cannot_release_locks(self):
        sc = self.scene
        editor = TrimapEditor(sc.trimap.copy())
        stale = editor.trimap.copy()              # computed before the locks
        base = editor.version
        lock = disk_mask(sc.trimap.shape, (20, 20), 3)
        editor.paint(lock, TRIMAP_BG)             # newer stroke locks pixels
        editor.apply_external(stale, base_version=base)
        # The delayed map predates the lock: locked pixels survive.
        self.assertTrue((editor.trimap[lock] == TRIMAP_BG).all())

    def test_write_back_applies_to_unlocked_pixels(self):
        sc = self.scene
        editor = TrimapEditor(sc.trimap.copy())
        base = editor.version
        lock = disk_mask(sc.trimap.shape, (20, 20), 3)
        editor.paint(lock, TRIMAP_BG)
        incoming = editor.trimap.copy()
        free = disk_mask(sc.trimap.shape, (60, 100), 4)
        incoming[free] = TRIMAP_FG                # change outside the lock
        incoming[lock] = TRIMAP_UNKNOWN           # stale release attempt
        editor.apply_external(incoming, base_version=base)
        self.assertTrue((editor.trimap[free] == TRIMAP_FG).all())
        self.assertTrue((editor.trimap[lock] == TRIMAP_BG).all())

    def test_write_back_with_future_or_foreign_version_fails(self):
        editor = TrimapEditor(self.scene.trimap.copy())
        with self.assertRaises(ValueError):
            editor.apply_external(editor.trimap.copy(),
                                  base_version=editor.version + 1)

    def test_swapped_write_back_order_keeps_hair_outline(self):
        sc = self.scene
        lock_a = disk_mask(sc.trimap.shape, (30, 40), 3)
        lock_b = disk_mask(sc.trimap.shape, (60, 90), 3)
        region_x = disk_mask(sc.trimap.shape, (10, 110), 3)
        region_y = disk_mask(sc.trimap.shape, (80, 20), 3)

        def run(order):
            ed = TrimapEditor(sc.trimap.copy())
            v0 = ed.version
            # Two async write-backs, both computed from the state at v0
            # (before any lock existed); each would silently release the
            # locks if applied unguarded.
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

        t_forward = run(("wb1", "wb2"))
        t_swapped = run(("wb2", "wb1"))
        # Disjoint write-back regions: order must not change the final map.
        self.assertTrue(np.array_equal(t_forward, t_swapped))
        # Both locks survive in either order; both write-backs land.
        self.assertTrue((t_forward[lock_a] == TRIMAP_BG).all())
        self.assertTrue((t_forward[lock_b] == TRIMAP_FG).all())
        self.assertTrue((t_forward[region_x] == TRIMAP_FG).all())
        self.assertTrue((t_forward[region_y] == TRIMAP_FG).all())
        # The hair outline from the merged map honours every hard constraint.
        alpha = solve_alpha_global(sc.image, t_forward)
        self.assertEqual(
            fixed_point_violation(alpha, t_forward)["max_violation"], 0.0)
        strand = (sc.alpha_gt > 0.05) & (sc.alpha_gt < 0.95)
        self.assertLess(np.abs(alpha - sc.alpha_gt)[strand].mean(), 0.35)


class TiledRingConstraintTests(unittest.TestCase):
    def test_overlap_ring_uses_fractional_global_alpha(self):
        sc = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)
        br = solve_alpha_blocks(sc.image, sc.trimap, block=64, overlap=12)
        self.assertEqual(br.fixed_violation, 0.0)
        strand = (sc.alpha_gt > 0.05) & (sc.alpha_gt < 0.95)
        resid = np.abs(br.alpha - br.alpha_global)
        # Binarized ring labels bent strands at seams by up to ~0.49; the
        # fractional clamp keeps tiled and global alpha aligned on strands.
        self.assertLess(resid[strand].max(), 0.15)
        self.assertLess(resid[strand].mean(), 0.02)
        self.assertLess(br.seam_residual.mean(), 0.02)

    def test_tile_solve_never_relaxes_fixed_points(self):
        sc = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)
        br = solve_alpha_blocks(sc.image, sc.trimap, block=64, overlap=12)
        bg = sc.trimap == TRIMAP_BG
        fg = sc.trimap == TRIMAP_FG
        self.assertTrue((br.alpha[bg] == 0.0).all())
        self.assertTrue((br.alpha[fg] == 1.0).all())
        self.assertTrue((br.alpha_global[bg] == 0.0).all())
        self.assertTrue((br.alpha_global[fg] == 1.0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
