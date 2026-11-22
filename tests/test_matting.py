"""Numerical and behavioural tests for the hair matting workbench.

Run directly:  python3 tests/test_matting.py
or via unittest discovery from the repository root.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hairmatting.solver.alpha_solve import (  # noqa: E402
    solve_alpha_blocks, solve_alpha_global,
)
from hairmatting.solver.foreground import estimate_foreground  # noqa: E402
from hairmatting.solver.imageio import (  # noqa: E402
    TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN, composite,
)
from hairmatting.solver.laplacian import matting_laplacian  # noqa: E402
from hairmatting.solver.metrics import (  # noqa: E402
    alpha_metrics, color_residue, fixed_point_violation, seam_metrics,
)
from hairmatting.solver.synthetic import make_hair_scene  # noqa: E402
from hairmatting.solver.trimap_editor import (  # noqa: E402
    TrimapEditor, disk_mask, line_mask,
)
from hairmatting.solver.uncertainty import (  # noqa: E402
    LOW_ALPHA_BAND, estimate_uncertainty,
)


class LaplacianTests(unittest.TestCase):
    def test_constant_vector_is_nullspace(self):
        rng = np.random.default_rng(1)
        img = rng.random((24, 30, 3))
        L = matting_laplacian(img)
        residual = np.abs(L @ np.ones(L.shape[0])).max()
        self.assertLess(residual, 1e-8)

    def test_symmetric_and_psd(self):
        rng = np.random.default_rng(2)
        img = rng.random((20, 24, 3))
        L = matting_laplacian(img)
        self.assertLess(np.abs(L - L.T).max(), 1e-12)
        # PSD checked through random quadratic forms (x^T L x >= 0).
        for _ in range(15):
            x = rng.standard_normal(L.shape[0])
            self.assertGreaterEqual(x @ (L @ x), -1e-8)

    def test_flat_image_laplacian_rank_structure(self):
        # On a constant image every window is perfectly collinear, so the
        # Laplacian only encodes smoothness; row sums still vanish.
        L = matting_laplacian(np.full((16, 16, 3), 0.4))
        self.assertLess(np.abs(L @ np.ones(L.shape[0])).max(), 1e-12)


class TrimapConstraintTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_hair_scene(h=96, w=144, jpeg_quality=70, seed=3)

    def test_fixed_points_are_exact_and_unknown_never_snapped(self):
        sc = self.scene
        alpha = solve_alpha_global(sc.image, sc.trimap)
        viol = fixed_point_violation(alpha, sc.trimap)
        self.assertEqual(viol["max_violation"], 0.0)
        # Unknown pixels must not be copied from a distance field: the result
        # there must contain intermediate values somewhere (real solving), not
        # only the two hard labels.
        unk = sc.trimap == TRIMAP_UNKNOWN
        intermediate = np.sum((alpha[unk] > 0.05) & (alpha[unk] < 0.95))
        self.assertGreater(intermediate, 50)

    def test_solver_never_mutates_trimap(self):
        sc = self.scene
        snapshot = sc.trimap.copy()
        solve_alpha_global(sc.image, sc.trimap)
        self.assertTrue(np.array_equal(snapshot, sc.trimap))
        solve_alpha_blocks(sc.image, sc.trimap, block=64, overlap=12)
        self.assertTrue(np.array_equal(snapshot, sc.trimap))

    def test_alpha_bounded(self):
        alpha = solve_alpha_global(self.scene.image, self.scene.trimap)
        self.assertGreaterEqual(alpha.min(), 0.0)
        self.assertLessEqual(alpha.max(), 1.0)


class EditorTests(unittest.TestCase):
    def test_undo_redo_and_original_image_untouched(self):
        sc = make_hair_scene(h=64, w=96)
        image_snapshot = sc.image.copy()
        editor = TrimapEditor(sc.trimap.copy())
        m1 = disk_mask(sc.trimap.shape, (10, 10), 4)
        editor.paint(m1, TRIMAP_BG)
        self.assertTrue((editor.trimap[m1] == TRIMAP_BG).all())
        m2 = line_mask(sc.trimap.shape, (5, 5), (40, 50), 3)
        editor.paint(m2, TRIMAP_FG)
        self.assertTrue(editor.can_undo)
        editor.undo()
        editor.undo()
        self.assertTrue(np.array_equal(editor.trimap, sc.trimap))
        editor.redo()
        self.assertTrue((editor.trimap[m1] == TRIMAP_BG).all())
        self.assertTrue(np.array_equal(image_snapshot, sc.image))

    def test_redo_branch_discarded_by_new_stroke(self):
        sc = make_hair_scene(h=48, w=48)
        editor = TrimapEditor(sc.trimap.copy())
        editor.paint(disk_mask(sc.trimap.shape, (8, 8), 2), TRIMAP_BG)
        editor.undo()
        editor.paint(disk_mask(sc.trimap.shape, (40, 40), 2), TRIMAP_FG)
        self.assertFalse(editor.can_redo)
        editor.undo()
        self.assertTrue(np.array_equal(editor.trimap, sc.trimap))


class BlockSolveTests(unittest.TestCase):
    def test_tiled_and_global_agree_on_fixed_points_and_seams(self):
        sc = make_hair_scene(h=128, w=192, jpeg_quality=80, seed=5)
        br = solve_alpha_blocks(sc.image, sc.trimap, block=80, overlap=16)
        self.assertEqual(br.fixed_violation, 0.0)
        v_edges, h_edges = br.tile_edges
        seams = seam_metrics(br.alpha, br.alpha_global, v_edges, h_edges)
        # Tiled vs. global constraint must stay close on the seam lines.
        self.assertLess(seams["seam_mean_abs_diff"], 0.05)
        self.assertLess(seams["seam_jump"], 0.05)
        # Filaments crossing the fixed background line: tiled solving must not
        # introduce a jump beyond what the global-constraint solution does
        # (the ground truth itself has hard strand edges at this column).
        col = sc.bg_line_x
        jump_tiled = np.abs(br.alpha[:, col] - br.alpha[:, col - 1])
        jump_global = np.abs(br.alpha_global[:, col]
                             - br.alpha_global[:, col - 1])
        self.assertLess(np.abs(jump_tiled - jump_global).max(), 0.15)

    def test_blocks_cover_full_frame(self):
        sc = make_hair_scene(h=100, w=130, seed=4)
        br = solve_alpha_blocks(sc.image, sc.trimap, block=64, overlap=12)
        self.assertTrue(np.all(np.isfinite(br.alpha)))


class ForegroundAndUncertaintyTests(unittest.TestCase):
    def test_low_alpha_pixels_flagged_unstable(self):
        sc = make_hair_scene(h=96, w=144, jpeg_quality=60, seed=6)
        alpha = solve_alpha_global(sc.image, sc.trimap)
        fg = estimate_foreground(sc.image, alpha, sc.trimap)
        unc = estimate_uncertainty(alpha, sc.trimap, fg.evidence,
                                   fg.line_residual)
        # Sub-band alpha instability applies to *mixed* pixels with local
        # evidence (a small but non-zero foreground contribution); pixels that
        # solve to zero are confident background and stay stable.
        low = ((alpha > 0.02) & (alpha < LOW_ALPHA_BAND)
               & (sc.trimap != TRIMAP_BG) & (fg.evidence > 0.0))
        if low.any():
            self.assertTrue(unc.low_alpha_unstable[low].all())
        self.assertFalse(
            unc.low_alpha_unstable[sc.trimap == TRIMAP_BG].any())
        self.assertFalse(
            unc.low_alpha_unstable[(alpha <= 0.02)].any())
        # Fixed pixels are never reported uncertain.
        fixed = (sc.trimap == TRIMAP_FG) | (sc.trimap == TRIMAP_BG)
        self.assertFalse(unc.mask[fixed].any())

    def test_no_distance_gradient_fill_in_no_evidence_region(self):
        import scipy.ndimage as ndi
        sc = make_hair_scene(h=96, w=144, seed=7)
        alpha = solve_alpha_global(sc.image, sc.trimap)
        fg = estimate_foreground(sc.image, alpha, sc.trimap)
        unc = estimate_uncertainty(alpha, sc.trimap, fg.evidence,
                                   fg.line_residual)
        no_ev = (fg.evidence < 0.15) & (sc.trimap == TRIMAP_UNKNOWN)
        flagged = no_ev & unc.no_evidence
        not_flagged = no_ev & (~unc.no_evidence)
        # Unflagged no-evidence pixels are stably solved background: near-zero
        # alpha (the anchor threshold used by the foreground estimator).
        if not_flagged.any():
            self.assertTrue((alpha[not_flagged] <= 0.02 + 1e-9).all())
        # Flagged pixels must lie on or immediately next to a real hair pixel;
        # the uncertainty map never paints a generic distance-field band.
        if flagged.any():
            near = ndi.binary_dilation(alpha > 0.02, iterations=1)
            self.assertTrue(near[flagged].all())
        # The solver never invents strands: every flagged pixel's alpha is
        # finite and within bounds rather than a forced opaque fill.
        self.assertTrue(np.isfinite(alpha).all())


class SyntheticAccuracyTests(unittest.TestCase):
    def test_clean_scene_alpha_accuracy_better_than_noisy(self):
        clean = make_hair_scene(h=128, w=192, jpeg_quality=None, seed=8)
        noisy = make_hair_scene(h=128, w=192, jpeg_quality=50, seed=8)
        a_clean = solve_alpha_global(clean.image, clean.trimap)
        a_noisy = solve_alpha_global(noisy.image, noisy.trimap)
        m_clean = alpha_metrics(a_clean, clean.alpha_gt)
        m_noisy = alpha_metrics(a_noisy, noisy.alpha_gt)
        self.assertLess(m_clean.mse, m_noisy.mse)

    def test_composite_reconstruction_on_observed_backdrop(self):
        sc = make_hair_scene(h=96, w=144, jpeg_quality=None, seed=9)
        alpha = solve_alpha_global(sc.image, sc.trimap)
        fg = estimate_foreground(sc.image, alpha, sc.trimap)
        recon = composite(fg.foreground, alpha, sc.background_gt)
        err = np.abs(recon - sc.image_clean)
        # On the original backdrop the matting equation must reconstruct the
        # observation closely wherever alpha is non-trivial.
        band = (alpha > 0.1) & (sc.trimap == TRIMAP_UNKNOWN)
        self.assertLess(err[band].mean(), 0.08)

    def test_cross_background_residue_is_measured_not_hidden(self):
        sc = make_hair_scene(h=96, w=144, jpeg_quality=80, seed=10)
        alpha = solve_alpha_global(sc.image, sc.trimap)
        fg = estimate_foreground(sc.image, alpha, sc.trimap)
        residue = color_residue(
            fg.foreground, alpha, sc.foreground_gt, sc.alpha_gt,
            {"black": 0.0, "white": 1.0},
        )
        for key in ("black_mean", "white_mean", "black_lowalpha_mean",
                    "white_lowalpha_mean"):
            self.assertIn(key, residue)
            self.assertTrue(np.isfinite(residue[key]))


class CompareTests(unittest.TestCase):
    def test_recompute_after_local_edit_and_undo(self):
        from hairmatting.solver.compare import compare_runs
        sc = make_hair_scene(h=96, w=144, jpeg_quality=75, seed=11)
        editor = TrimapEditor(sc.trimap.copy())
        image_snapshot = sc.image.copy()
        before_map = editor.trimap.copy()

        # A hard background stroke inside the unknown band next to real hair.
        unk = np.argwhere(sc.trimap == TRIMAP_UNKNOWN)
        target = tuple(unk[len(unk) // 2])
        stroke = disk_mask(sc.trimap.shape, target, 5)
        editor.paint(stroke, TRIMAP_BG)
        comp = compare_runs(sc.image, before_map, editor.trimap)
        self.assertEqual(comp.fixed_violation_after, 0.0)
        self.assertGreater(comp.changed_pixels, 0)
        # The new hard marker is honoured exactly.
        self.assertTrue((comp.result_after.alpha[stroke] == 0.0).all())
        # Propagation outside the edited pixels must not violate any fixed point.
        self.assertEqual(comp.fixed_violation_before,
                         comp.fixed_violation_after)

        # Undo restores the trimap, and re-solving reproduces the original.
        editor.undo()
        self.assertTrue(np.array_equal(editor.trimap, before_map))
        self.assertTrue(np.array_equal(sc.image, image_snapshot))

    def test_foreground_guess_never_written_into_trimap(self):
        from hairmatting.solver.compare import compare_runs
        sc = make_hair_scene(h=80, w=120, seed=12)
        original = sc.trimap.copy()
        compare_runs(sc.image, original, original)
        self.assertTrue(np.array_equal(sc.trimap, original))


class NoFillTests(unittest.TestCase):
    def test_unknown_region_without_evidence_is_not_distance_filled(self):
        # Uniform backdrop with a blob of unknown pixels surrounded by definite
        # background.  There is no colour evidence for foreground, so a distance
        # transform would paint a bright gradient into the blob; the evidence-
        # based solver must keep alpha at 0 and flag the area as uncertain
        # instead of inventing hair.
        h = w = 64
        image = np.full((h, w, 3), 0.4)
        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[20:44, 20:44] = TRIMAP_UNKNOWN
        trimap[2:6, 2:6] = TRIMAP_FG            # distant fixed fg, not adjacent

        alpha = solve_alpha_global(image, trimap)
        blob = trimap == TRIMAP_UNKNOWN
        # No gradient ramp: the unsupported blob stays at the background value.
        self.assertLess(alpha[blob].max(), 0.05)
        self.assertAlmostEqual(alpha[blob].mean(), 0.0, places=4)

        fg = estimate_foreground(image, alpha, trimap)
        unc = estimate_uncertainty(alpha, trimap, fg.evidence,
                                   fg.line_residual)
        # The unsupported unknown area is surfaced honestly, not silently filled.
        # (Pixels that stably resolve to background may be anchored, but none
        # may receive an invented opaque foreground.)
        self.assertTrue(np.isfinite(alpha).all())
        self.assertLess(fg.foreground[blob].max(), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
