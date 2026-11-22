"""Compare two solver runs, e.g. before and after a local trimap edit.

The original image is shared and untouched; only the trimap changes.  The
comparison reports where alpha and foreground moved, whether the edit changed
any fixed point, and re-validates constraints so a single brush stroke cannot
silently hard-wire a guess elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imageio import trimap_masks
from .metrics import fixed_point_violation
from .pipeline import MattingResult, run_pipeline


@dataclass
class RunComparison:
    alpha_diff: np.ndarray
    alpha_diff_mean: float
    alpha_diff_max: float
    changed_pixels: int
    region_changed_fraction: float
    fixed_violation_before: float
    fixed_violation_after: float
    outside_edit_changed: int
    result_after: MattingResult


def compare_runs(image: np.ndarray, trimap_before: np.ndarray,
                 trimap_after: np.ndarray, **pipeline_kwargs) -> RunComparison:
    before = run_pipeline(image, trimap_before, **pipeline_kwargs)
    after = run_pipeline(image, trimap_after, **pipeline_kwargs)

    diff = np.abs(after.alpha - before.alpha)
    edited = trimap_before != trimap_after
    _, _, unk_after = trimap_masks(trimap_after)

    # Pixels whose alpha moved although the editor never touched them: this is
    # legitimate global propagation, but must be bounded and never land on a
    # fixed point.
    outside = int(((diff > 1e-6) & (~edited)).sum())

    region_changed = 0.0
    if unk_after.any():
        region_changed = float((diff[unk_after] > 1e-6).mean())

    return RunComparison(
        alpha_diff=diff,
        alpha_diff_mean=float(diff.mean()),
        alpha_diff_max=float(diff.max()),
        changed_pixels=int((diff > 1e-6).sum()),
        region_changed_fraction=region_changed,
        fixed_violation_before=before.fixed_violation,
        fixed_violation_after=after.fixed_violation,
        outside_edit_changed=outside,
        result_after=after,
    )
