"""Quantitative validation metrics for matting results."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imageio import composite, trimap_masks


@dataclass
class AlphaMetrics:
    sad: float                       # sum of absolute differences
    mse: float
    max_error: float
    gradient_error: float
    per_region: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"sad": self.sad, "mse": self.mse, "max_error": self.max_error,
             "gradient_error": self.gradient_error}
        d.update({f"region:{k}_mse": v for k, v in self.per_region.items()})
        return d


def alpha_metrics(alpha_est: np.ndarray, alpha_gt: np.ndarray,
                  region_masks: dict[str, np.ndarray] | None = None
                  ) -> AlphaMetrics:
    err = np.abs(alpha_est - alpha_gt)
    gy, gx = np.gradient(alpha_est - alpha_gt)
    grad_err = np.sqrt(gy ** 2 + gx ** 2).mean()
    per_region = {}
    if region_masks:
        for name, mask in region_masks.items():
            if mask.any():
                per_region[name] = float(((alpha_est - alpha_gt) ** 2)[mask].mean())
    return AlphaMetrics(
        sad=float(err.sum()),
        mse=float((err ** 2).mean()),
        max_error=float(err.max()),
        gradient_error=float(grad_err),
        per_region=per_region,
    )


def color_residue(fg_est: np.ndarray, alpha_est: np.ndarray,
                  fg_gt: np.ndarray, alpha_gt: np.ndarray,
                  test_backgrounds) -> dict:
    """Composite estimated vs. ground truth onto black and white backgrounds.

    The visible colour residue after a background swap is the honest test of
    foreground-colour quality: a fit that merely matched the original backdrop
    fails immediately on the opposite backdrop.
    """
    out = {}
    for name, bg in test_backgrounds.items():
        comp_est = composite(fg_est, alpha_est, bg)
        comp_gt = composite(fg_gt, alpha_gt, bg)
        diff = np.abs(comp_est - comp_gt)
        out[f"{name}_max"] = float(diff.max())
        out[f"{name}_mean"] = float(diff.mean())
        # Residue weighted toward low-alpha pixels, where F is unobservable.
        low = alpha_gt < 0.05
        if low.any():
            out[f"{name}_lowalpha_mean"] = float(diff[low].mean())
    return out


def fixed_point_violation(alpha: np.ndarray, trimap: np.ndarray) -> dict:
    bg, fg, _ = trimap_masks(trimap)
    return {
        "bg_max_alpha": float(alpha[bg].max()) if bg.any() else 0.0,
        "fg_min_alpha": float(alpha[fg].min()) if fg.any() else 1.0,
        "max_violation": float(max(
            alpha[bg].max(initial=0.0),
            (1.0 - alpha[fg]).max(initial=0.0),
        )),
    }


def seam_metrics(alpha_tiled: np.ndarray, alpha_global: np.ndarray,
                 vertical_edges: list[int], horizontal_edges: list[int]
                 ) -> dict:
    """Compare overlapping-window tiled solve against the global constraint."""
    h, w = alpha_tiled.shape
    rows = []
    for e in vertical_edges:
        if 0 < e < h:
            rows.append(e)
    cols = []
    for e in horizontal_edges:
        if 0 < e < w:
            cols.append(e)
    mask = np.zeros((h, w), dtype=bool)
    for r in rows:
        mask[r, :] = True
    for c in cols:
        mask[:, c] = True
    if not mask.any():
        return {"seam_mean_abs_diff": 0.0, "seam_max_abs_diff": 0.0,
                "seam_jump": 0.0}
    diff = np.abs(alpha_tiled - alpha_global)[mask]
    jumps = []
    for r in rows:
        jumps.append(np.abs(alpha_tiled[r, :] - alpha_tiled[r - 1, :]).mean())
    for c in cols:
        jumps.append(np.abs(alpha_tiled[:, c] - alpha_tiled[:, c - 1]).mean())
    return {
        "seam_mean_abs_diff": float(diff.mean()),
        "seam_max_abs_diff": float(diff.max()),
        "seam_jump": float(np.mean(jumps)) if jumps else 0.0,
    }


def region_masks_for_scene(shape: tuple[int, int], bg_line_x: int,
                           trimap: np.ndarray) -> dict[str, np.ndarray]:
    """Curtain segment vs. same-colour flat segment, restricted to unknown band."""
    h, w = shape
    _, _, unk = trimap_masks(trimap)
    curtain = np.zeros((h, w), dtype=bool)
    curtain[:, :bg_line_x] = True
    flat = ~curtain
    return {
        "curtain_unknown": curtain & unk,
        "flat_unknown": flat & unk,
        "all_unknown": unk,
    }
