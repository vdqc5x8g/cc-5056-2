"""End-to-end matting pipeline used by both the CLI and the GUI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alpha_solve import solve_alpha_global, solve_alpha_blocks
from .foreground import estimate_foreground
from .imageio import composite
from .uncertainty import estimate_uncertainty


@dataclass
class MattingResult:
    alpha: np.ndarray
    alpha_global: np.ndarray | None
    foreground: np.ndarray
    evidence: np.ndarray
    uncertainty_score: np.ndarray
    uncertain_mask: np.ndarray
    composite_black: np.ndarray
    composite_white: np.ndarray
    fixed_violation: float
    block_info: dict | None


def run_pipeline(image: np.ndarray, trimap: np.ndarray,
                 use_blocks: bool = False, block: int = 160,
                 overlap: int = 32, radius: int = 1,
                 eps: float = 1e-7, lam: float = 100.0) -> MattingResult:
    if use_blocks:
        br = solve_alpha_blocks(image, trimap, block=block, overlap=overlap,
                                radius=radius, eps=eps, lam=lam)
        alpha = br.alpha
        alpha_global = br.alpha_global
        block_info = {
            "fixed_violation": br.fixed_violation,
            "seam_residual_mean": float(br.seam_residual.mean()),
            "seam_residual_max": float(br.seam_residual.max()),
            "tile_edges": br.tile_edges,
        }
        fixed_violation = br.fixed_violation
    else:
        alpha = solve_alpha_global(image, trimap, radius=radius, eps=eps,
                                   lam=lam)
        alpha_global = None
        block_info = None
        bg = trimap == 0
        fg = trimap == 255
        fixed_violation = float(max(
            alpha[bg].max(initial=0.0),
            (1.0 - alpha[fg]).max(initial=0.0),
        ))

    fg_result = estimate_foreground(image, alpha, trimap, radius=radius)
    unc = estimate_uncertainty(alpha, trimap, fg_result.evidence,
                               fg_result.line_residual)

    return MattingResult(
        alpha=alpha,
        alpha_global=alpha_global,
        foreground=fg_result.foreground,
        evidence=fg_result.evidence,
        uncertainty_score=unc.score,
        uncertain_mask=unc.mask,
        composite_black=composite(fg_result.foreground, alpha, 0.0),
        composite_white=composite(fg_result.foreground, alpha, 1.0),
        fixed_violation=fixed_violation,
        block_info=block_info,
    )
