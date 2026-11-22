"""Uncertainty reporting.

A pixel is reported as uncertain when the chosen algorithm's local assumptions
do not support a numerical answer.  Three independent failure modes are
distinguished:

* no evidence:      no window around the pixel contains an alpha gradient,
                    so only the smoothness prior acts;
* line violation:   observed colours do not lie on a local colour line,
                    violating the matting model itself;
* low-alpha unstable: alpha is close to 0, where foreground colour is
                    numerically unobservable (I ~= B regardless of F) and
                    alpha inversion amplifies sensor/compression noise.

Uncertain pixels are visualised as such; the solver never invents hair strands
or a distance-field gradient to fill them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from .imageio import TRIMAP_BG, TRIMAP_FG


@dataclass
class UncertaintyMap:
    score: np.ndarray          # 0 reliable .. 1 unusable
    no_evidence: np.ndarray
    line_violation: np.ndarray
    low_alpha_unstable: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return self.score >= 0.5


# Below this alpha the foreground colour contribution alpha*F is below one
# 8-bit code level even for F=1: F is simply not observable.
LOW_ALPHA_LIMIT = 1.0 / 255.0
# Alpha values below this band have inversion error amplified beyond 10%.
LOW_ALPHA_BAND = 0.05


def estimate_uncertainty(alpha: np.ndarray, trimap: np.ndarray,
                         evidence: np.ndarray, line_residual: np.ndarray,
                         noise_floor: float = 2.0 / 255.0) -> UncertaintyMap:
    fixed = (trimap == TRIMAP_FG) | (trimap == TRIMAP_BG)

    # A pixel is "no evidence" only when the local model offers nothing *and*
    # the solution is non-trivial there: it solves to non-negligible alpha or
    # sits immediately next to a hair pixel.  Unknown pixels that stably solve
    # to (near) zero against the backdrop are confident background, not gaps to
    # be flagged or filled.  The 0.02 threshold matches the foreground
    # estimator's confident-background anchor.
    hair = alpha > 0.02
    near_hair = hair | ndi.binary_dilation(hair, iterations=1)
    no_evidence = (evidence < 0.15) & (~fixed) & near_hair
    # A colour-line residual above the sensor/compression noise floor means the
    # two-line assumption is locally violated.
    line_violation = (line_residual > 3.0 * noise_floor) & (~fixed)

    # Error propagation for alpha: with additive noise sigma on I, the inversion
    # alpha = (I-B)/(F-B) has error ~ sigma / |F-B|, which diverges as alpha->0
    # when F and B are close.  Only *mixed* pixels qualify: a pixel that solves
    # to (near) zero is a confident background observation and stays stable, and
    # a definite-bg pixel that correctly solves to 0 is never uncertain either.
    low_alpha_unstable = (
        (alpha > 0.02)
        & (alpha < LOW_ALPHA_BAND)
        & (trimap != TRIMAP_BG)
        & (evidence > 0.0)
    )

    score = np.zeros_like(alpha)
    score += 0.5 * no_evidence.astype(np.float64)
    score += 0.4 * line_violation.astype(np.float64)
    score += 0.6 * low_alpha_unstable.astype(np.float64)
    score[fixed] = 0.0
    score = np.clip(score, 0.0, 1.0)
    return UncertaintyMap(
        score=score,
        no_evidence=no_evidence,
        line_violation=line_violation,
        low_alpha_unstable=low_alpha_unstable,
    )
