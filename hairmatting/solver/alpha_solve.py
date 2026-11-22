"""Alpha estimation with hard trimap constraints.

Global solve (Levin et al.):
    alpha* = argmin  alpha^T L alpha + lambda * || W (alpha - alpha_hat) ||^2
where W is 1 on *manually marked* trimap pixels (definite fg/bg) and 0 on
unknown pixels.  Unknown pixels are therefore free: nothing in the trimap is
ever painted into them by construction.

Block solve: the image is split into tiles with an overlap.  Each tile is an
independent closed-form solve whose border ring is clamped to the currently
known alpha (trimap labels plus already solved overlap values), and tiles are
blended by a distance ramp inside the overlap.  The same global constrained
system is also exposed so callers can compare tiled vs. global alpha along
block seams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .imageio import trimap_masks, TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN
from .laplacian import matting_laplacian


@dataclass
class BlockResult:
    alpha: np.ndarray
    seam_alpha: np.ndarray          # alpha values along block boundary lines
    seam_residual: np.ndarray       # |alpha_tiled - alpha_global| on seams
    fixed_violation: float          # max |alpha - label| on constrained pixels
    tile_edges: tuple                # (vertical_edges, horizontal_edges) in pixels
    alpha_global: np.ndarray


def _constrained_solve(L: sp.csr_matrix, constrained_idx: np.ndarray,
                       constrained_values: np.ndarray, n: int,
                       lam: float, method: str = "auto") -> np.ndarray:
    """Solve (L + lambda W) alpha = lambda W alpha_hat.

    Direct sparse LU is used for small systems; larger systems use Jacobi-
    preconditioned conjugate gradient, which exploits the SPD, banded structure
    of the matting Laplacian and avoids the fill-in blow-up of a full LU.
    """
    W = sp.coo_matrix(
        (np.ones(len(constrained_idx)), (constrained_idx, constrained_idx)),
        shape=(n, n),
    ).tocsr()
    A = (L + lam * W).tocsr()
    b = np.zeros(n, dtype=np.float64)
    b[constrained_idx] = lam * constrained_values

    use_cg = method == "cg" or (method == "auto" and n > 60_000)
    if not use_cg:
        alpha = spla.spsolve(A, b)
    else:
        diag = A.diagonal()
        diag[diag < 1e-12] = 1e-12
        precon = spla.LinearOperator(
            (n, n), matvec=lambda x: x / diag, dtype=np.float64
        )
        alpha, info = spla.cg(A, b, rtol=1e-7, maxiter=4000, M=precon)
        if info != 0:
            # Fall back to a direct solve if CG fails to converge.
            alpha = spla.spsolve(A, b)
    return np.clip(alpha, 0.0, 1.0)


def solve_alpha_global(image: np.ndarray, trimap: np.ndarray,
                       radius: int = 1, eps: float = 1e-7,
                       lam: float = 100.0, L: sp.csr_matrix | None = None,
                       method: str = "auto"
                       ) -> np.ndarray:
    """Closed-form alpha; fixed trimap pixels are hard constraints."""
    h, w = trimap.shape
    if L is None:
        L = matting_laplacian(image, radius=radius, eps=eps)
    bg, fg, _ = trimap_masks(trimap)
    idx = np.flatnonzero((bg | fg).ravel())
    values = np.where(fg.ravel()[idx], 1.0, 0.0)
    alpha = _constrained_solve(L, idx, values, h * w, lam,
                               method=method).reshape(h, w)
    # Constraints are part of the linear system at high weight; snap the labels
    # so fixed points are exact regardless of solver tolerance.
    alpha[bg] = 0.0
    alpha[fg] = 1.0
    return alpha


def _ramp_weights(tile_shape, overlap: int, axis_pos: tuple[int, int],
                  full_shape) -> np.ndarray:
    """Triangular (distance) blend weights: 1 at tile center, ramp in overlap."""
    th, tw = tile_shape
    wy = np.ones(th, dtype=np.float64)
    wx = np.ones(tw, dtype=np.float64)
    top, left = axis_pos
    h, w = full_shape
    if top > 0:
        ramp = np.linspace(0.0, 1.0, overlap + 2)[1:-1]
        wy[:overlap] = ramp
    if top + th < h:
        ramp = np.linspace(1.0, 0.0, overlap + 2)[1:-1]
        wy[-overlap:] = ramp
    if left > 0:
        ramp = np.linspace(0.0, 1.0, overlap + 2)[1:-1]
        wx[:overlap] = ramp
    if left + tw < w:
        ramp = np.linspace(1.0, 0.0, overlap + 2)[1:-1]
        wx[-overlap:] = ramp
    return wy[:, None] * wx[None, :]


def solve_alpha_blocks(image: np.ndarray, trimap: np.ndarray,
                       block: int = 160, overlap: int = 32,
                       radius: int = 1, eps: float = 1e-7,
                       lam: float = 100.0) -> BlockResult:
    """Tile-based solve with overlapping windows and ramp blending."""
    h, w = trimap.shape
    L = matting_laplacian(image, radius=radius, eps=eps)
    alpha_global = solve_alpha_global(image, trimap, radius, eps, lam, L=L,
                                      method="auto")

    # Tile origins: interior blocks start `overlap` px inside the previous tile,
    # and the last tile is shifted so it always ends exactly on the image edge.
    def _origins(length: int) -> list[int]:
        if length <= block:
            return [0]
        origins = list(range(0, length - overlap, block - overlap))
        if origins[-1] + block < length:
            origins.append(length - block)
        return sorted(set(origins))

    tops = _origins(h)
    lefts = _origins(w)

    acc = np.zeros((h, w), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    fixed_mask = np.zeros((h, w), dtype=bool)
    bg, fg, _ = trimap_masks(trimap)
    fixed_mask[bg | fg] = True

    for top in tops:
        for left in lefts:
            bot = min(top + block, h)
            right = min(left + block, w)
            t_trimap = trimap[top:bot, left:right].copy()
            sub_h, sub_w = t_trimap.shape

            # Outer overlap ring that belongs to another tile's core is clamped
            # to the global solution so adjacent tiles share boundary values.
            if top > 0:
                t_trimap[:overlap, :] = np.where(
                    t_trimap[:overlap, :] == TRIMAP_UNKNOWN,
                    np.where(alpha_global[top:top + min(overlap, sub_h), left:right] >= 0.5,
                             TRIMAP_FG, TRIMAP_BG),
                    t_trimap[:overlap, :],
                )
            if bot < h:
                t_trimap[-overlap:, :] = np.where(
                    t_trimap[-overlap:, :] == TRIMAP_UNKNOWN,
                    np.where(alpha_global[bot - min(overlap, sub_h):bot, left:right] >= 0.5,
                             TRIMAP_FG, TRIMAP_BG),
                    t_trimap[-overlap:, :],
                )
            if left > 0:
                t_trimap[:, :overlap] = np.where(
                    t_trimap[:, :overlap] == TRIMAP_UNKNOWN,
                    np.where(alpha_global[top:bot, left:left + min(overlap, sub_w)] >= 0.5,
                             TRIMAP_FG, TRIMAP_BG),
                    t_trimap[:, :overlap],
                )
            if right < w:
                t_trimap[:, -overlap:] = np.where(
                    t_trimap[:, -overlap:] == TRIMAP_UNKNOWN,
                    np.where(alpha_global[top:bot, right - min(overlap, sub_w):right] >= 0.5,
                             TRIMAP_FG, TRIMAP_BG),
                    t_trimap[:, -overlap:],
                )

            sub_img = image[top:bot, left:right]
            sub_L = matting_laplacian(sub_img, radius=radius, eps=eps)
            # Tiles are small enough for a direct LU, which gives the most
            # accurate boundary values for seam blending.
            t_alpha = solve_alpha_global(sub_img, t_trimap, radius, eps, lam,
                                         L=sub_L, method="direct")
            weights = _ramp_weights((sub_h, sub_w), overlap, (top, left), (h, w))
            acc[top:bot, left:right] += t_alpha * weights
            wsum[top:bot, left:right] += weights

    alpha_tiled = acc / np.maximum(wsum, 1e-12)
    alpha_tiled[bg] = 0.0
    alpha_tiled[fg] = 1.0

    # Seam diagnostics on interior tile boundary lines: one tile's core ends
    # where the next overlapping tile's ramps begin.
    v_edges = [tops[k] + block for k in range(len(tops) - 1)]
    h_edges = [lefts[k] + block for k in range(len(lefts) - 1)]
    seam_mask = np.zeros((h, w), dtype=bool)
    for e in v_edges:
        e = min(e, h - 1)
        seam_mask[e, :] = True
    for e in h_edges:
        e = min(e, w - 1)
        seam_mask[:, e] = True
    seam_residual = np.abs(alpha_tiled - alpha_global)
    fixed_violation = float(
        max(np.abs(alpha_tiled[bg] - 0.0).max(initial=0.0),
            np.abs(alpha_tiled[fg] - 1.0).max(initial=0.0))
    )

    return BlockResult(
        alpha=alpha_tiled,
        seam_alpha=alpha_tiled[seam_mask],
        seam_residual=seam_residual[seam_mask],
        fixed_violation=fixed_violation,
        tile_edges=(v_edges, h_edges),
        alpha_global=alpha_global,
    )
