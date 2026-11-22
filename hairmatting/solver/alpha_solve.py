"""Alpha estimation with hard trimap constraints.

Global solve (Levin et al., original hard-constraint form): fixed trimap
pixels keep their labels exactly and the unknown band minimises the matting
Laplacian energy subject to them:

    alpha_f* = argmin  alpha^T L alpha   s.t.  alpha_c = labels
             = solve( L_ff alpha_f = -L_fc alpha_c )

assembled by elimination, so confirmed foreground/background pixels are never
moved by the unknown-region solve (no soft lambda that trades fixed-point
error for a smaller residual, no post-hoc snap that would hide such drift).

Block solve: the image is split into tiles with an overlap.  Each tile is an
independent hard-constraint solve whose overlap ring is clamped to the
*fractional* alpha of the global reference solve (never a binarized label,
which would bend fine strands at tile seams), and tiles are blended by a
distance ramp inside the overlap.  The same global constrained system is also
exposed so callers can compare tiled vs. global alpha along block seams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .imageio import trimap_masks
from .laplacian import matting_laplacian


@dataclass
class BlockResult:
    alpha: np.ndarray
    seam_alpha: np.ndarray          # alpha values along block boundary lines
    seam_residual: np.ndarray       # |alpha_tiled - alpha_global| on seams
    fixed_violation: float          # max |alpha - label| on constrained pixels
    tile_edges: tuple                # (vertical_edges, horizontal_edges) in pixels
    alpha_global: np.ndarray


def _hard_solve(L: sp.csr_matrix, fixed_idx: np.ndarray,
                fixed_values: np.ndarray, n: int,
                method: str = "auto") -> np.ndarray:
    """Solve L alpha = 0 with alpha[fixed_idx] == fixed_values exactly.

    Elimination form of Levin's constrained system: the free unknowns satisfy
    L_ff a_f = -L_fc a_c.  L_ff is a principal submatrix of the SPD matting
    Laplacian and is positive definite as long as every connected component
    of the pixel graph touches at least one constraint.  Direct sparse LU is
    used for small systems; larger ones use Jacobi-preconditioned CG.
    """
    if len(fixed_idx) == 0:
        raise ValueError(
            "no constraints: a trimap without definite labels is ill-posed")
    fixed_mask = np.zeros(n, dtype=bool)
    fixed_mask[fixed_idx] = True
    free_idx = np.flatnonzero(~fixed_mask)

    alpha = np.empty(n, dtype=np.float64)
    alpha[fixed_idx] = fixed_values
    if len(free_idx) == 0:
        return alpha

    Lff = L[free_idx][:, free_idx].tocsr()
    rhs = -np.asarray(L[free_idx][:, fixed_idx] @ fixed_values).ravel()

    use_cg = method == "cg" or (method == "auto" and len(free_idx) > 60_000)
    x = None
    if not use_cg:
        x = spla.spsolve(Lff, rhs)
        if not np.all(np.isfinite(x)):
            use_cg = True
    if use_cg:
        diag = Lff.diagonal()
        diag[diag < 1e-12] = 1e-12
        precon = spla.LinearOperator(
            (len(free_idx), len(free_idx)),
            matvec=lambda v: v / diag, dtype=np.float64,
        )
        x, info = spla.cg(Lff, rhs, rtol=1e-8, maxiter=5000, M=precon)
        if info != 0:
            x = spla.spsolve(Lff, rhs)
        if not np.all(np.isfinite(x)):
            raise RuntimeError("alpha solve failed to produce finite values")
    alpha[free_idx] = x
    return alpha


def solve_alpha_global(image: np.ndarray, trimap: np.ndarray,
                       radius: int = 1, eps: float = 1e-7,
                       L: sp.csr_matrix | None = None,
                       method: str = "auto"
                       ) -> np.ndarray:
    """Closed-form alpha; fixed trimap pixels are exact hard constraints."""
    h, w = trimap.shape
    if L is None:
        L = matting_laplacian(image, radius=radius, eps=eps)
    bg, fg, _ = trimap_masks(trimap)
    idx = np.flatnonzero((bg | fg).ravel())
    values = np.where(fg.ravel()[idx], 1.0, 0.0)
    alpha = _hard_solve(L, idx, values, h * w, method=method).reshape(h, w)
    return np.clip(alpha, 0.0, 1.0)


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
                       radius: int = 1, eps: float = 1e-7
                       ) -> BlockResult:
    """Tile-based solve with overlapping windows and ramp blending.

    The overlap ring of every tile is constrained to the *fractional* alpha
    of the global reference solve, so adjacent tiles share boundary values
    without quantising fine strands to 0/1 at the seam.
    """
    h, w = trimap.shape
    L = matting_laplacian(image, radius=radius, eps=eps)
    alpha_global = solve_alpha_global(image, trimap, radius, eps, L=L,
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
    bg, fg, _ = trimap_masks(trimap)

    for top in tops:
        for left in lefts:
            bot = min(top + block, h)
            right = min(left + block, w)
            t_trimap = trimap[top:bot, left:right]
            sub_h, sub_w = t_trimap.shape
            t_bg, t_fg, t_unk = trimap_masks(t_trimap)

            # Overlap ring shared with a neighbouring tile: unknown pixels are
            # clamped to the fractional global alpha so tiles agree on strand
            # crossings instead of snapping them to 0 or 1.
            ring = np.zeros((sub_h, sub_w), dtype=bool)
            if top > 0:
                ring[:overlap, :] = True
            if bot < h:
                ring[-overlap:, :] = True
            if left > 0:
                ring[:, :overlap] = True
            if right < w:
                ring[:, -overlap:] = True
            ring &= t_unk

            label_idx = np.flatnonzero((t_bg | t_fg).ravel())
            label_val = np.where(t_fg.ravel()[label_idx], 1.0, 0.0)
            ring_idx = np.flatnonzero(ring.ravel())
            ring_val = alpha_global[top:bot, left:right].ravel()[ring_idx]
            con_idx = np.concatenate([label_idx, ring_idx])
            con_val = np.concatenate([label_val, ring_val])

            sub_img = image[top:bot, left:right]
            sub_L = matting_laplacian(sub_img, radius=radius, eps=eps)
            # Tiles are small enough for a direct LU, which gives the most
            # accurate boundary values for seam blending.
            t_alpha = _hard_solve(sub_L, con_idx, con_val, sub_h * sub_w,
                                  method="direct").reshape(sub_h, sub_w)
            t_alpha = np.clip(t_alpha, 0.0, 1.0)
            weights = _ramp_weights((sub_h, sub_w), overlap, (top, left),
                                    (h, w))
            acc[top:bot, left:right] += t_alpha * weights
            wsum[top:bot, left:right] += weights

    alpha_tiled = acc / np.maximum(wsum, 1e-12)
    # Every tile hard-constrains a labelled pixel to the same label, so the
    # blend can only differ from it by floating-point rounding; snap back to
    # the exact label rather than reporting 1-ulp noise as a violation.
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
