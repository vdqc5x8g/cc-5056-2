"""Closed-form matting Laplacian (Levin, Lischinski & Weiss, CVPR 2007).

For every 3x3 window w_k the colour-line assumption gives a local affinity:

    L_ij = sum_{k | (i,j) in w_k}
           [ delta_ij
             - 1/|w_k| * (1 + (I_i - mu_k)^T (C_k + eps/|w_k| I_3)^-1
                                (I_j - mu_k)) ]

The matrix is symmetric, positive semi-definite and has the constant vector in
its null space (L 1 = 0).  Only interior windows (radius r) are formed, so
border pixels belong to fewer windows; no image content is extrapolated.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp


def _window_stats(image: np.ndarray, radius: int, eps: float):
    """Per-window mean colour, inverse regularised covariance and member indices."""
    h, w, _ = image.shape
    size = 2 * radius + 1
    n_win = size * size

    # Box means over centered windows; only interior centers are queried later.
    means = np.empty((h, w, 3), dtype=np.float64)
    for c in range(3):
        # Zero-padded mode: only interior centers are queried below, where the
        # padding never enters the window, so stats are exact.
        means[..., c] = ndi.uniform_filter(image[..., c], size=size, mode="constant")

    cov = np.empty((h, w, 3, 3), dtype=np.float64)
    for c in range(3):
        for d in range(c, 3):
            prod_mean = ndi.uniform_filter(
                image[..., c] * image[..., d], size=size, mode="constant"
            )
            cov[..., c, d] = prod_mean - means[..., c] * means[..., d]
            cov[..., d, c] = cov[..., c, d]

    # Restrict to valid (fully inside the image) window centers.
    cy, cx = np.mgrid[radius:h - radius, radius:w - radius]
    cy = cy.ravel()
    cx = cx.ravel()
    mu = means[cy, cx]                                   # (K, 3)
    sigma = cov[cy, cx] + (eps / n_win) * np.eye(3)     # (K, 3, 3)
    inv_sigma = np.linalg.inv(sigma)                    # (K, 3, 3)

    # Global pixel indices of the 9 members of each window.
    offsets = np.array([(dy, dx) for dy in range(-radius, radius + 1)
                        for dx in range(-radius, radius + 1)])
    member_rows = cy[:, None] + offsets[None, :, 0]     # (K, n_win)
    member_cols = cx[:, None] + offsets[None, :, 1]
    member_idx = member_rows * w + member_cols          # (K, n_win)

    # Colour of every window member, and per-window centered colours.
    flat_img = image.reshape(-1, 3)
    member_colors = flat_img[member_idx]                # (K, n_win, 3)
    centered = member_colors - mu[:, None, :]           # (K, n_win, 3)
    return member_idx, centered, inv_sigma, n_win


def matting_laplacian(image: np.ndarray, radius: int = 1,
                      eps: float = 1e-7) -> sp.csr_matrix:
    """Build the (N x N) sparse matting Laplacian for an RGB image in [0, 1]."""
    h, w, d = image.shape
    if d != 3:
        raise ValueError("matting_laplacian expects an RGB image")
    n = h * w
    member_idx, centered, inv_sigma, n_win = _window_stats(image, radius, eps)
    n_windows = member_idx.shape[0]

    # Precompute inv_sigma @ (I_j - mu) for every window member j.
    # rhs[k, j, c] = sum_d inv_sigma[k, c, d] * centered[k, j, d]
    rhs = np.einsum("kcd,kjd->kjc", inv_sigma, centered)

    # Vectorised accumulation over all ordered member pairs (a, b) of every
    # window.  Each contributes -(1/m)(1 + c_ia^T Sigma^-1 c_jb); diagonal
    # pairs additionally carry delta_ii = +1.  Entries landing on the same
    # (row, col) are summed when the COO matrix is converted to CSR.
    total = n_windows * n_win * n_win
    rows_all = np.empty(total, dtype=np.int64)
    cols_all = np.empty(total, dtype=np.int64)
    vals_all = np.empty(total, dtype=np.float64)
    flat_cols = member_idx.reshape(-1)
    for a in range(n_win):
        c_i = centered[:, a, :]                          # (K, 3)
        # dot[k, b] = c_i[k] . rhs[k, b] for every member b
        dots = np.einsum("kc,kbc->kb", c_i, rhs)         # (K, n_win)
        t = -(1.0 / n_win) * (1.0 + dots)                # (K, n_win)
        t[:, a] += 1.0                                   # delta_ii
        sl = slice(a * n_windows * n_win,
                   (a + 1) * n_windows * n_win)
        rows_all[sl] = np.repeat(member_idx[:, a], n_win)
        cols_all[sl] = flat_cols
        vals_all[sl] = t.reshape(-1)

    L = sp.coo_matrix(
        (vals_all, (rows_all, cols_all)),
        shape=(n, n),
    ).tocsr()
    L = (L + L.T) * 0.5              # kill asymmetric floating-point noise
    return L.tocsr()
