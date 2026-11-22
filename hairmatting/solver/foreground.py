"""Edge foreground-colour estimation from real colour observations.

Observation model per pixel:  I_i = alpha_i F_i + (1 - alpha_i) B_i.
Within a small window the colour-line model says F and B each vary slowly, so
for every window containing the pixel we fit the two line endpoints from
local statistics, obtaining a local foreground guess f_ik *without ever
dividing by alpha*.  Local guesses are fused in one weighted least-squares
system per channel:

    sum_k w_ik (f_i - f_ik)^2  +  smoothness term  -> min

The weight w_ik is the *local colour evidence* (line fit quality times alpha
spread).  Pixels whose only support is the smoothness prior get a low
``evidence`` value and are reported as uncertain instead of being silently
painted with invented hair colour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ForegroundResult:
    foreground: np.ndarray      # (H, W, 3) estimated foreground colour [0, 1]
    evidence: np.ndarray        # (H, W) local evidence weight in [0, 1]
    line_residual: np.ndarray   # (H, W) colour-line fit residual
    uncertain: np.ndarray       # (H, W) bool: no reliable local evidence


def _local_line_endpoints(image: np.ndarray, radius: int, eps: float):
    """For every window: line direction d, and endpoints of the observed line.

    Returns per-window-center means mu, unit direction d, and the projections
    t_min/t_max of member pixels.  Endpoints: mu + t * d.
    """
    h, w, _ = image.shape
    size = 2 * radius + 1
    n_win = size * size
    means = np.empty((h, w, 3))
    for c in range(3):
        means[..., c] = ndi.uniform_filter(image[..., c], size=size,
                                           mode="constant")
    cov = np.empty((h, w, 3, 3))
    for c in range(3):
        for d in range(c, 3):
            pm = ndi.uniform_filter(image[..., c] * image[..., d], size=size,
                                    mode="constant")
            cov[..., c, d] = pm - means[..., c] * means[..., d]
            cov[..., d, c] = cov[..., c, d]
    cov += eps * np.eye(3)

    cy, cx = np.mgrid[radius:h - radius, radius:w - radius]
    cy, cx = cy.ravel(), cx.ravel()
    sigma = cov[cy, cx]
    mu = means[cy, cx]
    eigvals, eigvecs = np.linalg.eigh(sigma)       # ascending order
    direction = eigvecs[:, :, -1]                  # largest-eigenvalue vector

    offsets = [(dy, dx) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)]
    flat = image.reshape(-1, 3)
    member_idx = np.stack([
        (cy + dy) * w + (cx + dx) for dy, dx in offsets
    ], axis=1)
    members = flat[member_idx]                              # (K, m, 3)
    proj = np.einsum("kmc,kc->km", members - mu[:, None, :],
                     direction)                             # (K, m)
    # Reconstructed line colour residual: distance of members to the fitted line.
    recon = mu[:, None, :] + proj[..., None] * direction[:, None, :]
    residual = np.sqrt(((members - recon) ** 2).sum(axis=2)).mean(axis=1)
    return cy, cx, mu, direction, proj, residual, member_idx


def estimate_foreground(image: np.ndarray, alpha: np.ndarray,
                        trimap: np.ndarray, radius: int = 1,
                        eps: float = 1e-6,
                        smoothness: float = 0.02) -> ForegroundResult:
    h, w = trimap.shape
    n = h * w
    cy, cx, mu, direction, proj, residual, member_idx = _local_line_endpoints(
        image, radius, eps
    )
    flat_img = image.reshape(-1, 3)
    flat_alpha = alpha.ravel()

    # Accumulate per-pixel local foreground hypotheses.  Inside each window the
    # endpoint on the high-alpha side is taken as foreground, the other as
    # background (alpha ordering is measured directly - no alpha division).
    K = cy.shape[0]
    member_alpha = flat_alpha[member_idx]                  # (K, m)
    # Quantile spread per window without a Python-level percentile call.
    sorted_a = np.sort(member_alpha, axis=1)
    q10 = sorted_a[:, max(int(0.10 * sorted_a.shape[1]), 0)]
    q90 = sorted_a[:, min(int(0.90 * sorted_a.shape[1]),
                          sorted_a.shape[1] - 1)]
    spread = q90 - q10
    t_lo = proj.min(axis=1)                                # (K,)
    t_hi = proj.max(axis=1)
    span = np.maximum(t_hi - t_lo, 1e-12)
    near_hi_mask = proj >= (t_lo[:, None] + 0.75 * span[:, None])
    near_lo_mask = proj <= (t_lo[:, None] + 0.25 * span[:, None])
    a_near_hi = np.where(
        near_hi_mask.any(axis=1),
        (member_alpha * near_hi_mask).sum(axis=1)
        / np.maximum(near_hi_mask.sum(axis=1), 1),
        0.5,
    )
    a_near_lo = np.where(
        near_lo_mask.any(axis=1),
        (member_alpha * near_lo_mask).sum(axis=1)
        / np.maximum(near_lo_mask.sum(axis=1), 1),
        0.5,
    )
    hi_is_fg = (a_near_hi >= a_near_lo)
    f_t = np.where(hi_is_fg, t_hi, t_lo)
    f_end = mu + f_t[:, None] * direction                  # (K, 3)

    line_fit_quality = np.exp(-residual / 0.05)
    w_k = np.clip(spread / 0.5, 0.0, 1.0) * line_fit_quality  # (K,)

    # Scatter-accumulate every window's contribution to its member pixels.
    acc_f = np.zeros((n, 3), dtype=np.float64)
    acc_w = np.zeros(n, dtype=np.float64)
    line_res = np.zeros(n, dtype=np.float64)
    line_res_w = np.zeros(n, dtype=np.float64)
    evidence_raw = np.zeros(n, dtype=np.float64)
    flat_members = member_idx.reshape(-1)
    np.add.at(acc_f, flat_members,
              np.repeat(w_k, member_idx.shape[1])[:, None]
              * np.repeat(f_end, member_idx.shape[1], axis=0))
    np.add.at(acc_w, flat_members, np.repeat(w_k, member_idx.shape[1]))
    np.add.at(line_res, flat_members,
              np.repeat(w_k * residual, member_idx.shape[1]))
    np.add.at(line_res_w, flat_members,
              np.repeat(w_k, member_idx.shape[1]))
    np.maximum.at(evidence_raw, flat_members,
                  np.repeat(w_k, member_idx.shape[1]))

    has_local = acc_w > 1e-8
    f_local = np.zeros((n, 3))
    f_local[has_local] = acc_f[has_local] / acc_w[has_local, None]
    line_res_avg = np.zeros(n)
    line_res_avg[has_local] = line_res[has_local] / line_res_w[has_local]

    # Definite foreground pixels carry direct colour evidence with weight 1.
    definite_fg = (trimap == 255).ravel()
    definite_bg = (trimap == 0).ravel()
    evidence = np.clip(evidence_raw, 0.0, 1.0)
    evidence[definite_fg] = 1.0

    # Pixels that stably solve to (near) zero alpha are background: the
    # foreground colour is irrelevant there, so anchor them to the observed
    # colour with full weight.  This keeps the system non-singular and prevents
    # black wedges without inventing hair colour; genuinely unsupported mixed
    # pixels remain weight-free and are reported as uncertain.
    confident_bg = (flat_alpha < 0.02) & (~definite_fg)
    evidence[confident_bg] = 1.0

    # Weighted least squares per channel: local evidence + neighbour smoothness.
    # The smoothness edge (i, j) only carries weight where *both* endpoints have
    # local evidence, so colour cannot bleed along unsupported background pixels.
    target = f_local.copy()
    target[definite_fg] = flat_img[definite_fg]
    target[confident_bg] = flat_img[confident_bg]
    Wd = sp.diags(evidence)

    # 4-neighbour graph with evidence-gated edge weights.
    idx = np.arange(n).reshape(h, w)
    pair_a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    pair_b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    edge_w = np.minimum(evidence[pair_a], evidence[pair_b])
    rows = np.concatenate([pair_a, pair_b])
    cols = np.concatenate([pair_b, pair_a])
    data = np.concatenate([edge_w, edge_w])
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    deg = sp.diags(np.asarray(adj.sum(axis=1)).ravel())
    L_smooth = deg - adj

    # Pixels without any local evidence: target 0 with weight 0, smoothness only
    # through edges that themselves vanish without evidence.  A tiny diagonal
    # keeps the system non-singular; unsupported pixels then solve to 0 and are
    # surfaced through the uncertainty map instead of receiving invented colour.
    A = (Wd + smoothness * L_smooth + 1e-6 * sp.eye(n)).tocsr()

    foreground = np.empty((n, 3))
    for c in range(3):
        b = evidence * target[:, c]
        foreground[:, c] = spla.spsolve(A, b)
    foreground = np.clip(foreground, 0.0, 1.0).reshape(h, w, 3)

    # Definite background keeps observed colour (it will be replaced on swap).
    foreground[definite_bg.reshape(h, w)] = image[definite_bg.reshape(h, w)]

    # Unsupported mixed pixels are uncertain; confident background and definite
    # labels are not.
    uncertain = (~has_local) & (~definite_fg) & (~confident_bg)
    return ForegroundResult(
        foreground=foreground,
        evidence=evidence.reshape(h, w),
        line_residual=line_res_avg.reshape(h, w),
        uncertain=uncertain.reshape(h, w),
    )
