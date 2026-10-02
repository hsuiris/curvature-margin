"""DivEye surprisal features (nine statistics) with an optional position mask."""
import numpy as np
from scipy.stats import skew, kurtosis

FEATS9 = ["mean", "var", "skew", "kurt", "d1_mean", "d1_var", "d2_var", "d2_ent", "d2_acf"]


def feats(L, keep):
    x = L[keep]
    d1 = np.diff(L, prepend=np.nan)            # d1[t] = L[t]-L[t-1], computed on the ORIGINAL order
    d2 = np.diff(d1, prepend=np.nan)           # d2[t] = d1[t]-d1[t-1]
    k1 = keep & ~np.isnan(d1); k2 = keep & ~np.isnan(d2)   # difference belongs to the later position
    a, b = d1[k1], d2[k2]
    h, _ = np.histogram(b, bins=20); p = h[h > 0] / h.sum()
    prev = np.roll(k2, 1); prev[0] = False
    pair = k2 & prev   # (d2[t-1], d2[t]) with BOTH positions kept (audit fix 2026-09-20)
    mu = b.mean(); den = ((b - mu) ** 2).sum()
    acf = ((d2[np.roll(pair, -1)] - mu) * (d2[pair] - mu)).sum() / den if den > 0 else 0.0
    return [x.mean(), x.var(), skew(x), kurtosis(x), a.mean(), a.var(), b.var(), -(p * np.log(p)).sum(), acf]
