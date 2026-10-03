"""Paired significance testing and confidence intervals for SettleIN.

Every system is run on the same questions, so comparisons must be paired.
statsmodels' pairwise_tukeyhsd (used in the earlier exploratory scripts)
treats each system's scores as an independent sample. That throws away the
per-question pairing, so the large between-question variance swamps the
system effect and p-values come out far too high.

This module provides:
  - paired_randomization_test: two-sided Fisher randomization (sign-flip)
    test on per-question differences (Smucker, Allan & Carterette, 2007).
  - holm: Holm-Bonferroni adjustment across a family of comparisons.
  - bootstrap_ci: percentile bootstrap CI for a mean.
  - wilson_ci: Wilson score interval for a proportion (refusal rates).
  - roc_auc: threshold-free separability of a score between two groups.
  - compare_configs: all pairwise comparisons for one metric.
"""

from itertools import combinations
from math import sqrt

import numpy as np

N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
SEED = 42


def paired_randomization_test(a, b, n_perm: int = N_PERMUTATIONS, seed: int = SEED) -> float:
    """Two-sided paired randomization test on mean(a - b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.any(d):
        return 1.0
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm_means = np.abs((signs * d).mean(axis=1))
    # +1 smoothing so p is never exactly 0 (Phipson & Smyth, 2010)
    return float((np.sum(perm_means >= observed - 1e-12) + 1) / (n_perm + 1))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def bootstrap_ci(values, alpha: float = 0.05, n_boot: int = N_BOOTSTRAP,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of values."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def roc_auc(positive_scores, negative_scores) -> float:
    """P(score of a random positive > score of a random negative), ties = 0.5."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def compare_configs(per_question: dict[str, dict[str, float]], metric: str,
                    alpha: float = 0.01) -> list[dict]:
    """All pairwise paired comparisons for one metric.

    per_question: {config: {question_id: score}}. Only questions present in
    every config are used. diff = group1 - group2. p_holm is adjusted across
    the pairs in this call (one family per metric).
    """
    configs = list(per_question)
    common = sorted(set.intersection(*(set(v) for v in per_question.values())))
    rows = []
    for g1, g2 in combinations(configs, 2):
        a = [per_question[g1][q] for q in common]
        b = [per_question[g2][q] for q in common]
        diffs = np.asarray(a) - np.asarray(b)
        lo, hi = bootstrap_ci(diffs)
        rows.append({
            "metric": metric, "group1": g1, "group2": g2, "n": len(common),
            "diff": round(float(diffs.mean()), 4),
            "diff_ci95_lo": round(lo, 4), "diff_ci95_hi": round(hi, 4),
            "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()),
            "ties": int((diffs == 0).sum()),
            "p": paired_randomization_test(a, b),
        })
    adjusted = holm([r["p"] for r in rows])
    for r, p_adj in zip(rows, adjusted):
        r["p"] = round(r["p"], 4)
        r["p_holm"] = round(p_adj, 4)
        r["significant"] = p_adj < alpha
    return rows
