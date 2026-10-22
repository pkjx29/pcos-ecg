"""Synthetic class balancing.

Method: SMOTE (Chawla et al. 2002, JAIR 16:321) from imbalanced-learn. It creates new
minority rows by interpolating between a minority row and one of its k nearest minority
neighbours. It sits inside an imblearn Pipeline, which applies samplers during fit()
only, so within cross-validation it only ever sees the training fold. Synthetic rows
are not extra volunteers: participant counts in reports always refer to real people.

A Bayesian-network or copula synthesizer (e.g. SDV) is not used: it is not installed in
the reference environment, and SMOTE is the method this project's brief describes as
"synthetic class balancing". The `make_balancer` interface allows swapping it later.
"""

from __future__ import annotations

from dataclasses import dataclass

from imblearn.over_sampling import SMOTE


@dataclass
class BalancerDecision:
    sampler: SMOTE | None
    skipped_reason: str | None = None


def make_balancer(
    n_minority_train: int, k_neighbors: int, min_minority: int, seed: int
) -> BalancerDecision:
    """SMOTE for one training fold, or None (with a reason) when there is too little data."""
    if n_minority_train < min_minority:
        return BalancerDecision(
            None,
            f"only {n_minority_train} minority participants in training fold "
            f"(< {min_minority}); SMOTE skipped",
        )
    k = min(k_neighbors, n_minority_train - 1)
    return BalancerDecision(SMOTE(k_neighbors=k, random_state=seed))
