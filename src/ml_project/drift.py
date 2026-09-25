"""Drift monitoring: data drift (feature distributions) and prediction drift.

Data drift: has the distribution of INPUT features changed between a
reference period (training data) and a production window (recent
requests)? Checked here via:
  - categorical features (Contract, PaymentMethod): Population Stability
    Index (PSI)
  - numeric features (tenure, MonthlyCharges): two-sample
    Kolmogorov-Smirnov test (scipy)

Prediction drift: has the distribution of MODEL OUTPUTS changed? Checked
via PSI on the predicted-class distribution.

Prediction drift does NOT automatically mean the model is failing — it
means the population being scored looks different than before. It's a
signal to investigate (alongside data drift, and, once ground truth is
available, actual performance) — not a verdict on its own. Concept drift
(the relationship between features and the true label changing) is a
distinct, harder problem this project does not attempt to detect
directly — we only have proxies for it (data drift + prediction drift).

Method limitations:
  - PSI needs a reasonably sized sample per group to be stable; small
    production batches will look noisier than they are, and PSI is only
    meaningful for a fixed, small set of categories (true here).
  - KS assumes i.i.d. continuous samples and only detects *that*
    distributions differ, not *how* — a mean shift and a spread-only
    change can produce similar statistics without inspecting means/stds
    separately (which is why summaries include both).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

CATEGORICAL_DRIFT_FEATURES = ["Contract", "PaymentMethod"]
NUMERIC_DRIFT_FEATURES = ["tenure", "MonthlyCharges"]

# Standard PSI convention: <0.1 no significant change; 0.1-0.25 moderate
# shift, worth investigating; >=0.25 major shift. We flag `drifted=True`
# at the moderate threshold (0.1) — that's the "go look at this" signal —
# and separately record `severity` so callers can distinguish "worth a
# look" from "clearly a problem."
PSI_MODERATE_THRESHOLD = 0.1
PSI_MAJOR_THRESHOLD = 0.25
KS_ALPHA = 0.05


def _psi_severity(psi: float) -> str:
    if psi >= PSI_MAJOR_THRESHOLD:
        return "major"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


@dataclass
class DriftResult:
    feature: str
    method: str
    statistic: float
    drifted: bool
    reference_summary: dict
    production_summary: dict
    severity: str = "none"

    def summary(self) -> str:
        if not self.drifted:
            return f"✅ no significant drift — {self.feature} ({self.method}={self.statistic:.4f})"
        severity_tag = f" ({self.severity})" if self.severity != "none" else ""
        return f"⚠️ drift detected{severity_tag} — {self.feature} ({self.method}={self.statistic:.4f})"


def population_stability_index(
    reference_pct: pd.Series, production_pct: pd.Series, eps: float = 1e-4
) -> float:
    """PSI between two proportion distributions over the same categories.

    PSI = sum over categories of (prod% - ref%) * ln(prod% / ref%).
    `eps` avoids log(0)/division-by-zero for categories present in one
    distribution but not the other.
    """
    categories = sorted(set(reference_pct.index) | set(production_pct.index))
    psi = 0.0
    for cat in categories:
        ref = reference_pct.get(cat, 0.0) + eps
        prod = production_pct.get(cat, 0.0) + eps
        psi += (prod - ref) * np.log(prod / ref)
    return psi


def check_categorical_drift(reference: pd.Series, production: pd.Series, feature_name: str) -> DriftResult:
    ref_pct = reference.value_counts(normalize=True)
    prod_pct = production.value_counts(normalize=True)
    psi = population_stability_index(ref_pct, prod_pct)
    return DriftResult(
        feature=feature_name,
        method="PSI",
        statistic=round(psi, 4),
        drifted=psi >= PSI_MODERATE_THRESHOLD,
        severity=_psi_severity(psi),
        reference_summary=ref_pct.round(4).to_dict(),
        production_summary=prod_pct.round(4).to_dict(),
    )


def check_numeric_drift(reference: pd.Series, production: pd.Series, feature_name: str) -> DriftResult:
    statistic, p_value = stats.ks_2samp(reference.dropna(), production.dropna())
    return DriftResult(
        feature=feature_name,
        method="KS",
        statistic=round(float(statistic), 4),
        drifted=p_value < KS_ALPHA,
        reference_summary={"mean": round(float(reference.mean()), 2), "std": round(float(reference.std()), 2)},
        production_summary={"mean": round(float(production.mean()), 2), "std": round(float(production.std()), 2)},
    )


def check_prediction_drift(reference_predictions: pd.Series, production_predictions: pd.Series) -> DriftResult:
    """PSI on the predicted-class distribution.

    Remember: this measures whether the model's OUTPUTS shifted, which is
    a different question from whether the INPUTS shifted (check_*_drift
    above) or whether the model got worse (which needs ground truth).
    """
    ref_pct = pd.Series(reference_predictions).value_counts(normalize=True)
    prod_pct = pd.Series(production_predictions).value_counts(normalize=True)
    psi = population_stability_index(ref_pct, prod_pct)
    return DriftResult(
        feature="prediction",
        method="PSI",
        statistic=round(psi, 4),
        drifted=psi >= PSI_MODERATE_THRESHOLD,
        severity=_psi_severity(psi),
        reference_summary=ref_pct.round(4).to_dict(),
        production_summary=prod_pct.round(4).to_dict(),
    )


def run_data_drift_report(reference_df: pd.DataFrame, production_df: pd.DataFrame) -> list[DriftResult]:
    """Run all configured data-drift checks (categorical + numeric)."""
    results = []
    for col in CATEGORICAL_DRIFT_FEATURES:
        results.append(check_categorical_drift(reference_df[col], production_df[col], col))
    for col in NUMERIC_DRIFT_FEATURES:
        results.append(check_numeric_drift(reference_df[col], production_df[col], col))
    return results
