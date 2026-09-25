"""Data quality checks for incoming inference requests.

Runs before the model sees the data. Distinguishes a *malformed* request
(missing/out-of-range/wrong-type) from a request that's merely unusual but
valid (an unseen category — handled gracefully by the pipeline, not an
error). This distinction matters again in the drift-monitoring iteration:
drift is about the *shape of valid data* changing, not about individual
bad requests slipping through.

Known categories are the values the model was actually trained on — an
unexpected category doesn't fail the check, since the pipeline's
OneHotEncoder(handle_unknown="ignore") handles it, but it's still worth
flagging as a warning (rather than a failure) since it's a sign of
something the model has never seen an example of.
"""

from dataclasses import dataclass, field

from ml_project.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Sanity-check ranges for numeric features. Not the full training-data
# range — deliberately a bit wider, since valid new customers can have
# values at the edges of what training happened to contain. The point is
# catching impossible values (negative tenure, a $50,000 monthly bill),
# not being a strict re-implementation of the training distribution.
NUMERIC_RANGES = {
    "tenure": (0, 100),  # months; >100 months (~8yr) is implausible for this dataset
    "MonthlyCharges": (0, 500),  # dollars
}

# Categories actually seen during training, for the columns we use. Used
# only to WARN on an unseen value, never to reject it — the model can
# still produce a prediction via handle_unknown="ignore".
KNOWN_CATEGORIES = {
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


@dataclass
class QualityCheckResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        lines.append("✅ schema valid" if not self.errors else "❌ schema invalid")
        for err in self.errors:
            lines.append(f"❌ {err}")
        for warn in self.warnings:
            lines.append(f"⚠️  {warn}")
        return "\n".join(lines)


def check_request(features: dict) -> QualityCheckResult:
    """Validate a single incoming prediction request.

    `errors` means the request is malformed and should be rejected before
    reaching the model. `warnings` means the request is valid but unusual
    (worth logging, not worth rejecting).
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [f for f in REQUIRED_FEATURES if f not in features]
    if missing:
        errors.append(f"missing required feature(s): {missing}")

    unexpected = [k for k in features if k not in REQUIRED_FEATURES]
    if unexpected:
        warnings.append(f"unexpected column(s) present (will be ignored): {unexpected}")

    for col, (low, high) in NUMERIC_RANGES.items():
        if col not in features:
            continue
        value = features[col]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"'{col}' is not numeric: {value!r}")
            continue
        if not (low <= value <= high):
            errors.append(f"'{col}' out of expected range [{low}, {high}]: {value}")

    for col, known_values in KNOWN_CATEGORIES.items():
        if col not in features:
            continue
        if features[col] not in known_values:
            warnings.append(f"unexpected category detected for '{col}': {features[col]!r}")

    return QualityCheckResult(valid=(len(errors) == 0), errors=errors, warnings=warnings)
