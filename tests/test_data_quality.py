from ml_project.data_quality import check_request

VALID = {
    "tenure": 12,
    "MonthlyCharges": 55.0,
    "Contract": "One year",
    "PaymentMethod": "Mailed check",
}


def test_valid_request_passes_with_no_errors_or_warnings():
    result = check_request(VALID)
    assert result.valid
    assert result.errors == []
    assert result.warnings == []


def test_missing_feature_is_an_error():
    incomplete = {k: v for k, v in VALID.items() if k != "Contract"}
    result = check_request(incomplete)
    assert not result.valid
    assert any("missing required feature" in e for e in result.errors)


def test_unexpected_column_is_a_warning_not_an_error():
    result = check_request({**VALID, "some_extra_field": 123})
    assert result.valid
    assert any("unexpected column" in w for w in result.warnings)


def test_negative_tenure_is_an_error():
    result = check_request({**VALID, "tenure": -5})
    assert not result.valid
    assert any("out of expected range" in e for e in result.errors)


def test_extreme_monthly_charges_is_an_error():
    result = check_request({**VALID, "MonthlyCharges": 50000})
    assert not result.valid
    assert any("out of expected range" in e for e in result.errors)


def test_non_numeric_tenure_is_an_error():
    result = check_request({**VALID, "tenure": "twelve"})
    assert not result.valid
    assert any("not numeric" in e for e in result.errors)


def test_unknown_category_is_a_warning_not_an_error():
    result = check_request({**VALID, "Contract": "Lifetime plan"})
    assert result.valid
    assert any("unexpected category" in w for w in result.warnings)


def test_boundary_values_are_valid():
    result = check_request({**VALID, "tenure": 0, "MonthlyCharges": 0})
    assert result.valid
    assert result.errors == []


def test_summary_reflects_valid_state():
    result = check_request(VALID)
    assert "✅" in result.summary()


def test_summary_reflects_invalid_state():
    result = check_request({**VALID, "tenure": -1})
    summary = result.summary()
    assert "❌" in summary
