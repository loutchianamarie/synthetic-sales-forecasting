import pandas as pd
import pytest

from synthetic_forecasting.pipeline import evaluate, features, generate, validate


def test_generated_data_is_deterministic_and_valid():
    first = generate(seed=5)
    pd.testing.assert_frame_equal(first, generate(seed=5))
    assert len(first) == 4 * 48
    assert set(first["product_group"]) == {"A", "B", "C", "D"}
    assert len(validate(first)) == len(first)


def test_lag_features_use_prior_observations_only():
    data = generate()
    a = features(data).query("product_group == 'A'").iloc[0]
    original = data.query("product_group == 'A'").reset_index(drop=True)
    assert a["lag_1"] == original.loc[11, "units"]
    assert a["lag_12"] == original.loc[0, "units"]


def test_holdout_evaluation_has_expected_rows():
    result = evaluate(generate(), test_months=6)
    assert result.rows_test == 4 * 6
    assert result.rows_train == 4 * (48 - 12 - 6)
    assert result.baseline_mae >= 0
    assert result.forest_mae >= 0


def test_bad_data_is_rejected():
    data = generate()
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([data, data.iloc[[0]]]))
    data.loc[0, "units"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate(data)
