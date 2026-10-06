"""Generated data, lag features, and rolling one-step evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PRODUCTS = ("A", "B", "C", "D")


def generate(seed: int = 42, months: int = 48) -> pd.DataFrame:
    """Create fictional monthly units for four anonymous product groups."""
    if months < 24:
        raise ValueError("At least 24 months are required for seasonal lags")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=months, freq="MS")
    rows = []
    for group_idx, product in enumerate(PRODUCTS):
        for step, date in enumerate(dates):
            seasonal = 18 * np.sin(2 * np.pi * (date.month - 1) / 12 + group_idx / 3)
            value = 100 + group_idx * 24 + 0.8 * step + seasonal + rng.normal(0, 5)
            rows.append({"month": date, "product_group": product, "units": round(max(value, 0), 2)})
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "product_group", "units"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected columns: {sorted(required)}")
    data = frame[["month", "product_group", "units"]].copy()
    data["month"] = pd.to_datetime(data["month"], errors="raise")
    if data["month"].dt.day.ne(1).any():
        raise ValueError("Months must be first-of-month dates")
    if data.duplicated(["month", "product_group"]).any():
        raise ValueError("Duplicate product-month records")
    if data["units"].isna().any() or (data["units"] < 0).any():
        raise ValueError("Units must be nonnegative and present")
    if not np.isfinite(data["units"].to_numpy(dtype=float)).all():
        raise ValueError("Units must be finite")
    return data.sort_values(["product_group", "month"]).reset_index(drop=True)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    data = validate(frame)
    grouped = data.groupby("product_group", sort=False)["units"]
    data["lag_1"] = grouped.shift(1)
    data["lag_12"] = grouped.shift(12)
    data["month_number"] = data["month"].dt.month
    data["trend_month"] = (data["month"].dt.year - 2020) * 12 + data["month"].dt.month
    return data.dropna().reset_index(drop=True)


@dataclass(frozen=True)
class Evaluation:
    cutoff: str
    test_months: int
    baseline_mae: float
    forest_mae: float
    rows_train: int
    rows_test: int


def evaluate(frame: pd.DataFrame, test_months: int = 6) -> Evaluation:
    """Compare seasonal naive and forest on the last N months, one step ahead."""
    data = features(frame)
    months = sorted(data["month"].unique())
    if test_months < 1 or len(months) <= test_months + 6:
        raise ValueError("Holdout must leave at least seven feature months for training")
    cutoff = months[-test_months]
    train = data[data["month"] < cutoff]
    test = data[data["month"] >= cutoff]
    columns = ["lag_1", "lag_12", "month_number", "trend_month"]
    train_x = pd.get_dummies(train[[*columns, "product_group"]], columns=["product_group"])
    test_x = pd.get_dummies(test[[*columns, "product_group"]], columns=["product_group"])
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
    model.fit(train_x, train["units"])
    forest = model.predict(test_x)
    baseline = test["lag_12"].to_numpy()
    return Evaluation(
        cutoff=pd.Timestamp(cutoff).strftime("%Y-%m"),
        test_months=test_months,
        baseline_mae=round(float(mean_absolute_error(test["units"], baseline)), 3),
        forest_mae=round(float(mean_absolute_error(test["units"], forest)), 3),
        rows_train=len(train),
        rows_test=len(test),
    )
