"""Training script for startup funding success and amount prediction.

The script cleans the raw funding dataset, engineers simple date features,
fits a binary classifier to estimate funding success (defined here as being
above-median funding) and a regressor to estimate expected funding amount.
It prints evaluation metrics and the strongest drivers discovered by the
classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent / "startup_funding (1).csv"

RENAME_MAP = {
    "\ufeffSr No": "sr_no",
    "Date dd/mm/yyyy": "date",
    "Startup Name": "startup_name",
    "Industry Vertical": "industry_vertical",
    "SubVertical": "sub_vertical",
    "City  Location": "city_location",
    "Investors Name": "investors_name",
    "InvestmentnType": "investment_type",
    "Amount in USD": "amount_usd",
    "Remarks": "remarks",
}

CATEGORICAL_FEATURES = [
    "startup_name",
    "industry_vertical",
    "sub_vertical",
    "city_location",
    "investors_name",
    "investment_type",
]
NUMERIC_FEATURES = ["year", "month"]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME_MAP)

    df["amount_usd"] = (
        df["amount_usd"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\\s+", "", regex=True)
    )
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=["amount_usd", "year", "month"])
    df["success_flag"] = (df["amount_usd"] >= df["amount_usd"].median()).astype(int)
    df["log_amount"] = np.log1p(df["amount_usd"])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def train_models(
    features: pd.DataFrame,
    success_target: pd.Series,
    log_amount_target: pd.Series,
) -> Tuple[Pipeline, Pipeline, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    (
        X_train,
        X_test,
        y_success_train,
        y_success_test,
        y_amount_train,
        y_amount_test,
    ) = train_test_split(
        features,
        success_target,
        log_amount_target,
        test_size=0.2,
        random_state=42,
        stratify=success_target,
    )

    classifier = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_success_train)

    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=1,
                    max_depth=None,
                ),
            ),
        ]
    )
    regressor.fit(X_train, y_amount_train)

    return classifier, regressor, (
        X_test,
        y_success_test,
        y_amount_test,
        y_amount_train.index,
    )


def display_top_drivers(classifier: Pipeline, top_n: int = 10) -> None:
    preprocessor = classifier.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coef = classifier.named_steps["classifier"].coef_[0]
    order = np.argsort(np.abs(coef))[::-1][:top_n]

    print("\nTop drivers of funding success (by absolute coefficient):")
    for idx in order:
        print(f"  {feature_names[idx]}: {coef[idx]:.3f}")


def main() -> None:
    df = load_dataset()
    feature_frame = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    classifier, regressor, split = train_models(
        feature_frame, df["success_flag"], df["log_amount"]
    )
    X_test, y_success_test, y_amount_test, _ = split

    success_pred = classifier.predict(X_test)
    success_proba = classifier.predict_proba(X_test)[:, 1]
    print("Classification metrics for funding success (above-median flag):")
    print(classification_report(y_success_test, success_pred, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_success_test, success_proba):.3f}")

    log_amount_pred = regressor.predict(X_test)
    amount_pred = np.expm1(log_amount_pred)
    true_amount = np.expm1(y_amount_test)
    mae = mean_absolute_error(true_amount, amount_pred)
    r2 = r2_score(true_amount, amount_pred)
    print("\nRegression metrics for expected funding amount:")
    print(f"MAE (USD): {mae:,.0f}")
    print(f"R^2: {r2:.3f}")

    display_top_drivers(classifier)

    sample_output = pd.DataFrame(X_test).head(5).copy()
    sample_output["success_probability"] = success_proba[: len(sample_output)]
    sample_output["predicted_amount_usd"] = amount_pred[: len(sample_output)]
    print("\nSample predictions:")
    print(
        sample_output[
            [
                "startup_name",
                "industry_vertical",
                "city_location",
                "investment_type",
                "success_probability",
                "predicted_amount_usd",
            ]
        ]
        .sort_values(by="success_probability", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
