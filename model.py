"""Training script for startup funding success and amount prediction.

The script cleans the raw funding dataset, engineers simple date features,
fits a binary classifier to estimate funding success (defined here as being
above-median funding on the training set) and a regressor to estimate expected
funding amount.
It prints evaluation metrics and the strongest drivers discovered by the
classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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

DATA_PATH = Path(__file__).resolve().parent / "startup_funding.csv"
SAMPLE_PREVIEW = 5
REGRESSOR_PARAMS = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
    "min_samples_leaf": 5,
    "max_depth": 15,
}

RENAME_MAP = {
    "Sr No": "sr_no",  # BOM is stripped during CSV load via utf-8-sig encoding.
    "Date dd/mm/yyyy": "date",
    "Startup Name": "startup_name",
    "Industry Vertical": "industry_vertical",
    "SubVertical": "sub_vertical",
    "City  Location": "city_location",
    "Investors Name": "investors_name",
    # Source column literal is "InvestmentnType" (misses the 'e' in "Investment"); keep exact key for mapping.
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

@dataclass
class ModelSplit:
    X_test: pd.DataFrame
    success_test: pd.Series
    log_amount_test: pd.Series


@dataclass
class ModelArtifacts:
    classifier: Pipeline
    regressor: Pipeline
    split: ModelSplit


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=RENAME_MAP)

    df["amount_usd"] = (
        df["amount_usd"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    df["amount_usd"] = pd.to_numeric(
        df["amount_usd"], errors="coerce"
    )  # Coerce malformed amounts to NaN for safe dropping later.

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=["amount_usd", "year", "month"])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def train_models(
    features: pd.DataFrame, amount_target: pd.Series
) -> ModelArtifacts:
    X_train, X_test, amount_train, amount_test = train_test_split(
        features, amount_target, test_size=0.2, random_state=42
    )
    log_amount_train = np.log1p(amount_train)
    log_amount_test = np.log1p(amount_test)

    median_train = amount_train.median()
    success_train = (amount_train >= median_train).astype(int)
    success_test = (amount_test >= median_train).astype(int)

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
    classifier.fit(X_train, success_train)

    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(**REGRESSOR_PARAMS),
            ),
        ]
    )
    regressor.fit(X_train, log_amount_train)

    return ModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        split=ModelSplit(X_test, success_test, log_amount_test),
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
    parser = argparse.ArgumentParser(
        description="Train funding success classifier and funding amount regressor models with metrics and top-driver summary."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the funding CSV file (defaults to repository dataset).",
    )
    args = parser.parse_args()

    df = load_dataset(args.data_path)
    feature_frame = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    artifacts = train_models(feature_frame, df["amount_usd"])
    classifier = artifacts.classifier
    regressor = artifacts.regressor
    split = artifacts.split

    success_pred = classifier.predict(split.X_test)
    success_proba = classifier.predict_proba(split.X_test)[:, 1]
    print("Classification metrics for funding success (above-median flag):")
    print(classification_report(split.success_test, success_pred, digits=3))
    print(f"ROC AUC: {roc_auc_score(split.success_test, success_proba):.3f}")

    log_amount_pred = regressor.predict(split.X_test)
    amount_pred = np.expm1(log_amount_pred)
    true_amount = np.expm1(split.log_amount_test)
    mae = mean_absolute_error(true_amount, amount_pred)
    r2 = r2_score(true_amount, amount_pred)
    print("\nRegression metrics for expected funding amount:")
    print(f"MAE (USD): {mae:,.0f}")
    print(f"R^2: {r2:.3f}")

    display_top_drivers(classifier)

    sample_size = min(SAMPLE_PREVIEW, len(split.X_test))
    sample_output = split.X_test.iloc[:sample_size].copy()
    sample_output["success_probability"] = success_proba[:sample_size]
    sample_output["predicted_amount_usd"] = amount_pred[:sample_size]
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
