
"""NUMPA v2 — Numerical scaling engine.

Supports Min-Max, Standard and Robust scaling.
Fitted parameters can be reused on new datasets.

IMPORTANT:
For ML workflows, fit only on training data, then transform
validation and test data using the saved fitted parameters.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


ScalingStrategy = Literal["min_max", "standard", "robust"]


class ScalingConfig(BaseModel):
    """Configuration approved by the user."""

    strategy: ScalingStrategy

    # Require explicit column selection to avoid scaling
    # targets, identifiers or unrelated numeric columns.
    columns: list[str] = Field(min_length=1)

    # Only used for Min-Max scaling.
    feature_range: tuple[float, float] = (0.0, 1.0)

    @model_validator(mode="after")
    def validate_config(self) -> "ScalingConfig":
        """Reject invalid scaling configurations."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Selected columns contain duplicates.")

        minimum, maximum = self.feature_range

        if not (
            np.isfinite(minimum)
            and np.isfinite(maximum)
            and minimum < maximum
        ):
            raise ValueError(
                "feature_range must contain two finite "
                "values in increasing order."
            )

        return self


def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """Check that selected columns can be scaled."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before scaling."
        )

    for column in columns:
        if column not in dataframe.columns:
            raise ValueError(f"Unknown column: {column}")

        series = dataframe[column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise ValueError(
                f"Column '{column}' must contain real "
                "numeric values."
            )

        # Infinity is not a missing value. Reject it rather
        # than silently treating it as valid training data.
        values = series.to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        if np.isinf(values).any():
            raise ValueError(
                f"Column '{column}' contains infinite values. "
                "Clean them before scaling."
            )


def fit_scaler(
    dataframe: pd.DataFrame,
    config: ScalingConfig,
) -> dict[str, Any]:
    """Learn scaling parameters from the supplied dataset."""

    _validate_columns(dataframe, config.columns)

    parameters: dict[str, dict[str, Any]] = {}

    for column in config.columns:
        # Drop missing values when calculating parameters.
        # Missing values remain missing during transformation.
        values = (
            dataframe[column]
            .to_numpy(dtype="float64", na_value=np.nan)
        )

        valid = values[~np.isnan(values)]

        if valid.size == 0:
            raise ValueError(
                f"Column '{column}' contains no valid values "
                "from which to fit a scaler."
            )

        if config.strategy == "min_max":
            minimum = float(np.min(valid))
            maximum = float(np.max(valid))

            parameters[column] = {
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
            }

        elif config.strategy == "standard":
            mean = float(np.mean(valid))

            # Population standard deviation matches the
            # conventional StandardScaler calculation.
            std = float(np.std(valid, ddof=0))

            parameters[column] = {
                "mean": mean,
                "std": std,
            }

        elif config.strategy == "robust":
            median = float(np.median(valid))
            q1 = float(np.percentile(valid, 25))
            q3 = float(np.percentile(valid, 75))

            parameters[column] = {
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
            }

    return {
        "strategy": config.strategy,
        "columns": list(config.columns),
        "feature_range": list(config.feature_range),
        "parameters": parameters,
        "fitted_rows": int(len(dataframe)),
        "version": 1,
    }


def transform_scaler(
    dataframe: pd.DataFrame,
    fitted_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform a dataset using previously fitted parameters."""

    strategy = fitted_config["strategy"]
    columns = fitted_config["columns"]
    parameters = fitted_config["parameters"]

    if strategy not in {"min_max", "standard", "robust"}:
        raise ValueError(
            f"Unsupported scaling strategy: {strategy}"
        )

    _validate_columns(dataframe, columns)

    # Always preserve the original dataset.
    cleaned = dataframe.copy(deep=True)

    missing_counts: dict[str, int] = {}
    constant_columns: list[str] = []

    for column in columns:
        values = (
            cleaned[column]
            .to_numpy(dtype="float64", na_value=np.nan)
        )

        missing_counts[column] = int(
            np.isnan(values).sum()
        )

        params = parameters[column]

        if strategy == "min_max":
            source_min = float(params["min"])
            source_range = float(params["range"])

            target_min, target_max = (
                fitted_config["feature_range"]
            )

            if source_range == 0:
                # A constant training column contains no
                # variation. Map finite values to target_min.
                scaled = np.where(
                    np.isnan(values),
                    np.nan,
                    float(target_min),
                )
                constant_columns.append(column)

            else:
                scaled = (
                    (values - source_min)
                    / source_range
                    * (target_max - target_min)
                    + target_min
                )

        elif strategy == "standard":
            mean = float(params["mean"])
            std = float(params["std"])

            if std == 0:
                # Constant columns map to zero.
                scaled = np.where(
                    np.isnan(values),
                    np.nan,
                    0.0,
                )
                constant_columns.append(column)

            else:
                scaled = (values - mean) / std

        else:
            median = float(params["median"])
            iqr = float(params["iqr"])

            if iqr == 0:
                # Avoid division by zero while retaining
                # differences from the training median.
                scaled = values - median
                constant_columns.append(column)

            else:
                scaled = (values - median) / iqr

        # Reject overflow rather than producing invalid
        # values that may break downstream ML algorithms.
        if np.isinf(scaled).any():
            raise ValueError(
                f"Scaling column '{column}' produced infinity."
            )

        cleaned[column] = scaled

    report: dict[str, Any] = {
        "operation": "numerical_scaling",
        "strategy": strategy,
        "selected_columns": list(columns),
        "rows_before": int(len(dataframe)),
        "rows_after": int(len(cleaned)),
        "columns_before": int(len(dataframe.columns)),
        "columns_after": int(len(cleaned.columns)),
        "missing_values_by_column": missing_counts,
        "zero_scale_columns": constant_columns,
        "fitted_parameters": parameters,
        "status": "completed",
    }

    return cleaned, report


def scale_numeric(
    dataframe: pd.DataFrame,
    config: ScalingConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    """Fit and transform in one operation.

    Returns:
        cleaned: A new scaled DataFrame.
        report: An audit report.
        fitted_config: Parameters for future transformations.
    """

    fitted_config = fit_scaler(
        dataframe,
        config,
    )

    cleaned, report = transform_scaler(
        dataframe,
        fitted_config,
    )

    return cleaned, report, fitted_config
