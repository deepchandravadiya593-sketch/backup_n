
"""NUMPA v2 — Outlier detection and handling.

Methods:
    - IQR
    - Z-score
    - Modified Z-score (median absolute deviation)

Actions:
    - detect: report outliers without changing data
    - remove: remove rows containing selected outliers
    - clip: replace outliers with fitted boundary values
    - replace_null: replace outliers with missing values

Fit on training data and reuse the fitted configuration
on validation and test data to prevent data leakage.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


OutlierMethod = Literal["iqr", "zscore", "modified_zscore"]
OutlierAction = Literal["detect", "remove", "clip", "replace_null"]


class OutlierConfig(BaseModel):
    """User-approved configuration for handling outliers."""

    method: OutlierMethod = "iqr"
    action: OutlierAction = "detect"

    # Explicit selection protects targets and identifier columns.
    columns: list[str] = Field(min_length=1)

    # Standard IQR multiplier is 1.5.
    iqr_multiplier: float = Field(default=1.5, gt=0, le=100)

    # Conventional Z-score threshold is 3.
    zscore_threshold: float = Field(default=3.0, gt=0, le=100)

    # Common modified Z-score threshold is 3.5.
    modified_zscore_threshold: float = Field(
        default=3.5,
        gt=0,
        le=100,
    )

    @model_validator(mode="after")
    def validate_columns(self) -> "OutlierConfig":
        """Ensure each column is selected only once."""

        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Selected columns contain duplicates.")

        return self


def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """Verify that selected columns contain real numeric data."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Resolve duplicate column names before detecting outliers."
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
                f"Column '{column}' must contain real numeric values."
            )

        values = series.to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        if np.isinf(values).any():
            raise ValueError(
                f"Column '{column}' contains infinity. "
                "Clean infinite values before outlier detection."
            )


def fit_outlier_detector(
    dataframe: pd.DataFrame,
    config: OutlierConfig,
) -> dict[str, Any]:
    """Calculate reusable outlier boundaries for each column."""

    _validate_columns(dataframe, config.columns)

    parameters: dict[str, dict[str, Any]] = {}

    for column in config.columns:
        values = dataframe[column].to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        valid = values[np.isfinite(values)]

        if valid.size == 0:
            raise ValueError(
                f"Column '{column}' has no valid numeric values."
            )

        if config.method == "iqr":
            q1 = float(np.percentile(valid, 25))
            q3 = float(np.percentile(valid, 75))
            iqr = q3 - q1

            lower = q1 - config.iqr_multiplier * iqr
            upper = q3 + config.iqr_multiplier * iqr

            parameters[column] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": float(lower),
                "upper": float(upper),
            }

        elif config.method == "zscore":
            mean = float(np.mean(valid))
            std = float(np.std(valid, ddof=0))

            # A constant training column has no outliers.
            if std == 0:
                lower = upper = mean
            else:
                lower = mean - config.zscore_threshold * std
                upper = mean + config.zscore_threshold * std

            parameters[column] = {
                "mean": mean,
                "std": std,
                "lower": float(lower),
                "upper": float(upper),
            }

        else:
            median = float(np.median(valid))
            mad = float(
                np.median(np.abs(valid - median))
            )

            # Modified Z-score:
            # 0.6745 * (value - median) / MAD
            #
            # If MAD is zero, the method cannot distinguish
            # outliers reliably. Flag this as an unscorable
            # column rather than inventing a threshold.
            if mad == 0:
                lower = None
                upper = None
            else:
                distance = (
                    config.modified_zscore_threshold
                    * mad
                    / 0.6745
                )
                lower = median - distance
                upper = median + distance

            parameters[column] = {
                "median": median,
                "mad": mad,
                "lower": (
                    float(lower) if lower is not None else None
                ),
                "upper": (
                    float(upper) if upper is not None else None
                ),
            }

    return {
        "version": 1,
        "method": config.method,
        "action": config.action,
        "columns": list(config.columns),
        "parameters": parameters,
        "fitted_rows": int(len(dataframe)),
    }


def transform_outliers(
    dataframe: pd.DataFrame,
    fitted_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect or handle outliers using previously fitted boundaries."""

    columns = fitted_config["columns"]
    parameters = fitted_config["parameters"]
    method = fitted_config["method"]
    action = fitted_config["action"]

    if method not in {"iqr", "zscore", "modified_zscore"}:
        raise ValueError(f"Unsupported outlier method: {method}")

    if action not in {"detect", "remove", "clip", "replace_null"}:
        raise ValueError(f"Unsupported outlier action: {action}")

    _validate_columns(dataframe, columns)

    # Use positional masks because DataFrame indices may repeat.
    combined_mask = np.zeros(len(dataframe), dtype=bool)
    column_masks: dict[str, np.ndarray] = {}
    outliers_by_column: dict[str, int] = {}
    skipped_columns: list[str] = []

    for column in columns:
        values = dataframe[column].to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        params = parameters[column]
        lower = params["lower"]
        upper = params["upper"]

        if lower is None or upper is None:
            # Modified Z-score cannot operate when fitted MAD
            # is zero. Report the skipped check explicitly.
            skipped_columns.append(column)
            mask = np.zeros(len(dataframe), dtype=bool)

        else:
            # Missing values are not classified as outliers.
            mask = (
                np.isfinite(values)
                & (
                    (values < float(lower))
                    | (values > float(upper))
                )
            )

        column_masks[column] = mask
        combined_mask |= mask
        outliers_by_column[column] = int(mask.sum())

    affected_positions = np.flatnonzero(
        combined_mask
    ).astype(int).tolist()

    # Never mutate the uploaded original dataset.
    cleaned = dataframe.copy(deep=True)

    if action == "remove":
        cleaned = cleaned.iloc[~combined_mask].copy()

    elif action == "clip":
        for column in columns:
            params = parameters[column]

            if column in skipped_columns:
                continue

            mask = column_masks[column]

            if mask.any():
                # Replace only detected outliers, preserving
                # missing values and unaffected observations.
                values = cleaned[column].to_numpy(
                    dtype="float64",
                    na_value=np.nan,
                )

                clipped = np.clip(
                    values,
                    params["lower"],
                    params["upper"],
                )

                cleaned[column] = np.where(
                    mask,
                    clipped,
                    values,
                )

    elif action == "replace_null":
        for column in columns:
            mask = column_masks[column]

            if mask.any():
                # Assign positionally to avoid problems
                # with duplicate row index labels.
                column_position = cleaned.columns.get_loc(
                    column
                )

                cleaned.iloc[
                    np.flatnonzero(mask),
                    column_position,
                ] = np.nan

    # The detect action intentionally changes nothing.

    report: dict[str, Any] = {
        "operation": "outlier_handling",
        "method": method,
        "action": action,
        "selected_columns": list(columns),
        "rows_before": int(len(dataframe)),
        "rows_after": int(len(cleaned)),
        "rows_removed": int(
            len(dataframe) - len(cleaned)
        ),
        "outliers_by_column": outliers_by_column,
        "total_affected_rows": len(affected_positions),
        "affected_row_positions": affected_positions,
        "skipped_columns": skipped_columns,
        "fitted_parameters": parameters,
        "status": "completed",
    }

    return cleaned, report


def handle_outliers(
    dataframe: pd.DataFrame,
    config: OutlierConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    """Fit and apply an outlier operation in one call.

    Returns:
        cleaned: A new DataFrame.
        report: The operation's audit report.
        fitted_config: Reusable detection parameters.
    """

    fitted_config = fit_outlier_detector(
        dataframe,
        config,
    )

    cleaned, report = transform_outliers(
        dataframe,
        fitted_config,
    )

    return cleaned, report, fitted_config
