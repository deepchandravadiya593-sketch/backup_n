
"""NUMPA v2 — Shared dataset storage service.

This development-stage service stores pandas DataFrames in memory.

Original and cleaned datasets receive separate IDs. Every stored
DataFrame is copied on write and on read to prevent accidental
modification of the stored version.

Production storage and per-user authorization will be added later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

import pandas as pd


# Limit memory usage during the initial development stage.
MAX_STORED_DATASETS = 20

# Distinguish uploaded originals from cleaning results.
DatasetKind = Literal["original", "cleaned"]

# Store records in memory until persistent storage is implemented.
_dataset_store: dict[str, dict[str, Any]] = {}

# Protect the shared dictionary during concurrent requests.
_store_lock = RLock()


def _utc_now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Reject invalid or empty datasets before saving."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The dataset must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot store an empty dataset.")

    if not dataframe.columns.is_unique:
        raise ValueError("Dataset column names must be unique.")


def _make_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return public metadata without exposing the stored DataFrame."""

    dataframe: pd.DataFrame = record["dataframe"]

    return {
        "dataset_id": record["dataset_id"],
        "name": record["name"],
        "kind": record["kind"],
        "parent_dataset_id": record["parent_dataset_id"],
        "created_at": record["created_at"],
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": [str(column) for column in dataframe.columns],
        "memory_usage_bytes": int(
            dataframe.memory_usage(index=True, deep=True).sum()
        ),
    }


def save_dataset(
    dataframe: pd.DataFrame,
    *,
    name: str,
    kind: DatasetKind = "original",
    parent_dataset_id: str | None = None,
) -> dict[str, Any]:
    """Save a dataset and return its metadata.

    Cleaned datasets must reference an existing parent dataset.
    Original datasets cannot have a parent.
    """

    _validate_dataframe(dataframe)

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name cannot be empty.")

    if kind not in ("original", "cleaned"):
        raise ValueError("Dataset kind must be 'original' or 'cleaned'.")

    if kind == "original" and parent_dataset_id is not None:
        raise ValueError("An original dataset cannot have a parent.")

    if kind == "cleaned" and not parent_dataset_id:
        raise ValueError("A cleaned dataset requires a parent dataset ID.")

    with _store_lock:
        # Validate the parent while holding the same lock used to save.
        if (
            parent_dataset_id is not None
            and parent_dataset_id not in _dataset_store
        ):
            raise KeyError(
                f"Parent dataset not found: {parent_dataset_id}"
            )

        if len(_dataset_store) >= MAX_STORED_DATASETS:
            raise ValueError(
                "Dataset storage is full. Delete an existing dataset "
                "before uploading another."
            )

        dataset_id = str(uuid4())

        record = {
            "dataset_id": dataset_id,
            "name": name.strip(),
            "kind": kind,
            "parent_dataset_id": parent_dataset_id,
            "created_at": _utc_now(),
            "dataframe": dataframe.copy(deep=True),
        }

        _dataset_store[dataset_id] = record

        return _make_metadata(record)


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve an independent copy of a stored dataset."""

    with _store_lock:
        record = _dataset_store.get(dataset_id)

        if record is None:
            raise KeyError(f"Dataset not found: {dataset_id}")

        return record["dataframe"].copy(deep=True)


def get_dataset_metadata(dataset_id: str) -> dict[str, Any]:
    """Retrieve metadata for one stored dataset."""

    with _store_lock:
        record = _dataset_store.get(dataset_id)

        if record is None:
            raise KeyError(f"Dataset not found: {dataset_id}")

        return _make_metadata(record)


def list_datasets() -> list[dict[str, Any]]:
    """List all datasets, newest first."""

    with _store_lock:
        records = sorted(
            _dataset_store.values(),
            key=lambda record: record["created_at"],
            reverse=True,
        )

        return [_make_metadata(record) for record in records]


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset if it has no dependent cleaned datasets.

    Returning False means the dataset does not exist.
    A parent cannot be deleted while cleaned datasets reference it.
    """

    with _store_lock:
        if dataset_id not in _dataset_store:
            return False

        has_children = any(
            record["parent_dataset_id"] == dataset_id
            for record in _dataset_store.values()
        )

        if has_children:
            raise ValueError(
                "Delete dependent cleaned datasets before deleting "
                "their parent dataset."
            )

        del _dataset_store[dataset_id]

        return True


def export_dataset_csv(dataset_id: str) -> bytes:
    """Export a stored dataset as UTF-8 CSV bytes.

    The UTF-8 BOM helps spreadsheet applications recognize
    non-ASCII characters correctly.
    """

    dataframe = get_dataset(dataset_id)

    buffer = BytesIO()

    # Do not export the pandas index as an extra CSV column.
    csv_content = dataframe.to_csv(index=False)

    buffer.write(csv_content.encode("utf-8-sig"))

    return buffer.getvalue()


def count_datasets() -> int:
    """Return the number of datasets currently in memory."""

    with _store_lock:
        return len(_dataset_store)
