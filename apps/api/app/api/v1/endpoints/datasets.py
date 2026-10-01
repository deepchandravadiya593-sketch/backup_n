
"""NUMPA v2 - Dataset API endpoints.

Supports uploading, listing, inspecting, validating,
exporting and deleting datasets.

The router must be defined before any @router decorators.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.services.analysis.loader import load_dataset
from app.services.analysis.profiler import generate_dataset_profile
from app.services.analysis.validator import validate_dataset
from app.services.analysis.statistics import (
    generate_dataset_statistics,
)
from app.services.reports.pdf import (
    generate_dataset_overview_pdf,
)
from app.services.storage.datasets import (
    delete_dataset,
    export_dataset_csv,
    get_dataset,
    get_dataset_metadata,
    list_datasets,
    save_dataset,
)


# IMPORTANT: Define the router before declaring endpoints.
router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


def get_dataset_dataframe(
    dataset_id: str,
) -> pd.DataFrame:
    """Retrieve a dataset or return HTTP 404.

    Other endpoints, including Cleaning Studio,
    import this function.
    """

    try:
        dataframe = get_dataset(dataset_id)
    except (KeyError, LookupError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        ) from exc

    if dataframe is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise HTTPException(
            status_code=500,
            detail="Stored dataset has an invalid format.",
        )

    return dataframe


def _get_metadata(
    dataset_id: str,
) -> dict[str, Any]:
    """Retrieve dataset metadata or return HTTP 404."""

    try:
        metadata = get_dataset_metadata(dataset_id)
    except (KeyError, LookupError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        ) from exc

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return metadata


def _safe_filename(
    name: str,
    suffix: str,
) -> str:
    """Create a safe filename for HTTP downloads."""

    basename = (
        str(name)
        .replace("\\", "/")
        .split("/")[-1]
    )

    stem = Path(basename).stem

    safe_stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        stem,
    ).strip("_")

    safe_stem = safe_stem[:80] or "dataset"

    return f"{safe_stem}{suffix}"


# ---------------------------------------------------------
# Upload a dataset
# ---------------------------------------------------------

@router.post(
    "/upload",
    summary="Upload a CSV, Excel or JSON dataset",
)
async def upload_dataset(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Validate, profile and store an uploaded dataset."""

    try:
        dataframe = await load_dataset(file)

        metadata = save_dataset(
            dataframe,
            name=file.filename or "dataset",
        )

        profile = generate_dataset_profile(
            dataframe
        )

        validation = validate_dataset(
            dataframe
        )

        return {
            **metadata,
            "profile": profile,
            "validation": validation,
        }

    except HTTPException:
        raise

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()


# ---------------------------------------------------------
# List datasets
# ---------------------------------------------------------

@router.get(
    "",
    summary="List uploaded and cleaned datasets",
)
def get_datasets() -> dict[str, Any]:
    """Return all datasets currently held in storage."""

    datasets = list_datasets()

    return {
        "datasets": datasets,
        "total": len(datasets),
    }


# ---------------------------------------------------------
# Export dataset as CSV
# ---------------------------------------------------------

@router.get(
    "/{dataset_id}/export",
    summary="Download dataset as CSV",
)
def download_dataset_csv(
    dataset_id: str,
) -> Response:
    """Download the original or cleaned dataset."""

    metadata = _get_metadata(
        dataset_id
    )

    try:
        csv_data = export_dataset_csv(
            dataset_id
        )
    except (KeyError, LookupError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        ) from exc

    # Support storage services returning either text
    # or already encoded CSV bytes.
    if isinstance(csv_data, str):
        csv_bytes = csv_data.encode("utf-8-sig")
    elif isinstance(csv_data, bytes):
        csv_bytes = csv_data
    else:
        raise HTTPException(
            status_code=500,
            detail="CSV export returned an invalid format.",
        )

    filename = _safe_filename(
        metadata.get("name", "dataset"),
        ".csv",
    )

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------

@router.get(
    "/{dataset_id}/validation",
    summary="Get dataset validation results",
)
def get_dataset_validation(
    dataset_id: str,
) -> dict[str, Any]:
    """Run NUMPA's dataset validation checks."""

    dataframe = get_dataset_dataframe(
        dataset_id
    )

    return validate_dataset(
        dataframe
    )


# ---------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------

@router.get(
    "/{dataset_id}/statistics",
    summary="Get dataset statistics",
)
def get_dataset_statistics(
    dataset_id: str,
) -> dict[str, Any]:
    """Calculate dataset statistics."""

    dataframe = get_dataset_dataframe(
        dataset_id
    )

    return generate_dataset_statistics(
        dataframe
    )


# ---------------------------------------------------------
# Dataset overview PDF
# ---------------------------------------------------------

@router.get(
    "/{dataset_id}/report/overview",
    summary="Download dataset overview PDF",
)
def download_dataset_overview_pdf(
    dataset_id: str,
    include_preview: bool = True,
) -> StreamingResponse:
    """Generate a downloadable PDF overview report."""

    dataframe = get_dataset_dataframe(
        dataset_id
    )

    metadata = _get_metadata(
        dataset_id
    )

    dataset_name = metadata.get(
        "name",
        "Dataset",
    )

    try:
        pdf_bytes = generate_dataset_overview_pdf(
            dataframe,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            include_preview=include_preview,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=500,
            detail="PDF generation failed.",
        )

    filename = _safe_filename(
        dataset_name,
        "_overview.pdf",
    )

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "Content-Length": str(
                len(pdf_bytes)
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------
# Get a single dataset
# ---------------------------------------------------------

@router.get(
    "/{dataset_id}",
    summary="Get dataset details and profile",
)
def get_dataset_details(
    dataset_id: str,
) -> dict[str, Any]:
    """Return dataset metadata and its current profile."""

    dataframe = get_dataset_dataframe(
        dataset_id
    )

    metadata = _get_metadata(
        dataset_id
    )

    return {
        **metadata,
        "profile": generate_dataset_profile(
            dataframe
        ),
    }


# ---------------------------------------------------------
# Delete a dataset
# ---------------------------------------------------------

@router.delete(
    "/{dataset_id}",
    summary="Delete a stored dataset",
)
def remove_dataset(
    dataset_id: str,
) -> dict[str, Any]:
    """Delete a dataset, respecting storage lineage rules."""

    # Confirm the record exists before attempting deletion.
    _get_metadata(
        dataset_id
    )

    try:
        delete_dataset(
            dataset_id
        )

    except (KeyError, LookupError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        ) from exc

    except ValueError as exc:
        # For example, a dataset may still have
        # cleaned descendants referencing it.
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return {
        "dataset_id": dataset_id,
        "deleted": True,
    }
