
"""NUMPA dataset loading engine.

Supports CSV, Excel (.xlsx) and JSON datasets.
Performs basic file validation before loading data into pandas.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile


# Initial upload limit: 10 MB.
# Later, this will depend on the user's subscription.
MAX_FILE_SIZE = 10 * 1024 * 1024

# Only accept the file formats supported by NUMPA.
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json"}


async def load_dataset(file: UploadFile) -> pd.DataFrame:
    """Validate and load an uploaded dataset into a DataFrame."""

    # Extract the filename and validate its extension.
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use CSV, XLSX or JSON.",
        )

    try:
        # Read at most one byte beyond the limit so oversized
        # uploads can be rejected without loading the whole file.
        contents = await file.read(MAX_FILE_SIZE + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="The file exceeds the 10 MB upload limit.",
            )

        # Load the dataset according to its file extension.
        buffer = BytesIO(contents)

        if extension == ".csv":
            dataframe = pd.read_csv(buffer)

        elif extension == ".xlsx":
            dataframe = pd.read_excel(
                buffer,
                engine="openpyxl",
            )

        else:
            dataframe = pd.read_json(buffer)

        # Reject datasets without rows or columns.
        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The dataset must contain rows and columns.",
            )

        # Reject duplicate column names because they can cause
        # ambiguous cleaning and readiness calculations.
        if not dataframe.columns.is_unique:
            raise HTTPException(
                status_code=400,
                detail="The dataset contains duplicate column names.",
            )

        return dataframe

    except HTTPException:
        # Preserve validation errors and their HTTP status codes.
        raise

    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to parse the dataset: {exc}",
        ) from exc

    except Exception as exc:
        # Avoid exposing internal error details to API clients.
        raise HTTPException(
            status_code=422,
            detail="The uploaded file could not be processed.",
        ) from exc

    finally:
        # Close the uploaded file after processing.
        await file.close()
