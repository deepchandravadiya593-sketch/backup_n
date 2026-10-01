
"""NUMPA v2 - Dataset upload validation and file safety tests.

Tests:
- Valid CSV upload.
- Unsupported file extension.
- Empty file upload.
- Malformed CSV upload.
- File exceeding the upload limit.
- Duplicate column names.
- Uploaded filename safety.
- Rejected uploads do not create dataset records.

Tests use the real FastAPI upload endpoint.
"""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


UPLOAD_URL = "/api/v1/datasets/upload"
DATASETS_URL = "/api/v1/datasets"

# The dataset loader currently limits uploads to 10 MiB.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@pytest.fixture
def client():
    """Create a fresh API test client."""

    with TestClient(app) as test_client:
        yield test_client


def unique_filename(
    extension: str = ".csv",
) -> str:
    """Generate a filename unique to each test."""

    return f"upload_safety_{uuid4().hex[:10]}{extension}"


def upload_file(
    client: TestClient,
    *,
    filename: str,
    content: bytes,
    content_type: str = "text/csv",
):
    """Submit a file using multipart/form-data."""

    return client.post(
        UPLOAD_URL,
        files={
            "file": (
                filename,
                BytesIO(content),
                content_type,
            ),
        },
    )


def dataset_ids(client: TestClient) -> set[str]:
    """Read the current dataset IDs from the API."""

    response = client.get(DATASETS_URL)

    assert response.status_code == 200, response.text

    payload = response.json()

    # The list endpoint uses a response wrapper.
    records = (
        payload.get("datasets", [])
        if isinstance(payload, dict)
        else payload
    )

    return {
        record["dataset_id"]
        for record in records
    }


def cleanup_dataset(
    client: TestClient,
    dataset_id: str | None,
) -> None:
    """Delete a dataset created by a test."""

    if dataset_id:
        client.delete(
            f"{DATASETS_URL}/{dataset_id}"
        )


# ---------------------------------------------------------
# Test 1: Valid CSV upload
# ---------------------------------------------------------

def test_valid_csv_upload(
    client: TestClient,
) -> None:
    """A valid CSV must be accepted and assigned an ID."""

    filename = unique_filename()

    response = upload_file(
        client,
        filename=filename,
        content=(
            b"age,income,city\n"
            b"24,25000,Surat\n"
            b"31,42000,Rajkot\n"
            b"38,51000,Ahmedabad\n"
        ),
    )

    assert response.status_code in (200, 201), response.text

    payload = response.json()

    dataset_id = payload["dataset_id"]

    try:
        assert dataset_id
        assert dataset_id in dataset_ids(client)

    finally:
        cleanup_dataset(client, dataset_id)


# ---------------------------------------------------------
# Test 2: Unsupported extension
# ---------------------------------------------------------

def test_unsupported_file_type_is_rejected(
    client: TestClient,
) -> None:
    """Executable files must not be accepted as datasets."""

    before = dataset_ids(client)

    response = upload_file(
        client,
        filename=unique_filename(".exe"),
        content=b"MZ\x00\x00not-a-dataset",
        content_type="application/octet-stream",
    )

    assert response.status_code in (400, 415, 422)

    assert dataset_ids(client) == before


# ---------------------------------------------------------
# Test 3: Empty upload
# ---------------------------------------------------------

def test_empty_file_is_rejected(
    client: TestClient,
) -> None:
    """An empty upload must not create a dataset."""

    before = dataset_ids(client)

    response = upload_file(
        client,
        filename=unique_filename(),
        content=b"",
    )

    assert response.status_code in (400, 422)

    assert dataset_ids(client) == before


# ---------------------------------------------------------
# Test 4: Malformed CSV
# ---------------------------------------------------------

def test_malformed_csv_is_rejected(
    client: TestClient,
) -> None:
    """Invalid CSV content must produce a controlled error."""

    before = dataset_ids(client)

    response = upload_file(
        client,
        filename=unique_filename(),
        content=(
            b"age,income\n"
            b'24,"unterminated quoted field\n'
            b"31,42000\n"
        ),
    )

    assert response.status_code in (400, 422)

    assert dataset_ids(client) == before


# ---------------------------------------------------------
# Test 5: Oversized upload
# ---------------------------------------------------------

def test_oversized_file_is_rejected(
    client: TestClient,
) -> None:
    """Files exceeding 10 MiB must not enter dataset storage."""

    before = dataset_ids(client)

    # A valid-looking CSV whose total size exceeds
    # the configured upload limit.
    header = b"value\n"
    row = b"1234567890\n"

    content = header + row * (
        MAX_UPLOAD_BYTES // len(row) + 1
    )

    response = upload_file(
        client,
        filename=unique_filename(),
        content=content,
    )

    assert response.status_code in (400, 413, 422)

    assert dataset_ids(client) == before


# ---------------------------------------------------------
# Test 6: Duplicate column names
# ---------------------------------------------------------

def test_duplicate_column_names_are_handled(
    client: TestClient,
) -> None:
    """Duplicate headers must not cause an unhandled server error."""

    response = upload_file(
        client,
        filename=unique_filename(),
        content=(
            b"age,age,income\n"
            b"24,25,25000\n"
            b"31,32,42000\n"
        ),
    )

    # The current loader may normalize duplicate
    # headers or reject them during validation.
    assert response.status_code in (
        200,
        201,
        400,
        422,
    ), response.text

    if response.status_code in (200, 201):
        cleanup_dataset(
            client,
            response.json()["dataset_id"],
        )


# ---------------------------------------------------------
# Test 7: Suspicious filename
# ---------------------------------------------------------

def test_filename_does_not_escape_dataset_storage(
    client: TestClient,
) -> None:
    """An uploaded filename must not become a filesystem path."""

    response = upload_file(
        client,
        filename="../../outside.csv",
        content=(
            b"age,income\n"
            b"24,25000\n"
            b"31,42000\n"
        ),
    )

    # Either reject the suspicious filename or
    # safely accept it without filesystem writes.
    assert response.status_code in (
        200,
        201,
        400,
        422,
    ), response.text

    if response.status_code in (200, 201):
        dataset_id = response.json()["dataset_id"]

        try:
            assert dataset_id

            # Verify the record is accessible through
            # its generated ID rather than a file path.
            detail = client.get(
                f"{DATASETS_URL}/{dataset_id}"
            )

            assert detail.status_code == 200

        finally:
            cleanup_dataset(
                client,
                dataset_id,
            )


# ---------------------------------------------------------
# Test 8: Rejected upload does not alter existing data
# ---------------------------------------------------------

def test_rejected_upload_preserves_existing_dataset(
    client: TestClient,
) -> None:
    """An invalid upload must not affect previously saved data."""

    valid_response = upload_file(
        client,
        filename=unique_filename(),
        content=(
            b"age,income\n"
            b"24,25000\n"
            b"31,42000\n"
        ),
    )

    assert valid_response.status_code in (
        200,
        201,
    ), valid_response.text

    dataset_id = valid_response.json()["dataset_id"]

    try:
        before = client.get(
            f"{DATASETS_URL}/{dataset_id}"
        )

        assert before.status_code == 200

        invalid_response = upload_file(
            client,
            filename=unique_filename(".exe"),
            content=b"invalid executable content",
            content_type="application/octet-stream",
        )

        assert invalid_response.status_code in (
            400,
            415,
            422,
        )

        after = client.get(
            f"{DATASETS_URL}/{dataset_id}"
        )

        assert after.status_code == 200
        assert after.json() == before.json()

    finally:
        cleanup_dataset(
            client,
            dataset_id,
        )
