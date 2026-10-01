
"""NUMPA v2 - Readiness configuration API integration tests.

Tests the HTTP endpoints for:
- Built-in configuration retrieval.
- Creating and retrieving custom configurations.
- Creating a new configuration version.
- Duplicating configurations.
- Archiving and restoring configurations.
- Rejecting invalid scoring weights.

The configuration store is currently in memory.
Use unique names to avoid collisions between test runs.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.readiness import DEFAULT_DIMENSION_WEIGHTS


BASE = "/api/v1/readiness/configurations"


@pytest.fixture
def client():
    """Create an API test client."""

    with TestClient(app) as test_client:
        yield test_client


def unique_name(prefix: str) -> str:
    """Generate a unique configuration name."""

    return f"{prefix}-{uuid4().hex[:10]}"


def dimensions(
    *,
    custom: bool = False,
) -> dict:
    """Build valid NUMPA scoring dimensions."""

    result = {
        name: {
            "weight": weight,
            "enabled": True,
            "thresholds": {},
        }
        for name, weight in DEFAULT_DIMENSION_WEIGHTS.items()
    }

    if custom:
        # Redistribute five percentage points
        # without changing the total weight.
        result["completeness"]["weight"] = 15.0
        result["leakage_validation"]["weight"] = 20.0

    return result


def configuration_payload(
    name: str,
    *,
    custom: bool = False,
) -> dict:
    """Build a valid custom scoring configuration."""

    return {
        "name": name,
        "version": 1,
        "dimensions": dimensions(custom=custom),
        "parameters": {},
    }


def unwrap_configuration(data: dict) -> dict:
    """Handle a full configuration or a wrapped response."""

    if isinstance(data.get("configuration"), dict):
        return data["configuration"]

    return data


def configuration_id_from(data: dict) -> str:
    """Extract the ID from a configuration API response."""

    config = unwrap_configuration(data)

    configuration_id = config.get("configuration_id")

    assert isinstance(configuration_id, str)
    assert configuration_id

    return configuration_id


def create_configuration(
    client: TestClient,
    *,
    name: str | None = None,
    custom: bool = False,
) -> dict:
    """Create a configuration through the HTTP API."""

    payload = configuration_payload(
        name or unique_name("API Configuration"),
        custom=custom,
    )

    # The endpoint may accept the configuration
    # directly or inside a request wrapper.
    response = client.post(
        BASE,
        json=payload,
    )

    if response.status_code == 422:
        response = client.post(
            BASE,
            json={"configuration": payload},
        )

    assert response.status_code in (200, 201), response.text

    data = response.json()

    configuration_id_from(data)

    return data


def get_configuration(
    client: TestClient,
    configuration_id: str,
    *,
    version: int | None = None,
) -> dict:
    """Retrieve a configuration through the API."""

    params = (
        {"version": version}
        if version is not None
        else None
    )

    response = client.get(
        f"{BASE}/{configuration_id}",
        params=params,
    )

    assert response.status_code == 200, response.text

    return unwrap_configuration(response.json())


# ---------------------------------------------------------
# Test 1: Default configuration
# ---------------------------------------------------------

def test_default_configuration_api(
    client: TestClient,
) -> None:
    """The built-in configuration must be accessible."""

    response = client.get(f"{BASE}/default")

    assert response.status_code == 200, response.text

    config = unwrap_configuration(response.json())

    assert "dimensions" in config

    assert set(DEFAULT_DIMENSION_WEIGHTS).issubset(
        config["dimensions"]
    )

    total = sum(
        item["weight"]
        for item in config["dimensions"].values()
        if item.get("enabled", True)
    )

    assert total == pytest.approx(100.0)


# ---------------------------------------------------------
# Test 2: Create, list and retrieve
# ---------------------------------------------------------

def test_create_list_and_retrieve_configuration(
    client: TestClient,
) -> None:
    """A created configuration must be retrievable."""

    created = create_configuration(
        client,
        name=unique_name("Company Scoring"),
        custom=True,
    )

    configuration_id = configuration_id_from(created)

    saved = get_configuration(
        client,
        configuration_id,
    )

    assert saved["configuration_id"] == configuration_id

    assert (
        saved["dimensions"]["completeness"]["weight"]
        == 15.0
    )

    response = client.get(BASE)

    assert response.status_code == 200, response.text

    # Search the JSON response for the new ID.
    # This accommodates metadata-list wrappers.
    assert configuration_id in response.text


# ---------------------------------------------------------
# Test 3: Create a new version
# ---------------------------------------------------------

def test_configuration_versioning_api(
    client: TestClient,
) -> None:
    """Version 2 must not overwrite version 1."""

    created = create_configuration(client)

    configuration_id = configuration_id_from(created)

    original = get_configuration(
        client,
        configuration_id,
        version=1,
    )

    revised = configuration_payload(
        original["name"],
        custom=True,
    )

    revised["configuration_id"] = configuration_id
    revised["version"] = 2

    endpoint = f"{BASE}/{configuration_id}/versions"

    # Support direct configuration bodies and
    # wrapped version-creation request bodies.
    response = client.post(
        endpoint,
        json=revised,
    )

    if response.status_code == 422:
        response = client.post(
            endpoint,
            json={"configuration": revised},
        )

    assert response.status_code in (200, 201), response.text

    version_one = get_configuration(
        client,
        configuration_id,
        version=1,
    )

    version_two = get_configuration(
        client,
        configuration_id,
        version=2,
    )

    assert version_one["version"] == 1
    assert version_two["version"] == 2

    assert (
        version_one["dimensions"]["completeness"]["weight"]
        == 20.0
    )

    assert (
        version_two["dimensions"]["completeness"]["weight"]
        == 15.0
    )

    versions_response = client.get(endpoint)

    assert versions_response.status_code == 200


# ---------------------------------------------------------
# Test 4: Duplicate a configuration
# ---------------------------------------------------------

def test_configuration_duplication_api(
    client: TestClient,
) -> None:
    """Duplication must create an independent configuration."""

    created = create_configuration(
        client,
        custom=True,
    )

    original_id = configuration_id_from(created)

    response = client.post(
        f"{BASE}/{original_id}/duplicate",
        json={
            "name": unique_name("Duplicated Scoring"),
        },
    )

    assert response.status_code in (200, 201), response.text

    duplicate_id = configuration_id_from(
        response.json()
    )

    assert duplicate_id != original_id

    # Retrieve full configurations rather than
    # assuming duplication returns all dimensions.
    original = get_configuration(
        client,
        original_id,
    )

    duplicate = get_configuration(
        client,
        duplicate_id,
    )

    assert (
        duplicate["dimensions"]
        == original["dimensions"]
    )


# ---------------------------------------------------------
# Test 5: Archive and restore
# ---------------------------------------------------------

def test_archive_and_restore_api(
    client: TestClient,
) -> None:
    """A saved configuration must support both actions."""

    created = create_configuration(client)

    configuration_id = configuration_id_from(created)

    archive_response = client.post(
        f"{BASE}/{configuration_id}/archive"
    )

    assert archive_response.status_code == 200, (
        archive_response.text
    )

    restore_response = client.post(
        f"{BASE}/{configuration_id}/restore"
    )

    assert restore_response.status_code == 200, (
        restore_response.text
    )

    restored = get_configuration(
        client,
        configuration_id,
    )

    assert restored["configuration_id"] == configuration_id


# ---------------------------------------------------------
# Test 6: Invalid configuration weights
# ---------------------------------------------------------

def test_api_rejects_invalid_weights(
    client: TestClient,
) -> None:
    """The API must reject weights that do not total 100%."""

    payload = configuration_payload(
        unique_name("Invalid Weights"),
    )

    payload["dimensions"]["completeness"]["weight"] = 40.0

    response = client.post(
        BASE,
        json=payload,
    )

    assert response.status_code == 422
