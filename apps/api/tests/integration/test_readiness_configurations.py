
"""NUMPA v2 - Readiness configuration and versioning tests.

Tests:
- Retrieving the built-in configuration.
- Creating a custom configuration.
- Rejecting invalid dimension weights.
- Saving a new configuration version.
- Preserving earlier versions.
- Duplicating a configuration.
- Archiving and restoring configurations.

The tests use unique configuration names to avoid
collisions with records from other test runs.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app

from app.schemas.readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    ReadinessConfiguration,
    ScoringDimensionConfig,
)

from app.services.readiness.configurations import (
    ConfigurationManager,
)


# ---------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------

@pytest.fixture
def client():
    """Create a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def manager() -> ConfigurationManager:
    """Use an independent manager for storage-level tests."""

    return ConfigurationManager()


def unique_name(prefix: str) -> str:
    """Generate a recognizable, unique configuration name."""

    return f"{prefix}-{uuid4().hex[:8]}"


def default_dimensions() -> dict:
    """Return independent copies of default dimension settings."""

    return {
        name: {
            "weight": weight,
            "enabled": True,
            "thresholds": {},
        }
        for name, weight in DEFAULT_DIMENSION_WEIGHTS.items()
    }


def custom_dimensions() -> dict:
    """Create a valid custom weight distribution."""

    dimensions = default_dimensions()

    # Move five percentage points from completeness
    # to leakage/validation without changing the total.
    dimensions["completeness"]["weight"] = 15.0

    dimensions["leakage_validation"]["weight"] = 20.0

    return dimensions


def assert_valid_weights(dimensions: dict) -> None:
    """Verify that enabled dimension weights total 100%."""

    total = sum(
        dimension["weight"]
        for dimension in dimensions.values()
        if dimension.get("enabled", True)
    )

    assert total == pytest.approx(100.0)


# ---------------------------------------------------------
# Test 1: Built-in configuration
# ---------------------------------------------------------

def test_default_configuration_endpoint(
    client: TestClient,
) -> None:
    """The API must expose NUMPA's built-in configuration."""

    response = client.get(
        "/api/v1/readiness/configurations/default"
    )

    assert response.status_code == 200, response.text

    configuration = response.json()

    # Some endpoints wrap the configuration in
    # a response object; support either representation.
    if "configuration" in configuration:
        configuration = configuration["configuration"]

    assert "dimensions" in configuration

    assert_valid_weights(
        configuration["dimensions"]
    )

    assert set(
        DEFAULT_DIMENSION_WEIGHTS
    ).issubset(
        configuration["dimensions"]
    )


# ---------------------------------------------------------
# Test 2: Valid custom configuration
# ---------------------------------------------------------

def test_custom_configuration_schema() -> None:
    """A valid custom weight distribution must be accepted."""

    configuration = ReadinessConfiguration(
        name=unique_name("Custom Scoring"),
        dimensions={
            name: ScoringDimensionConfig(**settings)
            for name, settings in custom_dimensions().items()
        },
    )

    assert configuration.version == 1

    assert (
        configuration.dimensions[
            "completeness"
        ].weight
        == 15.0
    )

    assert (
        configuration.dimensions[
            "leakage_validation"
        ].weight
        == 20.0
    )


# ---------------------------------------------------------
# Test 3: Invalid weights
# ---------------------------------------------------------

def test_invalid_weights_are_rejected() -> None:
    """Enabled dimension weights must total 100%."""

    dimensions = default_dimensions()

    dimensions["completeness"]["weight"] = 30.0

    with pytest.raises(ValidationError):
        ReadinessConfiguration(
            name="Invalid Weights",
            dimensions={
                name: ScoringDimensionConfig(**settings)
                for name, settings in dimensions.items()
            },
        )


# ---------------------------------------------------------
# Test 4: Create and retrieve configuration
# ---------------------------------------------------------

def test_manager_creates_configuration(
    manager: ConfigurationManager,
) -> None:
    """A saved configuration must be retrievable."""

    configuration = ReadinessConfiguration(
        name=unique_name("Company Model"),
        dimensions={
            name: ScoringDimensionConfig(**settings)
            for name, settings in custom_dimensions().items()
        },
    )

    created = manager.create(configuration)

    # Work with the saved configuration ID,
    # not the unsaved input model.
    configuration_id = (
        created["configuration_id"]
        if isinstance(created, dict)
        else created.configuration_id
    )

    assert configuration_id

    retrieved = manager.get(
        configuration_id
    )

    if isinstance(retrieved, dict):
        assert retrieved["configuration_id"] == configuration_id
    else:
        assert retrieved.configuration_id == configuration_id


# ---------------------------------------------------------
# Test 5: New version preserves previous version
# ---------------------------------------------------------

def test_new_version_preserves_original(
    manager: ConfigurationManager,
) -> None:
    """Creating version 2 must not modify version 1."""

    initial = ReadinessConfiguration(
        name=unique_name("Versioned Model"),
    )

    created = manager.create(initial)

    # Normalize the manager result to a dictionary.
    first = (
        created
        if isinstance(created, dict)
        else created.model_dump(mode="json")
    )

    configuration_id = first["configuration_id"]

    # Retrieve version 1 before making changes.
    original = manager.get(
        configuration_id,
        version=1,
    )

    original_data = (
        original
        if isinstance(original, dict)
        else original.model_dump(mode="json")
    )

    # Create a revised configuration with valid weights.
    revised = ReadinessConfiguration(
        configuration_id=configuration_id,
        name=initial.name,
        version=2,
        dimensions={
            name: ScoringDimensionConfig(**settings)
            for name, settings in custom_dimensions().items()
        },
    )

    manager.create_version(
        configuration_id,
        revised,
    )

    # Version 1 must retain its original weight.
    saved_v1 = manager.get(
        configuration_id,
        version=1,
    )

    saved_v2 = manager.get(
        configuration_id,
        version=2,
    )

    v1 = (
        saved_v1
        if isinstance(saved_v1, dict)
        else saved_v1.model_dump(mode="json")
    )

    v2 = (
        saved_v2
        if isinstance(saved_v2, dict)
        else saved_v2.model_dump(mode="json")
    )

    assert v1["version"] == 1
    assert v2["version"] == 2

    assert (
        v1["dimensions"]["completeness"]["weight"]
        == original_data["dimensions"][
            "completeness"
        ]["weight"]
    )

    assert (
        v2["dimensions"]["completeness"]["weight"]
        == 15.0
    )


# ---------------------------------------------------------
# Test 6: Configuration duplication
# ---------------------------------------------------------

def test_manager_duplicates_configuration(
    manager: ConfigurationManager,
) -> None:
    """Duplicating a configuration must preserve its dimensions."""

    # Create the original configuration.
    original = manager.create(
        ReadinessConfiguration(
            name=unique_name("Original Model"),
        )
    )

    original_data = (
        original
        if isinstance(original, dict)
        else original.model_dump(mode="json")
    )

    original_id = original_data["configuration_id"]

    # Duplicate the saved configuration.
    duplicate = manager.duplicate(
        original_id,
        name=unique_name("Duplicated Model"),
    )

    duplicate_data = (
        duplicate
        if isinstance(duplicate, dict)
        else duplicate.model_dump(mode="json")
    )

    duplicate_id = duplicate_data["configuration_id"]

    # The duplicate must have its own identity.
    assert duplicate_id != original_id

    # Retrieve both full configurations. The duplicate()
    # response may contain metadata rather than dimensions.
    saved_original = manager.get(original_id)
    saved_duplicate = manager.get(duplicate_id)

    original_config = (
        saved_original
        if isinstance(saved_original, dict)
        else saved_original.model_dump(mode="json")
    )

    duplicate_config = (
        saved_duplicate
        if isinstance(saved_duplicate, dict)
        else saved_duplicate.model_dump(mode="json")
    )

    # The copied configuration must preserve scoring settings.
    assert (
        duplicate_config["dimensions"]
        == original_config["dimensions"]
    )


# ---------------------------------------------------------
# Test 7: Archive and restore
# ---------------------------------------------------------

def test_manager_archive_and_restore(
    manager: ConfigurationManager,
) -> None:
    """Archived configurations must be restorable."""

    created = manager.create(
        ReadinessConfiguration(
            name=unique_name("Archive Test"),
        )
    )

    created_data = (
        created
        if isinstance(created, dict)
        else created.model_dump(mode="json")
    )

    configuration_id = created_data["configuration_id"]

    # Archive the configuration.
    manager.archive(configuration_id)

    # Restoring should make it available again.
    manager.restore(configuration_id)

    restored = manager.get(
        configuration_id
    )

    restored_data = (
        restored
        if isinstance(restored, dict)
        else restored.model_dump(mode="json")
    )

    assert (
        restored_data["configuration_id"]
        == configuration_id
    )


# ---------------------------------------------------------
# Test 8: Reject unknown dimensions
# ---------------------------------------------------------

def test_manager_rejects_unknown_dimension(
    manager: ConfigurationManager,
) -> None:
    """Only implemented scoring dimensions may be saved."""

    dimensions = default_dimensions()

    # A zero-weight dimension preserves the total
    # while testing the unknown-name restriction.
    dimensions["imaginary_dimension"] = {
        "weight": 0.0,
        "enabled": True,
        "thresholds": {},
    }

    configuration = ReadinessConfiguration(
        name=unique_name("Unknown Dimension"),
        dimensions={
            name: ScoringDimensionConfig(**settings)
            for name, settings in dimensions.items()
        },
    )

    with pytest.raises(ValueError):
        manager.create(configuration)
