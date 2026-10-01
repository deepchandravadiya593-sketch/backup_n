
"""NUMPA v2 - Versioned ML readiness configuration manager.

Stores reusable scoring configurations with immutable versions.

This initial implementation uses thread-safe, in-memory storage.
Configurations disappear when the backend process restarts.

AI-generated suggestions must be approved by the user before
being saved. This service never changes weights automatically.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    ReadinessConfiguration,
    ScoringDimensionConfig,
)


# Only the seven defined scoring dimensions are supported.
KNOWN_DIMENSIONS = frozenset(DEFAULT_DIMENSION_WEIGHTS)

# Prevent accidental unbounded in-memory storage.
MAX_CONFIGURATIONS = 100
MAX_VERSIONS_PER_CONFIGURATION = 50


def _timestamp() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _validate_configuration(
    configuration: ReadinessConfiguration,
) -> ReadinessConfiguration:
    """Validate a configuration and return a defensive copy."""

    # Revalidate rather than trusting a previously constructed
    # model that might have been modified after creation.
    validated = ReadinessConfiguration.model_validate(
        configuration.model_dump(mode="python")
    )

    unknown = set(validated.dimensions) - KNOWN_DIMENSIONS

    if unknown:
        raise ValueError(
            "Unknown readiness dimensions: "
            + ", ".join(sorted(unknown))
        )

    # Every supported dimension must be explicitly present.
    # Disabling a dimension is allowed, but silently omitting
    # it would make later assessment comparisons ambiguous.
    missing = KNOWN_DIMENSIONS - set(validated.dimensions)

    if missing:
        raise ValueError(
            "Missing readiness dimensions: "
            + ", ".join(sorted(missing))
        )

    # The schema validates that enabled weights total 100.
    # Reject unsupported threshold overrides until the
    # scoring engine implements them.
    for name, settings in validated.dimensions.items():
        if settings.thresholds:
            raise ValueError(
                f"Custom thresholds for '{name}' are not "
                "supported by the current scoring engine."
            )

    # Arbitrary parameters are not yet consumed by the engine.
    if validated.parameters:
        raise ValueError(
            "Custom scoring parameters are not yet supported."
        )

    return validated.model_copy(deep=True)


def _default_configuration() -> ReadinessConfiguration:
    """Create an independent copy of NUMPA's default profile."""

    return ReadinessConfiguration(
        name="NUMPA Default",
        dimensions={
            name: ScoringDimensionConfig(
                weight=weight,
                enabled=True,
            )
            for name, weight in DEFAULT_DIMENSION_WEIGHTS.items()
        },
    )


class ConfigurationNotFoundError(LookupError):
    """Raised when a configuration or version does not exist."""


class ConfigurationLimitError(ValueError):
    """Raised when an in-memory storage limit is reached."""


class ConfigurationManager:
    """Thread-safe manager for versioned scoring configurations.

    Each configuration has a stable ID and one or more versions.
    Updating creates a new version; old versions are not edited.

    Returned dictionaries and Pydantic models are defensive
    copies so callers cannot mutate stored history.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, dict[str, Any]] = {}

    def create(
        self,
        configuration: ReadinessConfiguration,
        *,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Save a new configuration as version 1."""

        validated = _validate_configuration(configuration)

        # New records always receive server-generated IDs
        # and start at version 1.
        if validated.configuration_id is not None:
            raise ValueError(
                "Do not provide configuration_id when creating "
                "a new configuration."
            )

        if validated.version != 1:
            raise ValueError(
                "New configurations must start at version 1."
            )

        with self._lock:
            if len(self._records) >= MAX_CONFIGURATIONS:
                raise ConfigurationLimitError(
                    "The configuration storage limit has been reached."
                )

            configuration_id = str(uuid4())
            now = _timestamp()

            validated.configuration_id = configuration_id
            validated.version = 1

            self._records[configuration_id] = {
                "configuration_id": configuration_id,
                "name": validated.name,
                "created_by": created_by,
                "created_at": now,
                "updated_at": now,
                "archived": False,
                "latest_version": 1,
                "versions": {
                    1: validated.model_copy(deep=True),
                },
            }

            return self._serialize_record(
                self._records[configuration_id],
                include_configuration=True,
            )

    def get(
        self,
        configuration_id: str,
        *,
        version: int | None = None,
    ) -> ReadinessConfiguration:
        """Retrieve the latest or a specific immutable version."""

        with self._lock:
            record = self._get_record(configuration_id)

            selected_version = (
                record["latest_version"]
                if version is None
                else version
            )

            if (
                isinstance(selected_version, bool)
                or not isinstance(selected_version, int)
                or selected_version not in record["versions"]
            ):
                raise ConfigurationNotFoundError(
                    f"Configuration version not found: "
                    f"{configuration_id} / {selected_version}"
                )

            return record["versions"][
                selected_version
            ].model_copy(deep=True)

    def create_version(
        self,
        configuration_id: str,
        configuration: ReadinessConfiguration,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """Create a new version without modifying prior versions.

        expected_version supports optimistic concurrency:
        callers can prevent overwriting a newer update they
        have not yet reviewed.
        """

        validated = _validate_configuration(configuration)

        with self._lock:
            record = self._get_record(configuration_id)

            if record["archived"]:
                raise ValueError(
                    "Archived configurations cannot be updated."
                )

            latest = record["latest_version"]

            if (
                expected_version is not None
                and expected_version != latest
            ):
                raise ValueError(
                    "Configuration version conflict. "
                    f"Expected {expected_version}, current {latest}."
                )

            if latest >= MAX_VERSIONS_PER_CONFIGURATION:
                raise ConfigurationLimitError(
                    "The version limit has been reached."
                )

            # The caller must identify the existing record
            # or leave the ID unspecified.
            if validated.configuration_id not in (
                None,
                configuration_id,
            ):
                raise ValueError(
                    "The supplied configuration ID does not match."
                )

            new_version = latest + 1

            validated.configuration_id = configuration_id
            validated.version = new_version

            record["versions"][new_version] = (
                validated.model_copy(deep=True)
            )

            record["latest_version"] = new_version
            record["name"] = validated.name
            record["updated_at"] = _timestamp()

            return self._serialize_record(
                record,
                include_configuration=True,
            )

    def list_configurations(
        self,
        *,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """List configuration metadata without version payloads."""

        with self._lock:
            records = [
                self._serialize_record(record)
                for record in self._records.values()
                if include_archived or not record["archived"]
            ]

        return sorted(
            records,
            key=lambda item: (
                item["name"].lower(),
                item["configuration_id"],
            ),
        )

    def list_versions(
        self,
        configuration_id: str,
    ) -> list[dict[str, Any]]:
        """Return every saved version, newest first."""

        with self._lock:
            record = self._get_record(configuration_id)

            return [
                record["versions"][version].model_dump(
                    mode="json"
                )
                for version in sorted(
                    record["versions"],
                    reverse=True,
                )
            ]

    def archive(
        self,
        configuration_id: str,
    ) -> dict[str, Any]:
        """Archive a configuration while preserving its history."""

        with self._lock:
            record = self._get_record(configuration_id)

            record["archived"] = True
            record["updated_at"] = _timestamp()

            return self._serialize_record(record)

    def restore(
        self,
        configuration_id: str,
    ) -> dict[str, Any]:
        """Restore an archived configuration."""

        with self._lock:
            record = self._get_record(configuration_id)

            record["archived"] = False
            record["updated_at"] = _timestamp()

            return self._serialize_record(record)

    def duplicate(
        self,
        configuration_id: str,
        *,
        name: str,
        version: int | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Copy an existing version into a new configuration."""

        if not name or not name.strip():
            raise ValueError(
                "A name is required for the duplicated configuration."
            )

        original = self.get(
            configuration_id,
            version=version,
        )

        original.configuration_id = None
        original.version = 1
        original.name = name.strip()

        return self.create(
            original,
            created_by=created_by,
        )

    def get_default(self) -> ReadinessConfiguration:
        """Return the built-in default without saving it."""

        return _default_configuration()

    def count(self) -> int:
        """Return the number of stored configuration records."""

        with self._lock:
            return len(self._records)

    def _get_record(
        self,
        configuration_id: str,
    ) -> dict[str, Any]:
        """Find a record while the caller holds the lock."""

        if not configuration_id:
            raise ValueError(
                "configuration_id is required."
            )

        record = self._records.get(configuration_id)

        if record is None:
            raise ConfigurationNotFoundError(
                f"Configuration not found: {configuration_id}"
            )

        return record

    @staticmethod
    def _serialize_record(
        record: dict[str, Any],
        *,
        include_configuration: bool = False,
    ) -> dict[str, Any]:
        """Return JSON-compatible metadata and optional latest version."""

        result = {
            "configuration_id": record["configuration_id"],
            "name": record["name"],
            "created_by": record["created_by"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "archived": record["archived"],
            "latest_version": record["latest_version"],
            "version_count": len(record["versions"]),
        }

        if include_configuration:
            latest = record["latest_version"]

            result["configuration"] = record["versions"][
                latest
            ].model_dump(mode="json")

        return deepcopy(result)


# Shared in-memory manager for future API endpoints.
configuration_manager = ConfigurationManager()
