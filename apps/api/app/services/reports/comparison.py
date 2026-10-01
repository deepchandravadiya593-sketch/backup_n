
"""NUMPA v2 - Before-and-after readiness comparison.

Compares two assessment reports without modifying them.

A score comparison is permitted only when the assessments
use the same ML task, scoring configuration, dimension
weights and measurement coverage.

Issue comparisons use stable issue identifiers, dimensions,
columns and evidence rather than message text alone.

This service does not recalculate readiness assessments.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


# Differences smaller than this tolerance are ignored
# when comparing stored floating-point dimension weights.
WEIGHT_TOLERANCE = 0.0001


def _utc_now() -> str:
    """Return an ISO-formatted UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _require_mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    """Require a dictionary-like assessment section."""

    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be a dictionary."
        )

    return value


def _dimension_map(
    assessment: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index measured dimensions by their unique names."""

    raw_dimensions = assessment.get(
        "dimensions",
        [],
    )

    if not isinstance(raw_dimensions, list):
        raise ValueError(
            "Assessment dimensions must be a list."
        )

    result: dict[str, dict[str, Any]] = {}

    for dimension in raw_dimensions:
        _require_mapping(
            dimension,
            "Dimension",
        )

        name = dimension.get("name")

        if not isinstance(name, str) or not name:
            raise ValueError(
                "Every dimension must have a name."
            )

        if name in result:
            raise ValueError(
                f"Duplicate assessment dimension: {name}"
            )

        result[name] = dimension

    return result


def _pending_names(
    assessment: dict[str, Any],
) -> set[str]:
    """Support both old and new pending-dimension formats."""

    result: set[str] = set()

    for item in assessment.get(
        "pending_dimensions",
        [],
    ):
        if isinstance(item, str):
            result.add(item)

        elif isinstance(item, dict):
            name = item.get(
                "dimension",
                item.get("name"),
            )

            if isinstance(name, str):
                result.add(name)

    return result


def _configuration_identity(
    assessment: dict[str, Any],
) -> dict[str, Any]:
    """Extract configuration identity from an assessment."""

    configuration = assessment.get(
        "configuration",
        {},
    )

    if not isinstance(configuration, dict):
        configuration = {}

    return {
        "configuration_id": configuration.get(
            "configuration_id",
            assessment.get("configuration_id"),
        ),
        "name": configuration.get(
            "name",
            assessment.get("configuration_name"),
        ),
        "version": configuration.get(
            "version",
            assessment.get("configuration_version"),
        ),
    }


def _weight_map(
    assessment: dict[str, Any],
) -> dict[str, float]:
    """Read effective weights from the assessment audit."""

    audit = assessment.get(
        "audit",
        {},
    )

    if not isinstance(audit, dict):
        audit = {}

    raw_weights = audit.get(
        "configuration_weights",
    )

    if isinstance(raw_weights, dict):
        return {
            str(name): float(weight)
            for name, weight in raw_weights.items()
        }

    # Older assessments may lack an audit weight map.
    # In that case, use their measured dimensions,
    # but the comparison will flag incomplete coverage.
    return {
        name: float(dimension["weight"])
        for name, dimension
        in _dimension_map(assessment).items()
    }


def _issue_key(
    issue: dict[str, Any],
) -> tuple[str, str, str]:
    """Identify the same issue across two assessments.

    Evidence and messages may change when cleaning
    reduces the number of affected records. They are
    intentionally excluded from issue identity.
    """

    return (
        str(issue.get("code", "")),
        str(issue.get("dimension", "")),
        str(issue.get("column") or ""),
    )


def _issue_map(
    assessment: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index issues for before-and-after comparison."""

    raw_issues = assessment.get(
        "issues",
        [],
    )

    if not isinstance(raw_issues, list):
        raise ValueError(
            "Assessment issues must be a list."
        )

    result = {}

    for issue in raw_issues:
        _require_mapping(
            issue,
            "Readiness issue",
        )

        key = _issue_key(issue)

        if not key[0]:
            raise ValueError(
                "Every readiness issue must have a code."
            )

        # Keep the first matching issue. The comparison
        # operates on issue categories rather than
        # individual affected rows.
        result.setdefault(
            key,
            deepcopy(issue),
        )

    return result


def _score(
    assessment: dict[str, Any],
) -> float | None:
    """Read a final score or provisional score."""

    final_score = assessment.get(
        "overall_score",
    )

    if final_score is not None:
        return float(final_score)

    provisional = assessment.get(
        "provisional_score",
    )

    if provisional is not None:
        return float(provisional)

    return None


def _score_kind(
    assessment: dict[str, Any],
) -> str:
    """Distinguish final from provisional measurements."""

    if assessment.get(
        "overall_score"
    ) is not None:
        return "final"

    if assessment.get(
        "provisional_score"
    ) is not None:
        return "provisional"

    return "unavailable"


def _comparability(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Determine whether the assessments are comparable."""

    reasons: list[str] = []

    if before.get("task") != after.get("task"):
        reasons.append(
            "The assessments use different ML tasks."
        )

    if before.get(
        "target_column"
    ) != after.get(
        "target_column"
    ):
        reasons.append(
            "The assessments use different target columns."
        )

    if before.get(
        "validation_strategy"
    ) != after.get(
        "validation_strategy"
    ):
        reasons.append(
            "The validation strategies differ."
        )

    before_configuration = (
        _configuration_identity(before)
    )

    after_configuration = (
        _configuration_identity(after)
    )

    if (
        before_configuration
        != after_configuration
    ):
        reasons.append(
            "The scoring configuration or its version differs."
        )

    before_weights = _weight_map(before)
    after_weights = _weight_map(after)

    if set(before_weights) != set(after_weights):
        reasons.append(
            "The enabled scoring dimensions differ."
        )

    else:
        for name in sorted(before_weights):
            if (
                abs(
                    before_weights[name]
                    - after_weights[name]
                )
                > WEIGHT_TOLERANCE
            ):
                reasons.append(
                    f"The weight of '{name}' differs."
                )

    before_dimensions = _dimension_map(before)
    after_dimensions = _dimension_map(after)

    if (
        set(before_dimensions)
        != set(after_dimensions)
    ):
        reasons.append(
            "The measured dimension coverage differs."
        )

    if (
        _pending_names(before)
        != _pending_names(after)
    ):
        reasons.append(
            "The pending dimension coverage differs."
        )

    if _score_kind(before) != _score_kind(after):
        reasons.append(
            "One assessment is final and the other is not."
        )

    if (
        _score(before) is None
        or _score(after) is None
    ):
        reasons.append(
            "At least one assessment has no calculated score."
        )

    # Preserve order and remove repeated reasons.
    reasons = list(
        dict.fromkeys(reasons)
    )

    return {
        "comparable": not reasons,
        "reasons": reasons,
        "before_configuration": before_configuration,
        "after_configuration": after_configuration,
        "score_kind": (
            _score_kind(before)
            if _score_kind(before) == _score_kind(after)
            else "mixed"
        ),
    }


def compare_assessments(
    before_assessment: dict[str, Any],
    after_assessment: dict[str, Any],
) -> dict[str, Any]:
    """Compare initial and post-cleaning readiness reports.

    Returns dimension and issue changes even when the
    overall scores cannot be compared directly.
    """

    before = deepcopy(
        _require_mapping(
            before_assessment,
            "Before assessment",
        )
    )

    after = deepcopy(
        _require_mapping(
            after_assessment,
            "After assessment",
        )
    )

    for label, assessment in (
        ("Before", before),
        ("After", after),
    ):
        if not assessment.get(
            "assessment_id"
        ):
            raise ValueError(
                f"{label} assessment has no assessment_id."
            )

        if not assessment.get(
            "dataset_id"
        ):
            raise ValueError(
                f"{label} assessment has no dataset_id."
            )

        if not assessment.get("task"):
            raise ValueError(
                f"{label} assessment has no ML task."
            )

    if (
        before["assessment_id"]
        == after["assessment_id"]
    ):
        raise ValueError(
            "Choose two different assessments."
        )

    comparability = _comparability(
        before,
        after,
    )

    before_dimensions = _dimension_map(
        before
    )

    after_dimensions = _dimension_map(
        after
    )

    dimension_changes: list[
        dict[str, Any]
    ] = []

    for name in sorted(
        set(before_dimensions)
        | set(after_dimensions)
    ):
        previous = before_dimensions.get(
            name
        )

        current = after_dimensions.get(
            name
        )

        previous_score = (
            float(previous["score"])
            if previous is not None
            else None
        )

        current_score = (
            float(current["score"])
            if current is not None
            else None
        )

        # A per-dimension delta is only valid when the
        # same dimension and weight exist on both sides.
        same_weight = (
            previous is not None
            and current is not None
            and abs(
                float(previous["weight"])
                - float(current["weight"])
            )
            <= WEIGHT_TOLERANCE
        )

        change = (
            round(
                current_score - previous_score,
                2,
            )
            if (
                same_weight
                and previous_score is not None
                and current_score is not None
                and comparability["comparable"]
            )
            else None
        )

        dimension_changes.append({
            "name": name,
            "before_score": previous_score,
            "after_score": current_score,
            "change": change,
            "before_weight": (
                previous.get("weight")
                if previous is not None
                else None
            ),
            "after_weight": (
                current.get("weight")
                if current is not None
                else None
            ),
            "directly_comparable": (
                change is not None
            ),
        })

    before_issues = _issue_map(
        before
    )

    after_issues = _issue_map(
        after
    )

    before_keys = set(
        before_issues
    )

    after_keys = set(
        after_issues
    )

    resolved_issues = [
        before_issues[key]
        for key in sorted(
            before_keys - after_keys
        )
    ]

    remaining_issues = [
        {
            "before": before_issues[key],
            "after": after_issues[key],
        }
        for key in sorted(
            before_keys & after_keys
        )
    ]

    new_issues = [
        after_issues[key]
        for key in sorted(
            after_keys - before_keys
        )
    ]

    before_score = _score(before)
    after_score = _score(after)

    score_change = (
        round(
            after_score - before_score,
            2,
        )
        if (
            comparability["comparable"]
            and before_score is not None
            and after_score is not None
        )
        else None
    )

    # Issue changes remain useful, but are not proof
    # that the dataset improved when the assessment
    # tasks or configurations are different.
    return {
        "before_assessment_id": (
            before["assessment_id"]
        ),
        "after_assessment_id": (
            after["assessment_id"]
        ),
        "before_dataset_id": (
            before["dataset_id"]
        ),
        "after_dataset_id": (
            after["dataset_id"]
        ),
        "task": (
            before["task"]
            if before["task"] == after["task"]
            else None
        ),
        "comparable": comparability[
            "comparable"
        ],
        "comparability_reasons": (
            comparability["reasons"]
        ),
        "score_kind": comparability[
            "score_kind"
        ],
        "before_configuration": (
            comparability[
                "before_configuration"
            ]
        ),
        "after_configuration": (
            comparability[
                "after_configuration"
            ]
        ),
        "before_score": before_score,
        "after_score": after_score,
        "score_change": score_change,
        "dimension_changes": dimension_changes,
        "resolved_issues": resolved_issues,
        "remaining_issues": remaining_issues,
        "new_issues": new_issues,
        "issue_summary": {
            "before_count": len(
                before_issues
            ),
            "after_count": len(
                after_issues
            ),
            "resolved_count": len(
                resolved_issues
            ),
            "remaining_count": len(
                remaining_issues
            ),
            "new_count": len(
                new_issues
            ),
        },
        "before_pending_dimensions": sorted(
            _pending_names(before)
        ),
        "after_pending_dimensions": sorted(
            _pending_names(after)
        ),
        "compared_at": _utc_now(),
        "limitations": [
            (
                "Issue resolution means a diagnostic "
                "was not reported in the second assessment; "
                "it does not independently prove that "
                "the underlying problem was eliminated."
            ),
            (
                "Provisional scores do not represent "
                "final ML readiness."
            ),
            (
                "Score changes are suppressed when "
                "assessment settings or measurement "
                "coverage are incompatible."
            ),
        ],
    }
