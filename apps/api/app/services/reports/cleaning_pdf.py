
"""NUMPA v2 - Cleaning execution PDF report.

Generates an auditable PDF from an existing cleaning
execution result. It does not rerun cleaning operations,
modify datasets or invent missing execution statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------
# Report appearance
# ---------------------------------------------------------

NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#2563EB")
SLATE = colors.HexColor("#475569")
PALE_BLUE = colors.HexColor("#EFF6FF")
BORDER = colors.HexColor("#CBD5E1")

PAGE_WIDTH, _ = A4
CONTENT_WIDTH = PAGE_WIDTH - 36 * mm

MAX_STEPS = 100
MAX_DETAILS_PER_STEP = 20


def _safe_text(
    value: Any,
    *,
    limit: int = 250,
) -> str:
    """Escape text before inserting it into a PDF."""

    if value is None:
        return "Not recorded"

    if isinstance(value, bool):
        text = "Yes" if value else "No"
    elif isinstance(value, float):
        text = f"{value:,.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)

    text = " ".join(text.split())

    if len(text) > limit:
        text = text[: limit - 3] + "..."

    return escape(text)


def _styles() -> dict[str, ParagraphStyle]:
    """Define the report's typography."""

    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "CleaningTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=NAVY,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "CleaningHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=7,
        ),
        "subheading": ParagraphStyle(
            "CleaningSubheading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "CleaningBody",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "CleaningSmall",
            parent=base["Normal"],
            fontSize=7.5,
            leading=11,
            textColor=SLATE,
        ),
        "cell": ParagraphStyle(
            "CleaningCell",
            parent=base["Normal"],
            fontSize=7.5,
            leading=10,
            wordWrap="CJK",
        ),
        "header": ParagraphStyle(
            "CleaningHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.white,
        ),
    }


def _paragraph(
    value: Any,
    style: ParagraphStyle,
    *,
    limit: int = 250,
) -> Paragraph:
    """Create a paragraph containing safely escaped text."""

    return Paragraph(
        _safe_text(value, limit=limit),
        style,
    )


def _table(
    headers: list[str],
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    widths: list[float],
) -> Table:
    """Create a styled table with repeating headers."""

    data = [
        [
            _paragraph(header, styles["header"])
            for header in headers
        ]
    ]

    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                "Table row does not match its headers."
            )

        data.append(
            [
                _paragraph(value, styles["cell"])
                for value in row
            ]
        )

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, PALE_BLUE],
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    BORDER,
                ),
            ]
        )
    )

    return table


def _footer(
    canvas: Any,
    document: Any,
) -> None:
    """Add a footer and page number."""

    canvas.saveState()

    canvas.setStrokeColor(BORDER)
    canvas.line(
        18 * mm,
        15 * mm,
        PAGE_WIDTH - 18 * mm,
        15 * mm,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)

    canvas.drawString(
        18 * mm,
        10 * mm,
        "NUMPA | Cleaning Execution Report",
    )

    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


# ---------------------------------------------------------
# Execution report helpers
# ---------------------------------------------------------

def _as_mapping(
    value: Any,
) -> dict[str, Any]:
    """Return a dictionary or an empty mapping."""

    return value if isinstance(value, dict) else {}


def _first_recorded(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    """Return the first present, non-null field."""

    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return None


def _format_detail(
    value: Any,
) -> str:
    """Format metadata without dumping entire nested objects."""

    if isinstance(value, dict):
        return (
            f"{len(value)} recorded fields"
        )

    if isinstance(value, list):
        return (
            f"{len(value)} recorded items"
        )

    return str(value) if value is not None else "Not recorded"


def _summary_rows(
    result: dict[str, Any],
) -> list[list[Any]]:
    """Extract documented execution metadata."""

    return [
        [
            "Execution ID",
            result.get("execution_id"),
        ],
        [
            "Execution status",
            result.get("status"),
        ],
        [
            "Execution mode",
            result.get("mode"),
        ],
        [
            "Pipeline",
            result.get("pipeline_name"),
        ],
        [
            "Original dataset ID",
            _first_recorded(
                result,
                "dataset_id",
                "original_dataset_id",
                "source_dataset_id",
            ),
        ],
        [
            "Cleaned dataset ID",
            result.get("cleaned_dataset_id"),
        ],
        [
            "Executed at",
            result.get("executed_at"),
        ],
    ]


def _snapshot_rows(
    snapshot: dict[str, Any],
) -> list[list[Any]]:
    """Extract recorded before/after dataset metrics."""

    # These aliases accommodate reports from different
    # versions of the cleaning engine.
    fields = [
        (
            "Rows",
            ("rows", "row_count", "total_rows"),
        ),
        (
            "Columns",
            ("columns", "column_count", "total_columns"),
        ),
        (
            "Missing cells",
            ("missing_cells", "missing_values", "total_missing"),
        ),
        (
            "Duplicate rows",
            ("duplicate_rows", "duplicates"),
        ),
        (
            "Memory usage",
            ("memory_usage", "memory_usage_bytes"),
        ),
    ]

    rows: list[list[Any]] = []

    for label, keys in fields:
        value = _first_recorded(
            snapshot,
            *keys,
        )

        if value is not None:
            rows.append(
                [label, value]
            )

    return rows


def _step_records(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find the executed pipeline steps.

    Only documented step records are rendered. The
    function does not infer steps from configuration
    names or fabricate operation outcomes.
    """

    report = result.get("pipeline_report")

    if isinstance(report, list):
        candidates = report

    elif isinstance(report, dict):
        candidates = _first_recorded(
            report,
            "steps",
            "step_reports",
            "operations",
            "results",
        )

    else:
        candidates = None

    if not isinstance(candidates, list):
        return []

    return [
        step
        for step in candidates
        if isinstance(step, dict)
    ]


def _step_name(
    step: dict[str, Any],
    index: int,
) -> str:
    """Get the operation name from a recorded step."""

    value = _first_recorded(
        step,
        "operation",
        "operation_name",
        "name",
        "step_type",
        "type",
    )

    return (
        str(value)
        if value is not None
        else f"Step {index}"
    )


def _step_details(
    step: dict[str, Any],
) -> list[list[Any]]:
    """Prepare a bounded metadata table for one step."""

    rows: list[list[Any]] = []

    # Skip potentially large artifacts and repeated
    # identifiers. A PDF should describe execution,
    # not embed fitted transformers or dataset contents.
    excluded = {
        "operation",
        "operation_name",
        "name",
        "step_type",
        "type",
        "fitted_artifacts",
        "artifacts",
        "dataframe",
        "data",
        "preview",
        "sample_rows",
    }

    for key, value in step.items():
        if key in excluded:
            continue

        if len(rows) >= MAX_DETAILS_PER_STEP:
            rows.append(
                [
                    "Additional details",
                    "Omitted to keep the report concise",
                ]
            )
            break

        rows.append(
            [
                key.replace("_", " ").title(),
                _format_detail(value),
            ]
        )

    return rows


def _change_rows(
    result: dict[str, Any],
) -> list[list[Any]]:
    """Display only changes recorded by the engine."""

    changes = result.get("changes")

    if isinstance(changes, dict):
        return [
            [
                key.replace("_", " ").title(),
                _format_detail(value),
            ]
            for key, value in changes.items()
            if key not in {
                "data",
                "preview",
                "fitted_artifacts",
            }
        ][:30]

    if isinstance(changes, list):
        return [
            [
                f"Change {index}",
                _format_detail(item),
            ]
            for index, item in enumerate(
                changes[:30],
                start=1,
            )
        ]

    return []


def _warnings(
    result: dict[str, Any],
) -> list[str]:
    """Collect explicitly recorded execution warnings."""

    warnings: list[str] = []

    sources = [
        result,
        _as_mapping(result.get("pipeline_report")),
    ]

    for source in sources:
        for key in ("warnings", "errors"):
            items = source.get(key, [])

            if isinstance(items, str):
                items = [items]

            if isinstance(items, list):
                warnings.extend(
                    str(item)
                    for item in items
                )

    return list(
        dict.fromkeys(warnings)
    )[:30]


# ---------------------------------------------------------
# Public PDF generator
# ---------------------------------------------------------

def generate_cleaning_report_pdf(
    execution_result: dict[str, Any],
) -> bytes:
    """Generate a PDF from a recorded cleaning execution.

    Args:
        execution_result:
            The result returned by NUMPA's cleaning
            execution service or API.

    Returns:
        The generated PDF as bytes.

    Raises:
        TypeError:
            The supplied result is not a dictionary.
        ValueError:
            Required execution metadata is missing.
    """

    if not isinstance(execution_result, dict):
        raise TypeError(
            "Cleaning execution result must be a dictionary."
        )

    if not execution_result.get("execution_id"):
        raise ValueError(
            "Cleaning execution ID is required."
        )

    if not execution_result.get("status"):
        raise ValueError(
            "Cleaning execution status is required."
        )

    styles = _styles()
    story: list[Any] = []

    generated_at = datetime.now(
        timezone.utc
    ).strftime("%d %B %Y, %H:%M UTC")

    # Report title.
    story.append(
        Paragraph(
            "Data Cleaning Execution Report",
            styles["title"],
        )
    )

    story.append(
        _paragraph(
            f"Report generated: {generated_at}",
            styles["small"],
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # Section 1: Execution details.
    story.append(
        Paragraph(
            "1. Execution Details",
            styles["heading"],
        )
    )

    story.append(
        _table(
            ["Field", "Recorded value"],
            _summary_rows(execution_result),
            styles,
            [
                60 * mm,
                114 * mm,
            ],
        )
    )

    # Section 2: Before and after.
    story.append(
        Paragraph(
            "2. Dataset Before and After",
            styles["heading"],
        )
    )

    before = _as_mapping(
        execution_result.get("before")
    )

    after = _as_mapping(
        execution_result.get("after")
    )

    before_metrics = dict(
        _snapshot_rows(before)
    )

    after_metrics = dict(
        _snapshot_rows(after)
    )

    metric_names = list(
        dict.fromkeys(
            [
                *before_metrics.keys(),
                *after_metrics.keys(),
            ]
        )
    )

    if metric_names:
        comparison_rows = [
            [
                name,
                before_metrics.get(name),
                after_metrics.get(name),
            ]
            for name in metric_names
        ]

        story.append(
            _table(
                [
                    "Metric",
                    "Before",
                    "After",
                ],
                comparison_rows,
                styles,
                [
                    70 * mm,
                    52 * mm,
                    52 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                (
                    "The execution result does not contain "
                    "recognized before-and-after metrics."
                ),
                styles["body"],
            )
        )

    # Section 3: Individual cleaning operations.
    story.append(
        Paragraph(
            "3. Executed Pipeline Steps",
            styles["heading"],
        )
    )

    steps = _step_records(
        execution_result
    )

    if steps:
        for index, step in enumerate(
            steps[:MAX_STEPS],
            start=1,
        ):
            story.append(
                _paragraph(
                    f"Step {index}: {_step_name(step, index)}",
                    styles["subheading"],
                )
            )

            details = _step_details(
                step
            )

            if details:
                story.append(
                    _table(
                        ["Property", "Recorded result"],
                        details,
                        styles,
                        [
                            64 * mm,
                            110 * mm,
                        ],
                    )
                )

            else:
                story.append(
                    _paragraph(
                        "No additional step details were recorded.",
                        styles["body"],
                    )
                )

        if len(steps) > MAX_STEPS:
            story.append(
                _paragraph(
                    (
                        f"{len(steps) - MAX_STEPS} additional "
                        "steps were omitted from this PDF."
                    ),
                    styles["small"],
                )
            )

    else:
        story.append(
            _paragraph(
                (
                    "No individual step records were supplied "
                    "in the execution result."
                ),
                styles["body"],
            )
        )

    # Section 4: Summary of changes.
    story.append(
        Paragraph(
            "4. Recorded Changes",
            styles["heading"],
        )
    )

    change_rows = _change_rows(
        execution_result
    )

    if change_rows:
        story.append(
            _table(
                ["Change", "Recorded result"],
                change_rows,
                styles,
                [
                    70 * mm,
                    104 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                (
                    "No separate change summary was recorded. "
                    "See the individual pipeline steps above."
                ),
                styles["body"],
            )
        )

    # Section 5: Warnings and errors.
    story.append(
        Paragraph(
            "5. Execution Warnings",
            styles["heading"],
        )
    )

    warnings = _warnings(
        execution_result
    )

    if warnings:
        for index, warning in enumerate(
            warnings,
            start=1,
        ):
            story.append(
                _paragraph(
                    f"{index}. {warning}",
                    styles["body"],
                )
            )

    else:
        story.append(
            _paragraph(
                (
                    "No warnings or errors were included "
                    "in the supplied execution report."
                ),
                styles["body"],
            )
        )

    # Section 6: Important audit notes.
    story.append(
        Paragraph(
            "6. Report Notes",
            styles["heading"],
        )
    )

    story.append(
        _paragraph(
            (
                "This report documents the results recorded "
                "by the cleaning engine. Missing statistics "
                "are shown as unavailable rather than "
                "estimated. It does not independently "
                "verify the cleaning operation."
            ),
            styles["body"],
        )
    )

    story.append(
        _paragraph(
            (
                "Cleaning a dataset does not automatically "
                "make it suitable for machine learning. "
                "Run a new task-specific ML readiness "
                "assessment to evaluate the cleaned dataset."
            ),
            styles["body"],
        )
    )

    # Generate the PDF in memory.
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="NUMPA Cleaning Execution Report",
        author="NUMPA",
    )

    document.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer,
    )

    pdf_bytes = buffer.getvalue()
    buffer.close()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError(
            "PDF generation failed."
        )

    return pdf_bytes
