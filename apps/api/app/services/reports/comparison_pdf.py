
"""NUMPA v2 - ML readiness comparison PDF.

Generate a PDF from the existing comparison service's
output. Never calculate a score difference when the
comparison service marks assessments as incompatible.
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
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.services.reports.comparison import (
    compare_assessments,
)


# ---------------------------------------------------------
# PDF appearance
# ---------------------------------------------------------

NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#2563EB")
SLATE = colors.HexColor("#475569")
PALE_BLUE = colors.HexColor("#EFF6FF")
BORDER = colors.HexColor("#CBD5E1")
AMBER = colors.HexColor("#92400E")

PAGE_WIDTH, _ = A4
CONTENT_WIDTH = PAGE_WIDTH - 36 * mm


def _safe_text(
    value: Any,
    *,
    limit: int = 300,
) -> str:
    """Escape text and limit excessively long values."""

    if value is None:
        return "Not available"

    if isinstance(value, bool):
        text = "Yes" if value else "No"
    else:
        text = str(value)

    text = " ".join(text.split())

    if len(text) > limit:
        text = text[: limit - 3] + "..."

    return escape(text)


def _styles() -> dict[str, ParagraphStyle]:
    """Define report typography."""

    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ComparisonTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=NAVY,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "ComparisonHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "ComparisonBody",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ComparisonSmall",
            parent=base["Normal"],
            fontSize=7.5,
            leading=11,
            textColor=SLATE,
        ),
        "warning": ParagraphStyle(
            "ComparisonWarning",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
            textColor=AMBER,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "ComparisonCell",
            parent=base["Normal"],
            fontSize=7.5,
            leading=10,
            wordWrap="CJK",
        ),
        "header": ParagraphStyle(
            "ComparisonHeader",
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
) -> Paragraph:
    """Create a safe ReportLab paragraph."""

    return Paragraph(
        _safe_text(value),
        style,
    )


def _table(
    headers: list[str],
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    widths: list[float],
) -> Table:
    """Create a table with repeating column headers."""

    data = [
        [
            _paragraph(header, styles["header"])
            for header in headers
        ]
    ]

    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                "Table row does not match header count."
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
    """Draw the report footer."""

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
        "NUMPA | ML Readiness Comparison",
    )

    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


# ---------------------------------------------------------
# Comparison formatting
# ---------------------------------------------------------

def _score_text(
    score: Any,
) -> str:
    """Format an existing score without calculating one."""

    if score is None:
        return "Not available"

    if isinstance(score, bool):
        return "Invalid score"

    if isinstance(score, (int, float)):
        return f"{score:.2f}/100"

    return str(score)


def _change_text(
    change: Any,
) -> str:
    """Format an already-calculated score difference."""

    if change is None:
        return "Not comparable"

    if isinstance(change, bool):
        return "Invalid change"

    if isinstance(change, (int, float)):
        return f"{change:+.2f} points"

    return str(change)


def _issue_description(
    issue: dict[str, Any],
) -> str:
    """Format one recorded issue."""

    severity = issue.get(
        "severity",
        "unspecified",
    )

    message = issue.get(
        "message",
        issue.get("code", "Issue"),
    )

    column = issue.get("column")

    if column:
        return (
            f"[{severity}] {message} "
            f"(column: {column})"
        )

    return f"[{severity}] {message}"


def _issue_rows(
    issues: list[Any],
) -> list[list[str]]:
    """Convert issue records into table rows."""

    rows: list[list[str]] = []

    for issue in issues:
        if not isinstance(issue, dict):
            continue

        rows.append(
            [
                issue.get("dimension", "Dataset"),
                issue.get("severity", "Unspecified"),
                _issue_description(issue),
            ]
        )

    return rows


def _dimension_rows(
    comparison: dict[str, Any],
) -> list[list[str]]:
    """Prepare dimension changes for the report."""

    rows: list[list[str]] = []

    for item in comparison.get(
        "dimension_changes",
        [],
    ):
        if not isinstance(item, dict):
            continue

        directly_comparable = (
            comparison.get("comparable") is True
            and item.get("directly_comparable") is True
        )

        rows.append(
            [
                item.get("name", "Unknown"),
                _score_text(
                    item.get("before_score")
                ),
                _score_text(
                    item.get("after_score")
                ),
                (
                    _change_text(item.get("change"))
                    if directly_comparable
                    else "Not comparable"
                ),
            ]
        )

    return rows


def _configuration_text(
    value: Any,
) -> str:
    """Describe a recorded scoring configuration."""

    if not isinstance(value, dict):
        return "Not recorded"

    name = value.get(
        "name",
        value.get(
            "configuration_name",
            value.get(
                "configuration_id",
                "Unnamed configuration",
            ),
        ),
    )

    version = value.get(
        "version",
        value.get("configuration_version"),
    )

    if version is not None:
        return f"{name} (version {version})"

    return str(name)


# ---------------------------------------------------------
# Public PDF generator
# ---------------------------------------------------------

def generate_readiness_comparison_pdf(
    comparison: dict[str, Any],
) -> bytes:
    """Generate a PDF from a completed comparison result.

    The comparison must already have been evaluated by
    compare_assessments(). This function only renders it.
    """

    if not isinstance(comparison, dict):
        raise TypeError(
            "Comparison result must be a dictionary."
        )

    required = (
        "before_assessment_id",
        "after_assessment_id",
        "comparable",
    )

    missing = [
        field
        for field in required
        if field not in comparison
        or comparison[field] is None
    ]

    if missing:
        raise ValueError(
            "Missing comparison fields: "
            + ", ".join(missing)
        )

    if not isinstance(
        comparison["comparable"],
        bool,
    ):
        raise ValueError(
            "The comparable field must be a boolean."
        )

    # Prevent an incompatible comparison from
    # accidentally publishing a score difference.
    comparable = comparison["comparable"]

    before_score = comparison.get(
        "before_score"
    )

    after_score = comparison.get(
        "after_score"
    )

    score_change = (
        comparison.get("score_change")
        if comparable
        else None
    )

    styles = _styles()
    story: list[Any] = []

    generated_at = datetime.now(
        timezone.utc
    ).strftime("%d %B %Y, %H:%M UTC")

    # Report title.
    story.append(
        Paragraph(
            "ML Readiness Before/After Comparison",
            styles["title"],
        )
    )

    story.append(
        _paragraph(
            f"Generated: {generated_at}",
            styles["small"],
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # Section 1: Assessment identity.
    story.append(
        Paragraph(
            "1. Assessment Details",
            styles["heading"],
        )
    )

    story.append(
        _table(
            ["Field", "Before", "After"],
            [
                [
                    "Assessment ID",
                    comparison["before_assessment_id"],
                    comparison["after_assessment_id"],
                ],
                [
                    "Dataset ID",
                    comparison.get(
                        "before_dataset_id"
                    ),
                    comparison.get(
                        "after_dataset_id"
                    ),
                ],
                [
                    "ML task",
                    comparison.get("task"),
                    comparison.get("task"),
                ],
                [
                    "Configuration",
                    _configuration_text(
                        comparison.get(
                            "before_configuration"
                        )
                    ),
                    _configuration_text(
                        comparison.get(
                            "after_configuration"
                        )
                    ),
                ],
            ],
            styles,
            [
                38 * mm,
                68 * mm,
                68 * mm,
            ],
        )
    )

    # Section 2: Score comparison.
    story.append(
        Paragraph(
            "2. Readiness Score Comparison",
            styles["heading"],
        )
    )

    score_kind = comparison.get(
        "score_kind",
        "unavailable",
    )

    story.append(
        _table(
            ["Metric", "Result"],
            [
                [
                    "Comparison status",
                    (
                        "Comparable"
                        if comparable
                        else "Not comparable"
                    ),
                ],
                [
                    "Score type",
                    score_kind,
                ],
                [
                    "Before score",
                    _score_text(before_score),
                ],
                [
                    "After score",
                    _score_text(after_score),
                ],
                [
                    "Score change",
                    (
                        _change_text(score_change)
                        if comparable
                        else "Not comparable"
                    ),
                ],
            ],
            styles,
            [
                70 * mm,
                104 * mm,
            ],
        )
    )

    if not comparable:
        story.append(
            Spacer(1, 4 * mm)
        )

        story.append(
            Paragraph(
                (
                    "The assessments are not directly "
                    "comparable. Their scores are shown "
                    "for reference only; no improvement "
                    "or decline is claimed."
                ),
                styles["warning"],
            )
        )

        reasons = comparison.get(
            "comparability_reasons",
            [],
        )

        for reason in reasons:
            story.append(
                _paragraph(
                    f"Reason: {reason}",
                    styles["body"],
                )
            )

    elif score_kind == "provisional":
        story.append(
            Spacer(1, 4 * mm)
        )

        story.append(
            Paragraph(
                (
                    "Both scores are provisional. "
                    "Their difference describes only "
                    "the measured dimensions and "
                    "is not a final readiness improvement."
                ),
                styles["warning"],
            )
        )

    # Section 3: Individual dimensions.
    story.append(
        Paragraph(
            "3. Dimension-Level Comparison",
            styles["heading"],
        )
    )

    dimension_rows = _dimension_rows(
        comparison
    )

    if dimension_rows:
        story.append(
            _table(
                [
                    "Dimension",
                    "Before",
                    "After",
                    "Change",
                ],
                dimension_rows,
                styles,
                [
                    55 * mm,
                    35 * mm,
                    35 * mm,
                    49 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                "No dimension comparisons were recorded.",
                styles["body"],
            )
        )

    # Section 4: Issue summary.
    story.append(
        Paragraph(
            "4. Issue Summary",
            styles["heading"],
        )
    )

    issue_summary = comparison.get(
        "issue_summary",
        {},
    )

    if not isinstance(issue_summary, dict):
        issue_summary = {}

    story.append(
        _table(
            ["Metric", "Recorded count"],
            [
                [
                    "Before assessment",
                    issue_summary.get("before_count"),
                ],
                [
                    "After assessment",
                    issue_summary.get("after_count"),
                ],
                [
                    "Resolved issues",
                    issue_summary.get("resolved_count"),
                ],
                [
                    "Remaining issues",
                    issue_summary.get("remaining_count"),
                ],
                [
                    "New issues",
                    issue_summary.get("new_count"),
                ],
            ],
            styles,
            [
                100 * mm,
                74 * mm,
            ],
        )
    )

    # Section 5: Resolved issues.
    story.append(
        Paragraph(
            "5. Resolved Issues",
            styles["heading"],
        )
    )

    resolved_rows = _issue_rows(
        comparison.get(
            "resolved_issues",
            [],
        )
    )

    if resolved_rows:
        story.append(
            _table(
                [
                    "Dimension",
                    "Severity",
                    "Original finding",
                ],
                resolved_rows,
                styles,
                [
                    45 * mm,
                    30 * mm,
                    99 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                "No resolved issues were recorded.",
                styles["body"],
            )
        )

    # Section 6: Remaining issues.
    story.append(
        Paragraph(
            "6. Remaining Issues",
            styles["heading"],
        )
    )

    remaining_rows: list[list[str]] = []

    for item in comparison.get(
        "remaining_issues",
        [],
    ):
        if not isinstance(item, dict):
            continue

        # The comparison engine retains both versions
        # of each issue. Show its latest recorded state.
        latest = item.get(
            "after",
            item,
        )

        if isinstance(latest, dict):
            remaining_rows.extend(
                _issue_rows([latest])
            )

    if remaining_rows:
        story.append(
            _table(
                [
                    "Dimension",
                    "Severity",
                    "Current finding",
                ],
                remaining_rows,
                styles,
                [
                    45 * mm,
                    30 * mm,
                    99 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                "No remaining issues were recorded.",
                styles["body"],
            )
        )

    # Section 7: Newly detected issues.
    story.append(
        Paragraph(
            "7. Newly Detected Issues",
            styles["heading"],
        )
    )

    new_rows = _issue_rows(
        comparison.get(
            "new_issues",
            [],
        )
    )

    if new_rows:
        story.append(
            _table(
                [
                    "Dimension",
                    "Severity",
                    "New finding",
                ],
                new_rows,
                styles,
                [
                    45 * mm,
                    30 * mm,
                    99 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                "No newly detected issues were recorded.",
                styles["body"],
            )
        )

    # Section 8: Pending checks and limitations.
    story.append(
        Paragraph(
            "8. Outstanding Checks and Limitations",
            styles["heading"],
        )
    )

    before_pending = comparison.get(
        "before_pending_dimensions",
        [],
    )

    after_pending = comparison.get(
        "after_pending_dimensions",
        [],
    )

    if before_pending:
        story.append(
            _paragraph(
                (
                    "Before assessment pending: "
                    + ", ".join(
                        str(item)
                        for item in before_pending
                    )
                ),
                styles["body"],
            )
        )

    if after_pending:
        story.append(
            _paragraph(
                (
                    "After assessment pending: "
                    + ", ".join(
                        str(item)
                        for item in after_pending
                    )
                ),
                styles["body"],
            )
        )

    limitations = comparison.get(
        "limitations",
        [],
    )

    if limitations:
        for limitation in limitations:
            story.append(
                _paragraph(
                    limitation,
                    styles["body"],
                )
            )

    elif not before_pending and not after_pending:
        story.append(
            _paragraph(
                (
                    "No outstanding checks or limitations "
                    "were included in the comparison result."
                ),
                styles["body"],
            )
        )

    story.append(
        Spacer(1, 5 * mm)
    )

    story.append(
        _paragraph(
            (
                "This report documents NUMPA's recorded "
                "assessment comparison. A change in dataset "
                "readiness does not establish that a trained "
                "machine-learning model will perform better."
            ),
            styles["small"],
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
        title="NUMPA ML Readiness Comparison",
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
            "Comparison PDF generation failed."
        )

    return pdf_bytes


def generate_comparison_pdf_from_assessments(
    before_assessment: dict[str, Any],
    after_assessment: dict[str, Any],
) -> bytes:
    """Compare two assessments and generate their PDF.

    Always delegate comparability decisions and score
    calculations to the existing comparison engine.
    """

    comparison = compare_assessments(
        before_assessment,
        after_assessment,
    )

    return generate_readiness_comparison_pdf(
        comparison
    )
