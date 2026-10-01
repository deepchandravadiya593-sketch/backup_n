
"""NUMPA v2 - ML Readiness Assessment PDF.

Generate a downloadable PDF from an existing readiness
assessment. The report reflects the assessment's actual
status and never upgrades a provisional score to final.
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
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# Report appearance.
NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#2563EB")
SLATE = colors.HexColor("#475569")
PALE = colors.HexColor("#EFF6FF")
BORDER = colors.HexColor("#CBD5E1")
AMBER = colors.HexColor("#92400E")

PAGE_WIDTH, _ = A4
CONTENT_WIDTH = PAGE_WIDTH - 36 * mm


def _safe_text(
    value: Any,
    *,
    limit: int = 300,
) -> str:
    """Escape arbitrary values before rendering them."""

    if value is None:
        return "Not specified"

    text = str(value).replace("\n", " ").strip()

    if len(text) > limit:
        text = text[: limit - 3] + "..."

    return escape(text)


def _styles() -> dict[str, ParagraphStyle]:
    """Create consistent typography for the report."""

    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReadinessTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=NAVY,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "ReadinessHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "ReadinessBody",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ReadinessSmall",
            parent=base["Normal"],
            fontSize=7.5,
            leading=11,
            textColor=SLATE,
        ),
        "cell": ParagraphStyle(
            "ReadinessCell",
            parent=base["Normal"],
            fontSize=7.5,
            leading=10,
            wordWrap="CJK",
        ),
        "header": ParagraphStyle(
            "ReadinessHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.white,
        ),
        "warning": ParagraphStyle(
            "ReadinessWarning",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
            textColor=AMBER,
            spaceAfter=6,
        ),
    }


def _paragraph(
    value: Any,
    style: ParagraphStyle,
) -> Paragraph:
    """Convert plain text into a safe PDF paragraph."""

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
    """Build a multipage-friendly table."""

    data = [
        [
            _paragraph(header, styles["header"])
            for header in headers
        ]
    ]

    for row in rows:
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
                    [colors.white, PALE],
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


def _footer(canvas: Any, document: Any) -> None:
    """Add a footer and page number to every page."""

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
        "NUMPA | ML Readiness Assessment",
    )

    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def _configuration(
    assessment: dict[str, Any],
) -> dict[str, Any]:
    """Support nested and legacy configuration fields."""

    config = assessment.get("configuration")

    if isinstance(config, dict):
        return config

    return {
        "name": assessment.get(
            "configuration_name",
            "NUMPA Default",
        ),
        "version": assessment.get(
            "configuration_version",
            1,
        ),
    }


def _pending_names(
    assessment: dict[str, Any],
) -> list[str]:
    """Normalize pending-dimension representations."""

    names: list[str] = []

    for item in assessment.get(
        "pending_dimensions",
        [],
    ):
        if isinstance(item, str):
            names.append(item)

        elif isinstance(item, dict):
            name = item.get(
                "dimension",
                item.get("name"),
            )

            if name:
                names.append(str(name))

    return list(dict.fromkeys(names))


def _score_details(
    assessment: dict[str, Any],
) -> tuple[str, Any]:
    """Identify the actual score and its status."""

    final_score = assessment.get(
        "overall_score"
    )

    if final_score is not None:
        return "Final", final_score

    provisional = assessment.get(
        "provisional_score"
    )

    if provisional is not None:
        return "Provisional", provisional

    return "Unavailable", "Not calculated"


def _dimension_rows(
    assessment: dict[str, Any],
) -> list[list[Any]]:
    """Prepare measured dimensions for display."""

    rows: list[list[Any]] = []

    for dimension in assessment.get(
        "dimensions",
        [],
    ):
        if not isinstance(dimension, dict):
            continue

        score = dimension.get("score")
        weight = dimension.get("weight")

        rows.append(
            [
                dimension.get("name", "Unknown"),
                (
                    f"{float(score):.1f}/100"
                    if score is not None
                    else "Pending"
                ),
                (
                    f"{float(weight):.1f}%"
                    if weight is not None
                    else "Not set"
                ),
                dimension.get(
                    "status",
                    "Not specified",
                ),
            ]
        )

    return rows


def _issue_rows(
    assessment: dict[str, Any],
) -> list[list[Any]]:
    """Prepare detected issues for the report."""

    rows: list[list[Any]] = []

    for issue in assessment.get(
        "issues",
        [],
    ):
        if not isinstance(issue, dict):
            continue

        rows.append(
            [
                issue.get("severity", "Unknown"),
                issue.get("dimension", "Dataset"),
                issue.get("column") or "Dataset",
                issue.get(
                    "message",
                    issue.get("code", "Issue"),
                ),
            ]
        )

    return rows


def generate_readiness_assessment_pdf(
    assessment: dict[str, Any],
) -> bytes:
    """Generate a readiness report from an existing assessment.

    The report uses stored assessment results. It does not
    recalculate scores, invent validation evidence or
    change the assessment's status.
    """

    if not isinstance(assessment, dict):
        raise TypeError(
            "Assessment must be a dictionary."
        )

    required = (
        "assessment_id",
        "dataset_id",
        "task",
    )

    missing = [
        key
        for key in required
        if not assessment.get(key)
    ]

    if missing:
        raise ValueError(
            "Missing required assessment fields: "
            + ", ".join(missing)
        )

    styles = _styles()
    story: list[Any] = []

    score_label, score_value = _score_details(
        assessment
    )

    config = _configuration(
        assessment
    )

    generated_at = datetime.now(
        timezone.utc
    ).strftime("%d %B %Y, %H:%M UTC")

    # Section 1: Report identification.
    story.append(
        Paragraph(
            "ML Readiness Assessment",
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

    story.append(
        _table(
            ["Field", "Value"],
            [
                [
                    "Assessment ID",
                    assessment["assessment_id"],
                ],
                [
                    "Dataset ID",
                    assessment["dataset_id"],
                ],
                [
                    "ML task",
                    assessment["task"],
                ],
                [
                    "Target column",
                    assessment.get(
                        "target_column"
                    ),
                ],
                [
                    "Intended algorithm",
                    assessment.get(
                        "intended_algorithm"
                    ),
                ],
                [
                    "Validation strategy",
                    assessment.get(
                        "validation_strategy"
                    ),
                ],
                [
                    "Configuration",
                    (
                        f"{config.get('name', 'Default')} "
                        f"(version {config.get('version', 1)})"
                    ),
                ],
            ],
            styles,
            [
                54 * mm,
                120 * mm,
            ],
        )
    )

    # Section 2: Score and assessment status.
    story.append(
        Paragraph(
            "1. Readiness Score",
            styles["heading"],
        )
    )

    story.append(
        _table(
            ["Metric", "Result"],
            [
                [
                    "Score type",
                    score_label,
                ],
                [
                    "Score",
                    (
                        f"{float(score_value):.2f}/100"
                        if isinstance(
                            score_value,
                            (int, float),
                        )
                        else score_value
                    ),
                ],
                [
                    "Assessment status",
                    assessment.get(
                        "assessment_status",
                        "Not specified",
                    ),
                ],
                [
                    "Measured weight",
                    (
                        f"{assessment['measured_weight_percentage']:.1f}%"
                        if isinstance(
                            assessment.get(
                                "measured_weight_percentage"
                            ),
                            (int, float),
                        )
                        else "Not specified"
                    ),
                ],
            ],
            styles,
            [
                74 * mm,
                100 * mm,
            ],
        )
    )

    if score_label != "Final":
        story.append(
            Spacer(1, 4 * mm)
        )

        story.append(
            Paragraph(
                (
                    "This assessment is provisional. "
                    "Its score reflects only the dimensions "
                    "evaluated so far. It must not be "
                    "interpreted as a final ML readiness score."
                ),
                styles["warning"],
            )
        )

    # Section 3: Dimension results.
    story.append(
        Paragraph(
            "2. Scoring Dimensions",
            styles["heading"],
        )
    )

    dimension_rows = _dimension_rows(
        assessment
    )

    if dimension_rows:
        story.append(
            _table(
                [
                    "Dimension",
                    "Score",
                    "Weight",
                    "Status",
                ],
                dimension_rows,
                styles,
                [
                    65 * mm,
                    34 * mm,
                    32 * mm,
                    43 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                "No dimension scores are available.",
                styles["body"],
            )
        )

    # Section 4: Outstanding checks.
    pending = _pending_names(
        assessment
    )

    story.append(
        Paragraph(
            "3. Outstanding Checks",
            styles["heading"],
        )
    )

    if pending:
        for name in pending:
            story.append(
                _paragraph(
                    f"Pending: {name}",
                    styles["body"],
                )
            )

    else:
        story.append(
            _paragraph(
                "No pending dimensions were recorded.",
                styles["body"],
            )
        )

    # Section 5: Data-quality and leakage issues.
    story.append(
        Paragraph(
            "4. Detected Issues",
            styles["heading"],
        )
    )

    issue_rows = _issue_rows(
        assessment
    )

    if issue_rows:
        story.append(
            _table(
                [
                    "Severity",
                    "Dimension",
                    "Column",
                    "Finding",
                ],
                issue_rows,
                styles,
                [
                    25 * mm,
                    43 * mm,
                    35 * mm,
                    71 * mm,
                ],
            )
        )

    else:
        story.append(
            _paragraph(
                "No issues were recorded by the completed checks.",
                styles["body"],
            )
        )

    # Section 6: Recommended actions.
    story.append(
        Paragraph(
            "5. Recommendations",
            styles["heading"],
        )
    )

    recommendations = assessment.get(
        "recommendations",
        [],
    )

    if recommendations:
        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            story.append(
                _paragraph(
                    f"{index}. {recommendation}",
                    styles["body"],
                )
            )

    else:
        story.append(
            _paragraph(
                "No recommendations were recorded.",
                styles["body"],
            )
        )

    # Section 7: Audit limitations.
    story.append(
        Paragraph(
            "6. Assessment Limitations",
            styles["heading"],
        )
    )

    audit = assessment.get(
        "audit",
        {},
    )

    limitations = (
        audit.get("limitations", [])
        if isinstance(audit, dict)
        else []
    )

    leakage = assessment.get(
        "leakage",
        {},
    )

    if isinstance(leakage, dict):
        limitations = [
            *limitations,
            *leakage.get(
                "pending_checks",
                [],
            ),
        ]

    # Deduplicate repeated limitations while
    # preserving their original order.
    limitations = list(
        dict.fromkeys(
            str(item)
            for item in limitations
        )
    )

    if limitations:
        for limitation in limitations:
            story.append(
                _paragraph(
                    limitation,
                    styles["body"],
                )
            )

    else:
        story.append(
            _paragraph(
                (
                    "This report reflects the checks recorded "
                    "in the supplied assessment."
                ),
                styles["body"],
            )
        )

    story.append(
        Spacer(1, 5 * mm)
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=BORDER,
        )
    )

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        _paragraph(
            (
                "NUMPA evaluates dataset readiness, "
                "not the accuracy or reliability of "
                "a future trained ML model."
            ),
            styles["small"],
        )
    )

    # Generate the PDF entirely in memory.
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="NUMPA ML Readiness Assessment",
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
