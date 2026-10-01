
"""NUMPA v2 - Dataset overview PDF generation.

Creates an in-memory PDF from the existing dataset profile
and validation report.

No dataset rows are sent to an external service.
Only a limited preview is included in the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.services.analysis.profiler import generate_dataset_profile
from app.services.analysis.validator import validate_dataset


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm

NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#2563EB")
PALE_BLUE = colors.HexColor("#EFF6FF")
SLATE = colors.HexColor("#475569")
LIGHT_BORDER = colors.HexColor("#CBD5E1")


def _text(value: Any, *, limit: int = 180) -> str:
    """Convert a value into safe, length-limited PDF text."""

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    rendered = str(value).replace("\n", " ").strip()

    if len(rendered) > limit:
        rendered = rendered[: limit - 1] + "…"

    # Escape user-provided column names and cell values
    # before inserting them into ReportLab paragraphs.
    return escape(rendered)


def _styles() -> dict[str, ParagraphStyle]:
    """Create the report's reusable typography."""

    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "NUMPATitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=NAVY,
            spaceAfter=9,
        ),
        "subtitle": ParagraphStyle(
            "NUMPASubtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=15,
            textColor=SLATE,
            spaceAfter=15,
        ),
        "heading": ParagraphStyle(
            "NUMPAHeading",
            parent=base["Heading2"],
            fontSize=12,
            leading=16,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "NUMPABody",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            spaceAfter=5,
        ),
        "cell": ParagraphStyle(
            "NUMPACell",
            parent=base["Normal"],
            fontSize=7.5,
            leading=10,
            wordWrap="CJK",
        ),
        "header": ParagraphStyle(
            "NUMPAHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.white,
        ),
        "footer": ParagraphStyle(
            "NUMPAFooter",
            parent=base["Normal"],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=SLATE,
        ),
    }


def _paragraph(
    value: Any,
    style: ParagraphStyle,
    *,
    limit: int = 180,
) -> Paragraph:
    """Create a safe paragraph from user-controlled text."""

    return Paragraph(
        _text(value, limit=limit),
        style,
    )


def _table(
    headers: list[str],
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    *,
    widths: list[float] | None = None,
) -> Table:
    """Build a styled table with a repeating header."""

    content = [
        [
            _paragraph(header, styles["header"])
            for header in headers
        ]
    ]

    for row in rows:
        content.append(
            [
                _paragraph(value, styles["cell"])
                for value in row
            ]
        )

    table = Table(
        content,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    PALE_BLUE,
                ]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
                (
                    "LINEBELOW",
                    (0, 1),
                    (-1, -1),
                    0.25,
                    LIGHT_BORDER,
                ),
            ]
        )
    )

    return table


def _page_footer(
    canvas: Any,
    document: Any,
) -> None:
    """Draw a simple page number and report footer."""

    canvas.saveState()

    canvas.setStrokeColor(LIGHT_BORDER)
    canvas.line(
        MARGIN,
        15 * mm,
        PAGE_WIDTH - MARGIN,
        15 * mm,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)

    canvas.drawString(
        MARGIN,
        10 * mm,
        "NUMPA | Dataset Overview",
    )

    canvas.drawRightString(
        PAGE_WIDTH - MARGIN,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def _make_document(buffer: BytesIO) -> BaseDocTemplate:
    """Create a multipage A4 document with a footer."""

    document = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        title="NUMPA Dataset Overview",
        author="NUMPA",
    )

    frame = Frame(
        MARGIN,
        22 * mm,
        PAGE_WIDTH - 2 * MARGIN,
        PAGE_HEIGHT - 42 * mm,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    document.addPageTemplates(
        [
            PageTemplate(
                id="NUMPA",
                frames=[frame],
                onPage=_page_footer,
            )
        ]
    )

    return document


def _overview_rows(
    dataframe: pd.DataFrame,
    validation: dict[str, Any],
) -> list[list[Any]]:
    """Create the dataset's high-level statistics."""

    missing = int(dataframe.isna().sum().sum())

    return [
        ["Rows", f"{len(dataframe):,}"],
        ["Columns", f"{len(dataframe.columns):,}"],
        ["Missing cells", f"{missing:,}"],
        [
            "Missing cell percentage",
            (
                f"{100 * missing / dataframe.size:.2f}%"
                if dataframe.size
                else "0.00%"
            ),
        ],
        [
            "Validation issues",
            validation.get(
                "issue_count",
                len(validation.get("issues", [])),
            ),
        ],
        [
            "Validation status",
            (
                "Passed"
                if validation.get("valid", False)
                else "Needs attention"
            ),
        ],
    ]


def _column_rows(
    dataframe: pd.DataFrame,
) -> list[list[Any]]:
    """Generate concise statistics for every column."""

    rows: list[list[Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        rows.append(
            [
                str(column),
                str(series.dtype),
                missing_count,
                (
                    f"{100 * missing_count / len(dataframe):.1f}%"
                    if len(dataframe)
                    else "0.0%"
                ),
                unique_count,
            ]
        )

    return rows


def _validation_rows(
    validation: dict[str, Any],
) -> list[list[Any]]:
    """Convert validation issues into report table rows."""

    rows: list[list[Any]] = []

    for issue in validation.get("issues", []):
        if isinstance(issue, dict):
            rows.append(
                [
                    issue.get(
                        "code",
                        issue.get("type", "Issue"),
                    ),
                    issue.get(
                        "column",
                        "Dataset",
                    ),
                    issue.get(
                        "message",
                        issue.get("description", ""),
                    ),
                ]
            )

        else:
            rows.append(
                [
                    "Issue",
                    "Dataset",
                    str(issue),
                ]
            )

    return rows


def _preview_rows(
    dataframe: pd.DataFrame,
    *,
    max_rows: int = 5,
    max_columns: int = 5,
) -> tuple[list[str], list[list[Any]]]:
    """Limit preview size to keep PDF tables readable."""

    preview = dataframe.iloc[
        :max_rows,
        :max_columns,
    ]

    headers = [
        str(column)
        for column in preview.columns
    ]

    rows = [
        [
            preview.iloc[row_index, column_index]
            for column_index in range(len(headers))
        ]
        for row_index in range(len(preview))
    ]

    return headers, rows


def generate_dataset_overview_pdf(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str = "Dataset",
    dataset_id: str | None = None,
    profile: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    include_preview: bool = True,
) -> bytes:
    """Generate an A4 dataset overview PDF in memory.

    Existing profile and validation results may be supplied
    to avoid recalculation. When omitted, they are generated
    from the provided DataFrame.

    Returns:
        Complete PDF file as bytes.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "A pandas DataFrame is required."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot generate a report for an empty dataset."
        )

    if not dataframe.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    # Generate existing NUMPA diagnostics when the caller
    # has not already calculated them.
    if profile is None:
        profile = generate_dataset_profile(
            dataframe
        )

    if validation is None:
        validation = validate_dataset(
            dataframe
        )

    if not isinstance(profile, dict):
        raise TypeError(
            "The dataset profile must be a dictionary."
        )

    if not isinstance(validation, dict):
        raise TypeError(
            "The validation report must be a dictionary."
        )

    styles = _styles()
    story: list[Any] = []

    generated_at = datetime.now(
        timezone.utc
    ).strftime("%d %B %Y, %H:%M UTC")

    story.append(
        Paragraph(
            "Dataset Overview Report",
            styles["title"],
        )
    )

    story.append(
        Paragraph(
            (
                f"Dataset: {_text(dataset_name)}"
                f"<br/>Generated: {_text(generated_at)}"
                + (
                    f"<br/>Dataset ID: {_text(dataset_id)}"
                    if dataset_id
                    else ""
                )
            ),
            styles["subtitle"],
        )
    )

    # Section 1: Dataset summary.
    story.append(
        Paragraph(
            "1. Dataset Summary",
            styles["heading"],
        )
    )

    story.append(
        _table(
            ["Metric", "Value"],
            _overview_rows(
                dataframe,
                validation,
            ),
            styles,
            widths=[
                75 * mm,
                99 * mm,
            ],
        )
    )

    story.append(
        Spacer(1, 7 * mm)
    )

    # Section 2: Column-level statistics.
    story.append(
        Paragraph(
            "2. Column Overview",
            styles["heading"],
        )
    )

    story.append(
        Paragraph(
            (
                "Column types and missing-value statistics "
                "are calculated from the supplied dataset."
            ),
            styles["body"],
        )
    )

    story.append(
        _table(
            [
                "Column",
                "Data type",
                "Missing",
                "Missing %",
                "Unique",
            ],
            _column_rows(dataframe),
            styles,
            widths=[
                53 * mm,
                38 * mm,
                25 * mm,
                32 * mm,
                26 * mm,
            ],
        )
    )

    story.append(
        Spacer(1, 7 * mm)
    )

    # Section 3: Validation findings.
    story.append(
        Paragraph(
            "3. Data Quality Findings",
            styles["heading"],
        )
    )

    validation_rows = _validation_rows(
        validation
    )

    if validation_rows:
        story.append(
            _table(
                [
                    "Issue",
                    "Column",
                    "Description",
                ],
                validation_rows,
                styles,
                widths=[
                    45 * mm,
                    42 * mm,
                    87 * mm,
                ],
            )
        )

    else:
        story.append(
            Paragraph(
                (
                    "No issues were reported by the "
                    "current validation checks."
                ),
                styles["body"],
            )
        )

    # Section 4: A small, optional data preview.
    if include_preview:
        story.append(
            Paragraph(
                "4. Dataset Preview",
                styles["heading"],
            )
        )

        headers, rows = _preview_rows(
            dataframe
        )

        if headers and rows:
            column_width = (
                174 * mm / len(headers)
            )

            story.append(
                _table(
                    headers,
                    rows,
                    styles,
                    widths=[
                        column_width
                        for _ in headers
                    ],
                )
            )

            story.append(
                Spacer(1, 3 * mm)
            )

            story.append(
                Paragraph(
                    (
                        "Preview is limited to the first "
                        "5 rows and 5 columns."
                    ),
                    styles["body"],
                )
            )

    # Report limitations must remain explicit.
    story.append(
        Paragraph(
            "Report Notes",
            styles["heading"],
        )
    )

    story.append(
        Paragraph(
            (
                "This report describes dataset structure "
                "and preliminary quality checks. It does "
                "not certify that the dataset is suitable "
                "for a particular machine-learning task. "
                "Use NUMPA's task-specific readiness "
                "assessment for further evaluation."
            ),
            styles["body"],
        )
    )

    # Construct the PDF entirely in memory.
    buffer = BytesIO()

    document = _make_document(
        buffer
    )

    document.build(
        story
    )

    pdf_bytes = buffer.getvalue()
    buffer.close()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError(
            "PDF generation did not produce a valid PDF header."
        )

    return pdf_bytes
