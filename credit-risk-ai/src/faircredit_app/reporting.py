"""Privacy-minimizing PDF export for educational assessments."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import math
import unicodedata
from typing import Mapping

DISCLAIMER = (
    "Educational and research prototype only. This output is not a bank decision, "
    "loan offer, credit score, approval guarantee, or professional financial assessment."
)

NAVY = "#111827"
BLUE = "#1D4ED8"
PALE_BLUE = "#EAF0FF"
RED = "#B42318"
PALE_RED = "#FEF3F2"
INK = "#344054"
MUTED = "#475467"
LINE = "#D0D5DD"


def _pdf_text(text: object) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")


def _validate_report_data(
    bad_probability: float,
    risk_score: int,
    risk_category: str,
    predicted_label: str,
    dataset_rows: int,
    training_rows: int,
    test_rows: int,
    holdout_metrics: Mapping[str, float],
) -> None:
    if not math.isfinite(bad_probability) or not 0 <= bad_probability <= 1:
        raise ValueError("Bad-credit probability must be between zero and one.")
    if isinstance(risk_score, bool) or not isinstance(risk_score, int) or not 0 <= risk_score <= 100:
        raise ValueError("Risk score must be an integer between zero and 100.")
    expected_score = int(round((1 - bad_probability) * 100))
    if risk_score != expected_score:
        raise ValueError("Risk score must match the estimated bad-credit probability.")
    expected_category = "Low risk" if risk_score >= 67 else "Moderate risk" if risk_score >= 34 else "High risk"
    if risk_category != expected_category:
        raise ValueError("Risk category must match the illustrative risk-score band.")
    expected_label = (
        "Bad-credit history (dataset label 1)"
        if bad_probability >= 0.5
        else "Good-credit history (dataset label 0)"
    )
    if predicted_label != expected_label:
        raise ValueError("Predicted label must match the 50% bad-credit probability threshold.")
    if any(count <= 0 for count in (dataset_rows, training_rows, test_rows)):
        raise ValueError("Dataset, training, and holdout row counts must be positive.")
    if training_rows + test_rows != dataset_rows:
        raise ValueError("Training and holdout row counts must add up to the dataset size.")
    for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"):
        value = holdout_metrics.get(metric)
        if value is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Holdout metric {metric} must be between zero and one.")


def generate_assessment_pdf(
    *,
    report_id: str,
    bad_probability: float,
    risk_score: int,
    risk_category: str,
    predicted_label: str,
    dataset_rows: int,
    training_rows: int,
    test_rows: int,
    holdout_metrics: Mapping[str, float],
    model_parameters: Mapping[str, object],
) -> bytes:
    """Build a detailed PDF summary without including the submitted profile."""
    _validate_report_data(
        bad_probability, risk_score, risk_category, predicted_label,
        dataset_rows, training_rows, test_rows, holdout_metrics,
    )
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("PDF export is unavailable. Install the project requirements.") from exc

    buffer = BytesIO()
    report_id = escape(str(report_id))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page_width, page_height = letter
    margin = 0.68 * inch
    content_width = page_width - 2 * margin

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=27, textColor=colors.HexColor(NAVY),
        alignment=TA_LEFT, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportLead", parent=styles["BodyText"], fontSize=9.5, leading=14,
        textColor=colors.HexColor(MUTED), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.HexColor(NAVY),
        spaceBefore=12, spaceAfter=6, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyCopy", parent=styles["BodyText"], fontSize=8.7, leading=12.5,
        textColor=colors.HexColor(INK), spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SmallCopy", parent=styles["BodyText"], fontSize=7.6, leading=10.5,
        textColor=colors.HexColor(MUTED),
    ))
    styles.add(ParagraphStyle(
        name="CardLabel", parent=styles["BodyText"], fontSize=7.3, leading=9.5,
        textColor=colors.HexColor(MUTED), spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="CardValue", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=13.2, leading=16, textColor=colors.HexColor(NAVY),
    ))
    styles.add(ParagraphStyle(
        name="CardScore", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=21, leading=24, textColor=colors.HexColor(BLUE),
    ))
    styles.add(ParagraphStyle(
        name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.8, leading=10, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell", parent=styles["BodyText"], fontSize=8, leading=10.5,
        textColor=colors.HexColor(INK),
    ))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=12, textColor=colors.HexColor(RED),
    ))

    def paragraph(text: object, style: str = "BodyCopy") -> Paragraph:
        return Paragraph(_pdf_text(text), styles[style])

    def draw_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(BLUE))
        canvas.rect(0, page_height - 0.16 * inch, page_width, 0.16 * inch, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(margin, page_height - 0.42 * inch, "FAIRCREDIT  /  EDUCATIONAL ASSESSMENT")
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(page_width - margin, page_height - 0.42 * inch, f"Report {report_id}")
        canvas.setStrokeColor(colors.HexColor(LINE))
        canvas.line(margin, 0.48 * inch, page_width - margin, 0.48 * inch)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(margin, 0.31 * inch, "Generated in memory; applicant profile is not included.")
        canvas.drawRightString(page_width - margin, 0.31 * inch, f"Page {document.page}")
        canvas.restoreState()

    story = [
        Spacer(1, 0.2 * inch),
        paragraph("Assessment report", "ReportTitle"),
        paragraph(f"Hypothetical model assessment | Generated {generated}", "ReportLead"),
    ]

    category_background = PALE_RED if risk_category == "High risk" else PALE_BLUE
    category_color = RED if risk_category == "High risk" else BLUE
    category_style = ParagraphStyle(
        "CardCategory", parent=styles["CardValue"], textColor=colors.HexColor(category_color),
    )
    cards = [
        [
            Paragraph("ILLUSTRATIVE RISK SCORE", styles["CardLabel"]),
            Paragraph("RISK CATEGORY", styles["CardLabel"]),
        ],
        [
            Paragraph(f"{risk_score}<font size='11'> / 100</font>", styles["CardScore"]),
            Paragraph(_pdf_text(risk_category), category_style),
        ],
        [
            Paragraph("ESTIMATED BAD-LABEL PROBABILITY", styles["CardLabel"]),
            Paragraph("MODEL'S PREDICTED DATASET LABEL", styles["CardLabel"]),
        ],
        [
            Paragraph(f"{bad_probability:.1%}", styles["CardValue"]),
            Paragraph(_pdf_text(predicted_label), styles["CardValue"]),
        ],
    ]
    cards_table = Table(cards, colWidths=[content_width * 0.5] * 2, rowHeights=[None, None, None, None])
    cards_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(PALE_BLUE)),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor(category_background)),
        ("BOX", (0, 0), (0, -1), 0.6, colors.HexColor(LINE)),
        ("BOX", (1, 0), (1, -1), 0.6, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([
        cards_table,
        paragraph("Result interpretation", "SectionTitle"),
        paragraph(
            f"The model estimated a <b>{bad_probability:.1%}</b> probability for the dataset's "
            f"historical bad-credit label. The illustrative score is <b>{risk_score}/100</b>, "
            "calculated as 100 x (1 - estimated bad-label probability), rounded to the nearest point.",
        ),
        paragraph(
            "The risk band and predicted dataset label are different summaries: the band uses "
            "illustrative score ranges (Low: 67-100, Moderate: 34-66, High: 0-33), while the "
            "predicted label uses a 50% model-probability threshold. Neither is a lending decision.",
        ),
        paragraph("Model validation snapshot", "SectionTitle"),
    ])

    metric_descriptions = {
        "Accuracy": "Share of holdout examples classified correctly",
        "Precision": "Of predicted bad labels, share that were bad labels",
        "Recall": "Of actual bad labels, share identified by the model",
        "F1": "Balance of precision and recall",
        "ROC-AUC": "Ranking separation across thresholds",
    }
    metric_rows = [[paragraph("MEASURE", "TableHead"), paragraph("HOLDOUT RESULT", "TableHead"), paragraph("WHAT IT MEANS", "TableHead")]]
    for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"):
        metric_rows.append([
            paragraph(metric, "TableCell"),
            paragraph(f"{holdout_metrics[metric]:.1%}", "TableCell"),
            paragraph(metric_descriptions[metric], "TableCell"),
        ])
    metrics_table = Table(metric_rows, colWidths=[content_width * .22, content_width * .2, content_width * .58], repeatRows=1)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        metrics_table,
        paragraph(
            f"These are model-level results from one stratified holdout split, not measures of this "
            f"profile or a guarantee of future performance. The dataset contains {dataset_rows:,} "
            f"historical records ({training_rows:,} training, {test_rows:,} holdout).",
            "SmallCopy",
        ),
        paragraph("Model and data", "SectionTitle"),
    ])
    parameter_summary = ", ".join(
        f"{key.replace('_', ' ')}: {value}" for key, value in model_parameters.items()
    )
    model_rows = [
        [paragraph("Model", "TableCell"), paragraph("Random Forest classifier with preprocessing pipeline", "TableCell")],
        [paragraph("Features", "TableCell"), paragraph("Financial and credit-history fields; demographic fields excluded from scoring", "TableCell")],
        [paragraph("Training setup", "TableCell"), paragraph(f"Stratified 80/20 split; {training_rows:,} train / {test_rows:,} holdout", "TableCell")],
        [paragraph("Selected settings", "TableCell"), paragraph(parameter_summary or "Default project model settings", "TableCell")],
        [paragraph("Source data", "TableCell"), paragraph("UCI Statlog German Credit - historical 1994 sample", "TableCell")],
    ]
    model_table = Table(model_rows, colWidths=[content_width * .23, content_width * .77])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(PALE_BLUE)),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        model_table,
        paragraph("Important limitations", "SectionTitle"),
        KeepTogether([
            paragraph("- The probability is uncalibrated; it is not confidence, default likelihood, or loan-approval probability."),
            paragraph("- The historical dataset is small and may not represent current applicants, products, or populations."),
            paragraph("- No feature-level explanation is provided. Model output does not establish cause and effect."),
            paragraph("- Excluding demographics does not eliminate correlated proxy effects; inputs and score are not financial advice."),
        ]),
        paragraph("Privacy", "SectionTitle"),
        paragraph(
            "This PDF contains the score summary and model-level information only. It excludes the "
            "submitted financial profile, names, contact details, and demographic attributes. "
            "FairCredit generates the file in memory and does not retain it; if downloaded, it is saved by your browser.",
        ),
        Spacer(1, 8),
        paragraph(DISCLAIMER, "Callout"),
    ])

    document = SimpleDocTemplate(
        buffer, pagesize=letter, leftMargin=margin, rightMargin=margin,
        topMargin=0.62 * inch, bottomMargin=0.66 * inch,
        title="FairCredit educational assessment report",
        author="FairCredit educational prototype",
    )
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()
