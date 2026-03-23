"""
report_generator.py — PDF Report Generator for Spinal Curvature Analysis
Uses ReportLab to produce a clinical-style screening report.
"""

import io
import tempfile
import os
from datetime import datetime

import cv2
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.piecharts import Pie


# ─── Color palette ───────────────────────────────────────────────
DARK_BG    = colors.HexColor("#0f1117")
PANEL_BG   = colors.HexColor("#1c1f2e")
ACCENT     = colors.HexColor("#4f8ef7")
TEXT_MAIN  = colors.HexColor("#1a1a2e")
TEXT_LIGHT = colors.HexColor("#555555")
BORDER     = colors.HexColor("#d0d4e8")

SEV_COLORS = {
    "Normal":             colors.HexColor("#3cc86e"),
    "Mild Scoliosis":     colors.HexColor("#f5a623"),
    "Moderate Scoliosis": colors.HexColor("#e07820"),
    "Severe Scoliosis":   colors.HexColor("#d0312d"),
}


def _severity_color(severity: str) -> colors.Color:
    return SEV_COLORS.get(severity, colors.grey)


def _make_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontSize=26, fontName="Helvetica-Bold",
        textColor=ACCENT, alignment=TA_CENTER,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle",
        fontSize=11, fontName="Helvetica",
        textColor=TEXT_LIGHT, alignment=TA_CENTER,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=ACCENT, spaceBefore=16, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontSize=10, fontName="Helvetica",
        textColor=TEXT_MAIN, leading=15, spaceAfter=4,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer",
        fontSize=8.5, fontName="Helvetica-Oblique",
        textColor=TEXT_LIGHT, leading=13,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="SmallLabel",
        fontSize=8, fontName="Helvetica",
        textColor=TEXT_LIGHT,
    ))
    return styles


def _severity_gauge_drawing(angle: float, severity: str) -> Drawing:
    """Draw a simple horizontal gauge bar showing angle position."""
    w, h = 320, 52
    d = Drawing(w, h)

    max_angle = 50.0
    segments = [
        (0, 10,  "#3cc86e"),
        (10, 25, "#f5a623"),
        (25, 40, "#e07820"),
        (40, 50, "#d0312d"),
    ]
    bar_x, bar_y, bar_h = 10, 18, 16
    bar_w = w - 20

    for lo, hi, col in segments:
        seg_x = bar_x + (lo / max_angle) * bar_w
        seg_w = ((hi - lo) / max_angle) * bar_w
        r = Rect(seg_x, bar_y, seg_w, bar_h, fillColor=colors.HexColor(col), strokeWidth=0)
        d.add(r)

    # Labels
    for lo, label in [(0, "0"), (10, "10"), (25, "25"), (40, "40"), (50, "50+")]:
        lx = bar_x + (lo / max_angle) * bar_w
        d.add(String(lx, bar_y - 11, label, fontSize=7, fontName="Helvetica",
                     fillColor=colors.HexColor("#888888")))

    # Needle
    clamped = min(angle, max_angle)
    needle_x = bar_x + (clamped / max_angle) * bar_w
    d.add(Line(needle_x, bar_y - 4, needle_x, bar_y + bar_h + 4,
               strokeColor=colors.white, strokeWidth=2.5))
    d.add(String(needle_x - 16, bar_y + bar_h + 5,
                 f"{angle:.1f} deg", fontSize=8.5, fontName="Helvetica-Bold",
                 fillColor=colors.HexColor("#222222")))

    return d


def generate_pdf_report(
    result,
    image_path: str,
    patient_name: str = "Anonymous",
    notes: str = "",
) -> bytes:
    """
    Generate a PDF screening report.

    Parameters
    ----------
    result : SpineAnalysisResult
        Output from CobbAngleEstimator.analyze()
    image_path : str
        Path to the original photo (used to embed in report)
    patient_name : str
        Optional patient/subject name for the report header
    notes : str
        Optional clinician notes to include

    Returns
    -------
    bytes
        Raw PDF bytes ready to be written to a file or returned via HTTP
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = _make_styles()
    story = []
    W = letter[0] - 1.7 * inch  # usable width

    # ── Header ────────────────────────────────────────────────────
    story.append(Paragraph("Spinal Curvature Screening Report", styles["ReportTitle"]))
    story.append(Paragraph("Postural Cobb Angle Estimation · AI-Assisted Analysis", styles["Subtitle"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT))
    story.append(Spacer(1, 10))

    # ── Meta info table ───────────────────────────────────────────
    now = datetime.now().strftime("%B %d, %Y  %H:%M")
    meta_data = [
        ["Subject:", patient_name,   "Date:", now],
        ["Confidence:", f"{result.confidence * 100:.0f}%", "Method:", "MediaPipe Pose + Polynomial Fit"],
    ]
    meta_table = Table(meta_data, colWidths=[0.9*inch, 2.1*inch, 0.9*inch, 2.7*inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,0), (-1,-1), TEXT_MAIN),
        ("TEXTCOLOR", (0,0), (0,-1), TEXT_LIGHT),
        ("TEXTCOLOR", (2,0), (2,-1), TEXT_LIGHT),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── Measurements section ──────────────────────────────────────
    story.append(Paragraph("Measurements", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 8))

    sev_color = _severity_color(result.severity)

    # Big metric row
    metric_data = [[
        Paragraph(f"<font size='28' color='#1a1a2e'><b>{result.cobb_angle:.1f}</b></font><font size='14' color='#888888'> deg</font>", styles["BodyText2"]),
        Paragraph(f"<font size='13' color='#888888'>Classification</font><br/><font size='16'><b>{result.severity}</b></font>", styles["BodyText2"]),
        Paragraph(f"<font size='13' color='#888888'>Shoulder Tilt</font><br/><font size='16'><b>{abs(result.shoulder_tilt_deg):.1f} deg</b></font>", styles["BodyText2"]),
        Paragraph(f"<font size='13' color='#888888'>Hip Tilt</font><br/><font size='16'><b>{abs(result.hip_tilt_deg):.1f} deg</b></font>", styles["BodyText2"]),
    ]]
    metric_table = Table(metric_data, colWidths=[W*0.22, W*0.30, W*0.24, W*0.24])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f4f6fb")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f4f6fb")]),
        ("BOX", (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID", (0,0), (-1,-1), 0.5, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        # Highlight cobb cell
        ("BACKGROUND", (0,0), (0,0), colors.HexColor("#eef3ff")),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 12))

    # Gauge
    story.append(Paragraph("Severity Gauge", styles["SmallLabel"]))
    story.append(Spacer(1, 4))
    gauge = _severity_gauge_drawing(result.cobb_angle, result.severity)
    story.append(gauge)
    story.append(Spacer(1, 6))

    # Severity legend
    legend_items = [
        ("Normal", "< 10 deg", "#3cc86e"),
        ("Mild Scoliosis", "10-25 deg", "#f5a623"),
        ("Moderate Scoliosis", "25-40 deg", "#e07820"),
        ("Severe Scoliosis", "> 40 deg", "#d0312d"),
    ]
    legend_data = [[
        Paragraph(
            f'<font color="{col}"><b>●</b></font> <b>{label}</b> <font color="#888888">({rng})</font>',
            styles["SmallLabel"]
        )
        for label, rng, col in legend_items
    ]]
    legend_table = Table(legend_data, colWidths=[W/4]*4)
    legend_table.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 18))

    # ── Images section ────────────────────────────────────────────
    story.append(Paragraph("Annotated Image", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 8))

    # Save annotated image to a temp file for embedding
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        cv2.imwrite(tf.name, result.annotated_image)
        ann_tmp = tf.name

    try:
        img_h_px, img_w_px = result.annotated_image.shape[:2]
        aspect = img_h_px / img_w_px
        display_w = W * 0.75
        display_h = display_w * aspect
        # Cap height
        max_h = 3.8 * inch
        if display_h > max_h:
            display_h = max_h
            display_w = display_h / aspect

        img_table_data = [[RLImage(ann_tmp, width=display_w, height=display_h)]]
        img_table = Table(img_table_data, colWidths=[W])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BOX", (0,0), (-1,-1), 0.5, BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8f9fc")),
        ]))
        story.append(img_table)
        story.append(Paragraph(
            "Yellow line = estimated spine midline  ·  Blue line = shoulder reference  ·  Purple line = hip reference",
            styles["SmallLabel"]
        ))
    finally:
        os.unlink(ann_tmp)

    story.append(Spacer(1, 18))

    # ── Clinical notes ────────────────────────────────────────────
    story.append(Paragraph("Notes", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 6))

    notes_text = notes.strip() if notes.strip() else "No additional notes provided."
    story.append(Paragraph(notes_text, styles["BodyText2"]))
    story.append(Spacer(1, 18))

    # ── Interpretation guide ──────────────────────────────────────
    story.append(Paragraph("Interpretation Guide", styles["SectionHeader"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 6))

    interp = [
        ["Angle Range", "Classification", "Recommended Action"],
        ["< 10 deg",    "Normal",             "Routine monitoring; no clinical concern"],
        ["10-25 deg",   "Mild Scoliosis",     "Consider physician referral; monitor progression"],
        ["25-40 deg",   "Moderate Scoliosis", "Physician referral recommended; X-ray indicated"],
        ["> 40 deg",    "Severe Scoliosis",   "Urgent physician evaluation; X-ray required"],
    ]
    interp_table = Table(interp, colWidths=[1.2*inch, 1.7*inch, W - 2.9*inch])
    interp_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8edf8")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR", (0,0), (-1,-1), TEXT_MAIN),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOX", (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID", (0,0), (-1,-1), 0.3, BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fc")]),
    ]))
    story.append(interp_table)
    story.append(Spacer(1, 22))

    # ── Disclaimer ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 8))
    disc = (
        "<b>MEDICAL DISCLAIMER:</b> This report was generated by an AI-assisted postural screening tool. "
        "The Cobb angle estimate is derived from surface body landmarks detected in a standard photograph, "
        "not from radiographic imaging. This method provides an <i>approximation of postural curvature</i> "
        "and is NOT equivalent to a clinical Cobb angle measurement, which requires standing anteroposterior "
        "X-ray imaging for accurate vertebral endplate assessment. This report is intended for "
        "<b>educational and preliminary screening purposes only</b> and should not be used to make clinical "
        "decisions. Any individual with a positive screen or clinical concern should be referred to a "
        "qualified physician or orthopedic specialist for proper evaluation."
    )
    story.append(Paragraph(disc, styles["Disclaimer"]))

    doc.build(story)
    return buf.getvalue()