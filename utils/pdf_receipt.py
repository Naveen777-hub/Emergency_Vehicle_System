"""
utils/pdf_receipt.py
Generates a formal A4 PDF e-challan receipt using reportlab.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Colour palette (matches gov.css) ──────────────────────────────────────────
NAVY     = colors.HexColor("#003366")
NAVY_DRK = colors.HexColor("#002244")
SAFFRON  = colors.HexColor("#FF6600")
GREEN    = colors.HexColor("#138808")
DANGER   = colors.HexColor("#CC0000")
GREY     = colors.HexColor("#F0F0F0")
BORDER   = colors.HexColor("#CCCCCC")
TEXT     = colors.HexColor("#1A1A1A")
MUTED    = colors.HexColor("#777777")
WHITE    = colors.white
GOLD     = colors.HexColor("#FFD700")


# ── Helper style factory ───────────────────────────────────────────────────────
def _style(name, **kwargs):
    defaults = dict(
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=TEXT, spaceAfter=0, spaceBefore=0
    )
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)


def generate_receipt_pdf(challan) -> bytes:
    """
    Accept a Challan ORM object and return raw PDF bytes.
    """
    buffer = io.BytesIO()
    page_w, page_h = A4
    margin = 18 * mm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=margin,
        bottomMargin=margin,
        leftMargin=margin,
        rightMargin=margin,
    )

    usable_w = page_w - 2 * margin
    story = []

    # ── 1. GOVERNMENT HEADER BAND ─────────────────────────────────────────────
    header_data = [
        [
            Paragraph(
                "<b>GOVERNMENT OF INDIA</b><br/>"
                "<font size=8>Ministry of Road Transport and Highways</font><br/>"
                "<font size=7>Road Safety &amp; Enforcement Division</font>",
                _style("GovTitle", fontName="Helvetica-Bold", fontSize=13,
                       leading=16, textColor=WHITE, alignment=TA_CENTER),
            )
        ]
    ]
    header_table = Table(header_data, colWidths=[usable_w])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    story.append(header_table)

    # Saffron accent strip
    story.append(Table([[""]], colWidths=[usable_w], rowHeights=[4]))
    story[-1].setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SAFFRON)]))

    story.append(Spacer(1, 6 * mm))

    # ── 2. DOCUMENT TITLE ────────────────────────────────────────────────────
    story.append(Paragraph(
        "E-CHALLAN / TRAFFIC VIOLATION NOTICE",
        _style("DocTitle", fontName="Helvetica-Bold", fontSize=14, leading=18,
               textColor=NAVY, alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        "Under the Motor Vehicles Act, 1988 — Section 194B (Failure to Yield to Emergency Vehicle)",
        _style("DocSub", fontSize=8, textColor=MUTED, alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width=usable_w, thickness=1.5, color=NAVY))
    story.append(Spacer(1, 4 * mm))

    # ── 3. CHALLAN META ───────────────────────────────────────────────────────
    meta_rows = [
        ["Challan Number", challan.challan_number,
         "Date of Issue",  challan.created_at.strftime("%d %B %Y")],
        ["Time of Detection", challan.created_at.strftime("%H:%M:%S"),
         "System Reference", f"EVPS/{challan.id:06d}"],
    ]
    meta_col_w = [usable_w * 0.22, usable_w * 0.28, usable_w * 0.22, usable_w * 0.28]
    meta_table = Table(meta_rows, colWidths=meta_col_w)
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),  # label col 0
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),  # label col 2
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR", (2, 0), (2, -1), NAVY),
        ("BACKGROUND",(0, 0), (0, -1), GREY),
        ("BACKGROUND",(2, 0), (2, -1), GREY),
        ("GRID",      (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5 * mm))

    # ── 4. SECTION HEADING helper ─────────────────────────────────────────────
    def section_heading(text):
        row = [[Paragraph(text, _style("SH", fontName="Helvetica-Bold", fontSize=9,
                                       textColor=WHITE, alignment=TA_LEFT))]]
        t = Table(row, colWidths=[usable_w])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        return t

    def detail_row(label, value):
        return [
            Paragraph(label, _style("DL", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY)),
            Paragraph(str(value) if value else "—",
                      _style("DV", fontSize=9)),
        ]

    def detail_table(rows):
        t = Table(rows, colWidths=[usable_w * 0.35, usable_w * 0.65])
        t.setStyle(TableStyle([
            ("FONTSIZE",  (0, 0), (-1, -1), 9),
            ("GRID",      (0, 0), (-1, -1), 0.5, BORDER),
            ("BACKGROUND",(0, 0), (0, -1), GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, colors.HexColor("#F8F8F8")]),
        ]))
        return t

    # ── 5. VEHICLE DETAILS ────────────────────────────────────────────────────
    story.append(section_heading("VEHICLE DETAILS"))
    veh_rows = [
        detail_row("Registration Number",
                   Paragraph(f"<b>{challan.plate_number}</b>",
                              _style("Plate", fontName="Helvetica-Bold", fontSize=11,
                                     textColor=NAVY))),
        detail_row("Vehicle Type",
                   challan.vehicle.vehicle_type if challan.vehicle_id and challan.vehicle else "Not Registered"),
        detail_row("Registered Owner",
                   challan.vehicle.owner_name if challan.vehicle_id and challan.vehicle else "Not Registered"),
    ]
    story.append(detail_table(veh_rows))
    story.append(Spacer(1, 4 * mm))

    # ── 6. VIOLATION DETAILS ──────────────────────────────────────────────────
    story.append(section_heading("VIOLATION DETAILS"))
    vio_rows = [
        detail_row("Offence",       "Failure to yield right-of-way to an emergency vehicle"),
        detail_row("AI Action",     challan.action),
        detail_row("Traffic Density at Detection", f"{challan.density_pct:.1f}%"),
        detail_row("Free Space Detected",          f"{challan.free_space_px} pixels"),
        detail_row("System Decision Log",          challan.pipeline_status or "—"),
        detail_row("Detection System",             "EasyOCR Edge-Cloud Pipeline (Raspberry Pi)"),
    ]
    story.append(detail_table(vio_rows))
    story.append(Spacer(1, 4 * mm))

    # ── 7. PENALTY DETAILS ────────────────────────────────────────────────────
    story.append(section_heading("PENALTY DETAILS"))

    status_color = {"Paid": GREEN, "Disputed": DANGER}.get(challan.status, SAFFRON)
    pen_rows = [
        detail_row("Challan Amount",
                   Paragraph(f"<b>Rs. {challan.amount:.2f}</b>",
                              _style("Amt", fontName="Helvetica-Bold", fontSize=12,
                                     textColor=NAVY))),
        detail_row("Payment Status",
                   Paragraph(f"<b>{challan.status.upper()}</b>",
                              _style("Stat", fontName="Helvetica-Bold", fontSize=11,
                                     textColor=status_color))),
        detail_row("Payment Due Date",
                   challan.paid_at.strftime("%d %B %Y") if challan.paid_at else "Within 30 days of issue"),
    ]
    story.append(detail_table(pen_rows))
    story.append(Spacer(1, 5 * mm))

    # ── 8. NOTICE BOX ────────────────────────────────────────────────────────
    notice_text = (
        "IMPORTANT NOTICE: Non-payment of fine within the stipulated period may "
        "result in further legal proceedings under the Motor Vehicles Act, 1988. "
        "For disputes, file a written representation at the nearest Regional "
        "Transport Office within 15 days of the date of issue of this challan."
    )
    notice_data = [[Paragraph(notice_text, _style("Notice", fontSize=8, textColor=MUTED,
                                                   alignment=TA_LEFT))]]
    notice_table = Table(notice_data, colWidths=[usable_w])
    notice_table.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 1, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(notice_table)
    story.append(Spacer(1, 5 * mm))

    # ── 9. FOOTER ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=usable_w, thickness=1, color=BORDER))
    story.append(Spacer(1, 3 * mm))
    footer_data = [[
        Paragraph("Generated by: Emergency Vehicle Priority System (EVPS)",
                  _style("FL", fontSize=7, textColor=MUTED, alignment=TA_LEFT)),
        Paragraph(f"Generated on: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}",
                  _style("FR", fontSize=7, textColor=MUTED, alignment=TA_RIGHT)),
    ]]
    footer_table = Table(footer_data, colWidths=[usable_w * 0.6, usable_w * 0.4])
    footer_table.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_table)

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
