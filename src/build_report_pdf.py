#!/usr/bin/env python3
"""
src/build_report_pdf.py
Generates writeup/REPORT.pdf with:
- Exactly 2 pages of writing (body font >= 10pt, margins >= 0.75 in).
- Tables 1, 2a, 2b, and 3 placed on clearly labelled pages after the writing.
- Table 3 and Table 2b aligned to identical margins (width 492 pt inside printable frame).
- Font sizes in tables >= 8.0 pt.
- Zero em dashes in headers, footers, and text.
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_MD_PATH = os.path.join(BASE_DIR, "writeup", "REPORT.md")
TABLES_MD_PATH = os.path.join(BASE_DIR, "writeup", "TABLES.md")
PDF_OUTPUT_PATH = os.path.join(BASE_DIR, "writeup", "REPORT.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header (pages > 1) - no em dash!
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "The Relay: Technical Report, Rahul Teja Nalla")
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
        # Footer - no em dash!
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 34, page_text)
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        self.restoreState()

def build_pdf():
    # Page setup: letter is 612 x 792 pt. 0.75 in = 54 pt margins.
    # Frame printable width = 612 - 54*2 - 6*2 = 492 pt.
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=5
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    # Body font >= 10.0 pt
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.0,
        leading=12.2,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=2.5
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2
    )

    # Table styles: font >= 8.0 pt
    table_header_style = ParagraphStyle(
        "THStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=9.5,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TDStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=9.5,
        textColor=colors.HexColor("#1e293b")
    )

    table_title_style = ParagraphStyle(
        "TTableTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    table_note_style = ParagraphStyle(
        "TNoteStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.HexColor("#334155"),
        spaceBefore=3
    )

    story = []

    # Title
    story.append(Paragraph("The Relay: Production RAG Technical Report", title_style))
    story.append(Spacer(1, 2))

    # Read REPORT.md
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Split into sections by ##
    sections = re.split(r"\n(?=## )", md_text)

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        lines = sec.split("\n")
        header_line = lines[0].strip()
        if header_line.startswith("# "):
            continue
        header_text = header_line.lstrip("#").strip()
        story.append(Paragraph(header_text, h1_style))

        body_lines = lines[1:]
        current_p = []
        for bline in body_lines:
            bline_s = bline.strip()
            if not bline_s:
                if current_p:
                    text_p = " ".join(current_p)
                    text_p = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text_p)
                    text_p = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text_p)
                    text_p = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text_p)
                    story.append(Paragraph(text_p, body_style))
                    current_p = []
            elif bline_s.startswith("- ") or re.match(r"^\d+\.\s", bline_s):
                if current_p:
                    text_p = " ".join(current_p)
                    text_p = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text_p)
                    text_p = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text_p)
                    story.append(Paragraph(text_p, body_style))
                    current_p = []
                item_text = bline_s
                item_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", item_text)
                item_text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", item_text)
                story.append(Paragraph(item_text, bullet_style))
            else:
                current_p.append(bline_s)
        if current_p:
            text_p = " ".join(current_p)
            text_p = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text_p)
            text_p = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text_p)
            story.append(Paragraph(text_p, body_style))

    # PageBreak after Writing Section -> Writing is on Pages 1 & 2
    story.append(PageBreak())

    # =========================================================================
    # TABLES SECTION (On clearly labelled pages after the writing)
    # Width of all tables strictly equals 492 pt, aligned left with headings
    # =========================================================================

    def format_table_cells(headers, rows, col_widths):
        table_data = []
        h_row = [Paragraph(f"<b>{h}</b>", table_header_style) for h in headers]
        table_data.append(h_row)
        for r in rows:
            row_data = [Paragraph(str(c), table_cell_style) for c in r]
            table_data.append(row_data)

        t = Table(table_data, colWidths=col_widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        return t

    # --- Page 3: Table 1 & Table 2a (Sum = 492 pt) ---
    story.append(Paragraph("Benchmark Evidence & Operational Tables", title_style))
    story.append(Paragraph("Table 1: Limits you found in the real service", table_title_style))
    t1_headers = ["Limit", "Value you found", "How you found it", "Confidence"]
    t1_rows = [
        ["Requests per minute", "Not measured", "Empirically tested; binding limits are TPM/OTPM. Docs cite 30 RPM for other models.", "Low"],
        ["Tokens per minute (Total TPM)", "8,000 TPM", "Live HTTP response header: x-ratelimit-limit-tokens: 8000 (probe_raw.jsonl)", "High"],
        ["Output tokens per min (OTPM)", "1,000 OTPM", "Live HTTP 429 body: 'Limit 1000, Used 970, Requested 120' (probe_raw.jsonl)", "High"],
        ["Requests per day (RPD)", "1,000 req/day", "Live response header x-ratelimit-limit-requests: 1000 (probe_raw.jsonl)", "High"],
        ["Tokens per day (TPD)", "200,000 tok/day", "Live HTTP 429 body: 'Limit 200000, Used 199180, Requested 2032'. Rolling leaky bucket (~2.3 tok/s), NOT midnight.", "High"],
        ["Requests in flight at once", ">= 4 concurrent", "Empirically tested 4 simultaneous requests; all 200 OK in ~210ms (retry_experiments.jsonl).", "Medium"],
        ["What it returns on limit", "HTTP 429", "JSON body with type: tokens, code: rate_limit_exceeded (probe_raw.jsonl)", "High"],
        ["Time until it recovers", "3 to 6 s (median 6s), up to 12s", "probe_raw.jsonl shows retry-after of 3 to 6 s (median 6 s) and retry_experiments.jsonl shows waits up to 12 s", "High"]
    ]
    # Sum: 92 + 72 + 274 + 54 = 492 pt
    story.append(format_table_cells(t1_headers, t1_rows, [92, 72, 274, 54]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Table 2a: Latency, one row per load scenario (status=ok rows only)", table_title_style))
    t2a_headers = ["Scenario", "Type", "Reqs", "Dur (s)", "Offered Rate", "First token p50/p95/p99 (ms)", "Total time p50/p95/p99 (ms)"]
    t2a_rows = [
        ["steady_real", "real", "20", "382.7", "3.1/min (capacity ~3.0)", "638.4 / 1108.8 / 1247.0", "1045.0 / 1613.5 / 1614.8"],
        ["probe_pipeline", "real", "12", "144.6", "5.0/min", "968.1 / 7348.4 / 7433.5", "2369.5 / 7660.4 / 7804.6"],
        ["steady", "stand-in", "12", "66.6", "2.7 virt (10.8 real)", "276.9 / 337.0 / 356.0", "577.1 / 612.9 / 622.6"],
        ["burst", "stand-in", "20", "1.9", "158.7 virt (634.7 real)", "581.3 / 933.9 / 1028.5", "888.8 / 1241.4 / 1336.0"],
        ["sustained_overload", "stand-in", "33", "45.1", "11.0 virt (43.9 real)", "259.7 / 340.7 / 364.3", "567.2 / 648.2 / 671.8"],
        ["limit_halved", "stand-in", "12", "66.6", "2.7 virt (10.8 real)", "275.5 / 333.9 / 347.7", "577.2 / 601.2 / 607.8"],
        ["daily_budget_exhausted", "stand-in", "10", "1.0", "150.0 virt (600 real)", "n/a", "n/a"],
        ["daily_budget_exhausted_real", "real", "1", "1.0", "60.0/min", "n/a", "n/a"]
    ]
    # Sum: 80 + 38 + 26 + 36 + 94 + 109 + 109 = 492 pt
    story.append(format_table_cells(t2a_headers, t2a_rows, [80, 38, 26, 36, 94, 109, 109]))

    # --- Page 4: Table 2b & Table 3 (Sum = 492 pt, aligned left with headings) ---
    story.append(PageBreak())
    story.append(Paragraph("Table 2b: Outcomes, same rows", table_title_style))
    t2b_headers = ["Scenario", "Succ %", "429 %", "Drop %", "Drop Wait p50", "Graceful reply p50", "Retries per req", "Peak Q", "Tokens (in/out/tot)", "Cost/1k Ans"]
    t2b_rows = [
        ["steady_real", "35.0%", "65.0%", "0.0%", "-", "-", "0.65", "1", "2129 / 54 / 2183", "$1.918"],
        ["probe_pipeline", "66.7%", "33.3%", "0.0%", "-", "-", "1.0", "1", "2078 / 104 / 2182", "$2.078"],
        ["steady", "100.0%", "0.0%", "0.0%", "-", "-", "0.0", "1", "2164 / 37 / 2200", "$1.878"],
        ["burst", "45.0%", "0.0%", "55.0%", "30.9 ms", "-", "0.0", "20", "2097 / 38 / 2135", "$1.830"],
        ["sustained_overload", "36.4%", "0.0%", "63.6%", "27.5 ms", "-", "0.0", "23", "2156 / 34 / 2190", "$1.862"],
        ["limit_halved", "66.7%", "0.0%", "33.3%", "27.2 ms", "-", "0.0", "1", "2156 / 34 / 2191", "$1.863"],
        ["daily_budget_exhausted", "0.0%", "100.0%", "0.0%", "-", "46.0 ms", "0.0", "1", "0 / 0 / 0", "n/a"],
        ["daily_budget_exhausted_real", "0.0%", "100.0%", "0.0%", "-", "65.0 ms", "0.0", "1", "0 / 0 / 0", "n/a"]
    ]
    # Sum: 78 + 33 + 33 + 33 + 44 + 52 + 44 + 30 + 95 + 50 = 492 pt
    story.append(format_table_cells(t2b_headers, t2b_rows, [78, 33, 33, 33, 44, 52, 44, 30, 95, 50]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Table 3: Retrieval settings (Common n=15 benchmark cohort)", table_title_style))
    t3_headers = ["Setting", "n", "Correct %", "Wrong %", "Refused OK %", "Refused Wrong %", "Avg In", "Avg Out", "Total Tok", "First Token p50/p95", "Total p50/p95"]
    t3_rows = [
        ["small_k3_norerank", "15", "53.3%", "6.7%", "20.0%", "20.0%", "896.5", "72.9", "969.4", "766.5 / 3456.1 ms", "1438.7 / 4082.4 ms"],
        ["medium_k4_norerank", "15", "53.3%", "0.0%", "20.0%", "26.7%", "2195.5", "66.0", "2261.5", "947.8 / 1432.3 ms", "1318.0 / 2336.6 ms"],
        ["large_k3_rerank", "15", "46.7%", "6.7%", "20.0%", "26.7%", "2331.7", "80.9", "2412.6", "898.5 / 1700.2 ms", "1120.0 / 2269.8 ms"]
    ]
    # Sum: 76 + 16 + 35 + 34 + 44 + 47 + 33 + 33 + 36 + 69 + 69 = 492 pt (identical width & alignment to Table 2b)
    story.append(format_table_cells(t3_headers, t3_rows, [76, 16, 35, 34, 44, 47, 33, 33, 36, 69, 69]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Note on Setting 3</i>: Questions Q16–Q20 in Setting 3 were not answered because heavy context (2,331.7 input tokens) "
        "exhausted the real 200,000 tokens/day limit. Comparison is strictly over common n=15 questions.<br/>"
        "<i>Full 20-question benchmark (Settings 1 & 2)</i>: Setting 1 achieved 40.0% Correct, 5.0% Wrong, 35.0% Refused OK (894.7 in / 67.0 out). "
        "Setting 2 achieved 40.0% Correct, 0.0% Wrong, 35.0% Refused OK (2,209.3 in / 66.5 out). Parity achieved at 40.5% prompt token cost.",
        table_note_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build complete:", PDF_OUTPUT_PATH)

if __name__ == "__main__":
    build_pdf()
