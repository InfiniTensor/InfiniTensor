"""Render the Chinese project report to an embedded-font, paginated PDF."""

import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline(text):
    text = escape(text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#176C83">\1</link>', text
    )
    text = re.sub(r"`([^`]+)`", r'<font color="#176C83">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # Droid's CJK fallback font intentionally has no Latin glyphs. Select an
    # embedded Latin font for Latin text while keeping Chinese in the CJK font.
    parts = re.split(r"(<[^>]+>)", text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"[\x20-\x7e\u00a0-\u024f\u2000-\u206f\u2190-\u21ff]+",
            r'<font name="Latin">\g<0></font>',
            parts[i],
        )
    return "".join(parts)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=repo / "docs/dynamic_shape/report.md"
    )
    parser.add_argument("--author", default="jnfkdsn")
    parser.add_argument("--pr", help="Actual GitHub PR URL to put on the cover")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    args = parser.parse_args()
    if not args.font.exists():
        parser.error("Provide a Chinese TrueType font through --font")
    pdfmetrics.registerFont(TTFont("CJK", str(args.font)))
    pdfmetrics.registerFontFamily(
        "CJK", normal="CJK", bold="CJK", italic="CJK", boldItalic="CJK"
    )
    pdfmetrics.registerFont(
        TTFont("Latin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    mono = "Courier"
    if font.exists():
        pdfmetrics.registerFont(TTFont("Mono", str(font)))
        mono = "Mono"
    text = args.source.read_text(encoding="utf-8")
    text = re.sub(r"项目提交人：[^\n]+", f"项目提交人：{args.author}", text)
    if args.pr:
        text = re.sub(r"GitHub PR：[^\n]+", f"GitHub PR：[{args.pr}]({args.pr})", text)
    args.source.write_text(text, encoding="utf-8")
    output = args.output or args.source.parent / (
        f"【2026夏季InfiniTensor训练营- AI编译器方向】{args.author}_ONNX动态Shape项目报告.pdf"
    )
    styles = {
        "body": ParagraphStyle(
            "body",
            fontName="CJK",
            fontSize=10,
            leading=16,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "title": ParagraphStyle(
            "title",
            fontName="CJK",
            fontSize=23,
            leading=34,
            textColor=colors.HexColor("#133F52"),
            spaceAfter=22,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="CJK",
            fontSize=15,
            leading=23,
            textColor=colors.HexColor("#133F52"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName="CJK",
            fontSize=11.5,
            leading=18,
            textColor=colors.HexColor("#176C83"),
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "code",
            fontName=mono,
            fontSize=7.5,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#F1F5F7"),
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=10,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="CJK", fontSize=8, leading=12, wordWrap="CJK"
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="CJK",
            fontSize=8,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
    }
    width = A4[0] - 94
    story, paragraph = [], []

    def flush():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
        elif line.startswith("<!-- pagebreak -->"):
            flush()
            story.append(PageBreak())
        elif line.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(
                Preformatted("\n".join(code), styles["code"], maxLineLength=95)
            )
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", cell) for cell in cells):
                    rows.append(
                        [Paragraph(inline(cell), styles["cell"]) for cell in cells]
                    )
                i += 1
            i -= 1
            count = len(rows[0])
            table = Table(
                rows, colWidths=[width / count] * count, repeatRows=1, hAlign="LEFT"
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEBF0")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#F5F8FA")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#91AEBB")),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
        elif line.startswith("!["):
            flush()
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                picture = Image(str(args.source.parent / match[2]))
                ratio = width / picture.imageWidth
                picture.drawWidth = width
                picture.drawHeight = picture.imageHeight * ratio
                story.extend([picture, Paragraph(inline(match[1]), styles["caption"])])
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            style = styles["title" if level == 1 else "h2" if level == 2 else "h3"]
            story.append(Paragraph(inline(line.lstrip("#").strip()), style))
        elif line.startswith("- "):
            flush()
            story.append(Paragraph(inline("• " + line[2:]), styles["body"]))
        else:
            paragraph.append(line)
        i += 1
    flush()

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B5C8D1"))
        canvas.line(47, 39, A4[0] - 47, 39)
        canvas.setFont("CJK", 8)
        canvas.setFillColor(colors.HexColor("#526E7C"))
        footer = Paragraph(
            inline("InfiniTensor · ONNX 动态 Shape 项目报告"), styles["caption"]
        )
        footer.wrap(300, 20)
        footer.drawOn(canvas, 30, 21)
        canvas.setFont("Latin", 8)
        canvas.drawRightString(A4[0] - 47, 26, str(doc.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=47,
        rightMargin=47,
        topMargin=43,
        bottomMargin=53,
        title="ONNX 动态 Shape 子图编译与执行支持",
        author=args.author,
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(output)


if __name__ == "__main__":
    main()
