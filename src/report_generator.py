from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

import pandas as pd


def generate_business_report(
    summary_text,
    output_path="business_report.pdf"
):

    doc = SimpleDocTemplate(output_path)

    styles = getSampleStyleSheet()

    content = []

    title = Paragraph(
        "AI Business Intelligence Report",
        styles["Title"]
    )

    content.append(title)

    content.append(Spacer(1, 20))

    summary = Paragraph(
        summary_text.replace("\n", "<br/>"),
        styles["BodyText"]
    )

    content.append(summary)

    doc.build(content)

    return output_path