import time
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

from tools.file_io import _has_s3_credentials, get_s3_client, get_s3_bucket_name


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(6, 182, 212)  # Cyan
        self.cell(0, 10, "Agentic RAG Engine - Technical Report", border=False, ln=True, align="L")
        self.set_draw_color(226, 232, 240)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")


def generate_markdown_report(
    title: str,
    query: str,
    answer: str,
    sources: list[dict] | None = None,
    steps: list[dict] | None = None,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        f"**Generated Date**: {now_str}",
        f"**Engine**: Agentic RAG (Pinecone • BGE Reranker • DeepSeek / OpenRouter)",
        "---",
        "",
        "## ❓ User Prompt / Question",
        f"> {query}",
        "",
        "## ✨ Answer Output",
        answer,
        "",
    ]

    if sources:
        lines.append("---")
        lines.append("## 🔍 Retrieved Knowledge Sources & Rerank Scores")
        lines.append("| Rank | Source Document | Page | BGE Rerank Score | Content Snippet |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for idx, src in enumerate(sources, start=1):
            doc = src.get("source") or "Doc"
            page = src.get("page") or "?"
            score = src.get("rerank_score", "N/A")
            snippet = (src.get("content") or "").replace("\n", " ")[:100]
            lines.append(f"| #{idx} | {doc} | {page} | {score} | {snippet}... |")
        lines.append("")

    if steps:
        lines.append("---")
        lines.append("## 🤖 Agent Execution Trace")
        for idx, step in enumerate(steps, start=1):
            tool = step.get("tool") or "Action"
            tool_input = step.get("tool_input")
            obs = str(step.get("observation", ""))[:200]
            lines.append(f"### Step {idx}: Tool `{tool}`")
            lines.append(f"- **Input**: `{tool_input}`")
            lines.append(f"- **Observation**: {obs}...")
            lines.append("")

    return "\n".join(lines)


def generate_pdf_report(
    title: str,
    query: str,
    answer: str,
    sources: list[dict] | None = None,
    steps: list[dict] | None = None,
) -> bytes:
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, title[:60], ln=True)

    # Prompt Box
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "User Prompt:", ln=True)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(0, 6, query.encode("latin-1", "replace").decode("latin-1"))
    pdf.ln(4)

    # Answer Box
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(0, 8, "Generated Answer:", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(15, 23, 42)
    clean_answer = answer.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, clean_answer)
    pdf.ln(6)

    # Sources
    if sources:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, "Retrieved Sources:", ln=True)

        for idx, src in enumerate(sources[:5], start=1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(6, 182, 212)
            doc_name = src.get("source") or "Document"
            score = src.get("rerank_score", "N/A")
            pdf.cell(0, 6, f"[{idx}] {doc_name} (Score: {score})", ln=True)

            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 116, 139)
            snippet = (src.get("content") or "").encode("latin-1", "replace").decode("latin-1")[:250]
            pdf.multi_cell(0, 5, f"{snippet}...")
            pdf.ln(2)

    return bytes(pdf.output())


def save_and_upload_report(filename: str, content_bytes: bytes, content_type: str) -> dict:
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    local_path = outputs_dir / filename
    local_path.write_bytes(content_bytes)

    abs_path = str(local_path.resolve())

    result = {
        "filename": filename,
        "local_path": abs_path,
        "size_bytes": len(content_bytes),
        "s3_uploaded": False,
    }

    if _has_s3_credentials():
        try:
            s3_client = get_s3_client()
            bucket_name = get_s3_bucket_name()
            s3_key = f"outputs/{filename}"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=content_bytes,
                ContentType=content_type,
            )
            result["s3_uploaded"] = True
            result["s3_key"] = s3_key
            result["s3_bucket"] = bucket_name
        except Exception as exc:
            print(f"Warning: S3 report upload failed: {exc}")

    return result
