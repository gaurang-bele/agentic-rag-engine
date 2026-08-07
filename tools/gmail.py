import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

load_dotenv(override=True)


def get_smtp_server() -> str:
    load_dotenv(override=True)
    return os.getenv("SMTP_SERVER", "smtp.gmail.com").strip()


def get_smtp_port() -> int:
    load_dotenv(override=True)
    try:
        return int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        return 587


def get_sender_email() -> str:
    load_dotenv(override=True)
    raw = os.getenv("GMAIL_SENDER_EMAIL", "").strip()
    return raw.strip("'\"")


def get_app_password() -> str:
    load_dotenv(override=True)
    raw = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    return raw.strip("'\"").replace(" ", "")


def send_email(
    recipient: str = "",
    subject: str = "Agentic RAG Engine Notification",
    body: str = "Report content enclosed.",
    attachment_path: str | None = None,
) -> dict:
    recipient_value = (recipient or "").strip()
    subject_value = (subject or "Agentic RAG Engine Notification").strip()
    body_value = (body or "Report content enclosed.").strip()
    attachment_value = (attachment_path or "").strip()

    sender = get_sender_email()
    password = get_app_password()

    if not recipient_value:
        recipient_value = sender or "user@example.com"

    # If Gmail credentials are set in .env, perform live SMTP email dispatch
    if sender and password:
        try:
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = recipient_value
            msg["Subject"] = subject_value

            msg.attach(MIMEText(body_value, "plain", "utf-8"))

            attached_file_name = None
            if attachment_value:
                file_path = Path(attachment_value)
                if not file_path.exists():
                    alt_path = Path("outputs") / file_path.name
                    if alt_path.exists():
                        file_path = alt_path

                if file_path.exists() and file_path.is_file():
                    file_bytes = file_path.read_bytes()
                    part = MIMEApplication(file_bytes, Name=file_path.name)
                    part["Content-Disposition"] = f'attachment; filename="{file_path.name}"'
                    msg.attach(part)
                    attached_file_name = file_path.name

            server_name = get_smtp_server()
            port = get_smtp_port()

            if port == 465:
                with smtplib.SMTP_SSL(server_name, port, timeout=15) as server:
                    server.login(sender, password)
                    server.sendmail(sender, recipient_value, msg.as_string())
            else:
                with smtplib.SMTP(server_name, port, timeout=15) as server:
                    server.starttls()
                    server.login(sender, password)
                    server.sendmail(sender, recipient_value, msg.as_string())

            msg_str = f"Email successfully sent to {recipient_value} via Gmail SMTP."
            if attached_file_name:
                msg_str += f" (Attached file: {attached_file_name})"

            return {
                "status": "success",
                "message": msg_str,
                "recipient": recipient_value,
                "subject": subject_value,
                "attachment": attached_file_name,
                "mode": "live_smtp",
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"SMTP dispatch error ({exc}). Ensure GMAIL_APP_PASSWORD is an App Password and SMTP_PORT is correct.",
                "recipient": recipient_value,
                "subject": subject_value,
                "mode": "error",
            }


    # Simulation fallback if GMAIL credentials not configured in .env
    return {
        "status": "success",
        "message": (
            f"[Simulated Dispatch] Email to '{recipient_value}' with subject '{subject_value}' "
            f"processed successfully. Add GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD to .env for live SMTP delivery."
        ),
        "recipient": recipient_value,
        "subject": subject_value,
        "mode": "simulated",
    }


import json
from pydantic import BaseModel, Field, model_validator


class SendEmailInput(BaseModel):
    recipient: str = Field(default="", description="Target recipient email address (e.g. user@example.com)")
    subject: str = Field(default="Agentic RAG Engine Notification", description="Email subject line")
    body: str = Field(default="Report content enclosed.", description="Email body text message or report content")
    attachment_path: str | None = Field(default=None, description="Optional local file path or output file name to attach (e.g. outputs/ai_2026.txt or outputs/report.pdf)")


    @model_validator(mode="before")
    @classmethod
    def _recover_from_nested_json_string(cls, data):
        if not isinstance(data, dict):
            return data

        raw_recipient = data.get("recipient")
        if not isinstance(raw_recipient, str):
            return data

        stripped = raw_recipient.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return data

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return data

        if not isinstance(parsed, dict):
            return data

        recovered = dict(data)
        for key in ("recipient", "subject", "body"):
            if key in parsed and parsed[key]:
                recovered[key] = parsed[key]

        return recovered



def get_send_email_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="send_email",
        description=(
            "Send an email message, report summary, or notification to a recipient email address. "
            "Requires recipient, subject, and body text."
        ),
        func=send_email,
        args_schema=SendEmailInput,
    )
