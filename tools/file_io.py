import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from llama_parse import LlamaParse
from pydantic import BaseModel, Field, model_validator

load_dotenv()
_S3_CLIENT = None


def get_s3_bucket_name() -> str:
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise ValueError("Missing S3_BUCKET_NAME in .env")
    return bucket


def get_aws_region() -> str:
    return os.getenv("AWS_REGION", "ap-south-1").strip()


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _get_llamaparse_api_key() -> str:
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing LlamaParse API key. Set LLAMA_CLOUD_API_KEY (or LLAMAPARSE_API_KEY) in .env."
        )
    return api_key


def _parse_pdf_with_llamaparse(pdf_path: str) -> str:
    parser = LlamaParse(
        api_key=_get_llamaparse_api_key(),
        result_type=os.getenv("LLAMAPARSE_RESULT_TYPE", "markdown"),
    )
    parsed_docs = parser.load_data(pdf_path)
    texts: list[str] = []
    for parsed_doc in parsed_docs:
        text = (getattr(parsed_doc, "text", "") or "").strip()
        if text:
            texts.append(text)
    if not texts:
        raise ValueError("LlamaParse did not return any extractable text for this PDF.")
    return "\n\n---\n\n".join(texts)


def _verify_s3_object(bucket: str, key: str, expected_bytes: int) -> tuple[bool, str | None]:
    try:
        response = get_s3_client().head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error = (exc.response.get("Error") or {})
        code = str(error.get("Code") or "ClientError")
        message = str(error.get("Message") or str(exc))
        return False, f"{code}: {message}"

    content_length = response.get("ContentLength")
    if isinstance(content_length, int) and content_length != expected_bytes:
        return False, f"ContentLength mismatch (expected {expected_bytes}, got {content_length})"

    return True, None


def _has_s3_credentials() -> bool:
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    return bool(access_key and secret_key)


def _require_s3_credentials() -> tuple[str, str]:
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if not access_key or not secret_key:
        raise ValueError("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY in .env")
    return access_key, secret_key


def get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT

    access_key, secret_key = _require_s3_credentials()
    _S3_CLIENT = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=get_aws_region(),
        config=Config(
            connect_timeout=_get_int_env("S3_CONNECT_TIMEOUT_SECONDS", 10),
            read_timeout=_get_int_env("S3_READ_TIMEOUT_SECONDS", 30),
            retries={"max_attempts": _get_int_env("S3_MAX_ATTEMPTS", 3), "mode": "standard"},
        ),
    )
    return _S3_CLIENT


def _parse_s3_uri(path_or_key: str) -> tuple[str | None, str]:
    value = path_or_key.strip()
    if value.startswith("s3://"):
        parsed = urlparse(value)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError("Invalid S3 URI. Expected format: s3://bucket/key")
        return bucket, key
    return None, value


def _list_s3_keys(bucket: str, prefix: str, max_keys: int = 25) -> list[str]:
    prefix_value = prefix.strip().lstrip("/")
    if not prefix_value:
        return []

    try:
        response = get_s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix_value, MaxKeys=max_keys)
        contents = response.get("Contents") or []
        keys: list[str] = []
        for item in contents:
            key = item.get("Key")
            if isinstance(key, str) and key:
                keys.append(key)
        return keys
    except Exception:
        return []


def read_file(path_or_key: str, source: str = "auto", encoding: str = "utf-8") -> dict:
    mode = source.strip().lower()
    if mode not in {"auto", "local", "s3"}:
        raise ValueError("source must be one of: auto, local, s3")

    normalized_input = path_or_key.strip()
    if not normalized_input:
        raise ValueError("path_or_key is required")

    local_path = Path(normalized_input)
    is_local_pdf = local_path.suffix.lower() == ".pdf"
    if mode in {"auto", "local"} and local_path.exists() and local_path.is_dir():
        raise IsADirectoryError(f"Expected a file but found directory: {local_path}")

    if mode in {"auto", "local"} and local_path.exists() and local_path.is_file():
        if is_local_pdf:
            try:
                content = _parse_pdf_with_llamaparse(str(local_path))
                return {
                    "source": "local",
                    "path": str(local_path.resolve()),
                    "content": content,
                    "parsed_as": "pdf",
                }
            except Exception as exc:
                return {
                    "source": "local",
                    "path": str(local_path.resolve()),
                    "error": "PdfParseError",
                    "message": str(exc),
                }
        try:
            content = local_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            return {
                "source": "local",
                "path": str(local_path.resolve()),
                "error": "DecodeError",
                "message": str(exc),
                "hint": (
                    "The file is not valid UTF-8. Try specifying a different encoding "
                    "(e.g., latin-1, cp1252, utf-16) or convert the file to UTF-8."
                ),
            }
        return {
            "source": "local",
            "path": str(local_path.resolve()),
            "content": content,
            "parsed_as": "text",
        }

    if mode == "local":
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if mode == "auto" and not _has_s3_credentials():
        raise FileNotFoundError(
            f"Local file not found: '{normalized_input}' and no AWS credentials configured for S3 fallback."
        )

    bucket_from_uri, key = _parse_s3_uri(normalized_input)
    bucket = bucket_from_uri or get_s3_bucket_name()
    key = key.lstrip("/")
    is_s3_pdf = key.lower().endswith(".pdf")

    try:
        response = get_s3_client().get_object(Bucket=bucket, Key=key)
        raw_bytes = response["Body"].read()
        if is_s3_pdf:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(raw_bytes)
                    temp_path = temp_file.name
                content = _parse_pdf_with_llamaparse(temp_path)
                return {
                    "source": "s3",
                    "bucket": bucket,
                    "key": key,
                    "content": content,
                    "parsed_as": "pdf",
                }
            except Exception as exc:
                return {
                    "source": "s3",
                    "bucket": bucket,
                    "key": key,
                    "error": "PdfParseError",
                    "message": str(exc),
                }
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
        try:
            content = raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            return {
                "source": "s3",
                "bucket": bucket,
                "key": key,
                "error": "DecodeError",
                "message": str(exc),
                "hint": (
                    "The S3 object is not valid UTF-8. Try specifying a different encoding "
                    "(e.g., latin-1, cp1252, utf-16) or upload a UTF-8 encoded text file."
                ),
            }
        return {
            "source": "s3",
            "bucket": bucket,
            "key": key,
            "content": content,
            "parsed_as": "text",
        }
    except ClientError as exc:
        error = (exc.response.get("Error") or {})
        code = str(error.get("Code") or "ClientError")
        message = str(error.get("Message") or str(exc))

        prefix = ""
        if "/" in key:
            prefix = key.rsplit("/", 1)[0] + "/"

        suggestions = []
        if code in {"NoSuchKey", "404", "NotFound"} and prefix:
            suggestions = _list_s3_keys(bucket=bucket, prefix=prefix)

        return {
            "source": "s3",
            "bucket": bucket,
            "key": key,
            "error": code,
            "message": message,
            "hint": (
                "S3 object not found. Keys are case-sensitive and must match exactly. "
                "Verify the object exists at s3://{bucket}/{key} or upload it first."
            ).format(bucket=bucket, key=key),
            "nearby_keys": suggestions,
        }


def write_file(
    content: str,
    s3_key: str | None = None,
    content_type: str = "text/plain",
    local_path: str | None = None,
    encoding: str = "utf-8",
) -> dict:
    content_value = content
    key_value = (s3_key or "").strip().lstrip("/")
    content_type_value = content_type
    local_path_value = local_path
    encoding_value = encoding

    if not key_value:
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                parsed_content = parsed.get("content")
                if isinstance(parsed_content, str):
                    content_value = parsed_content

                parsed_s3_key = parsed.get("s3_key")
                if isinstance(parsed_s3_key, str):
                    key_value = parsed_s3_key.strip().lstrip("/")

                parsed_content_type = parsed.get("content_type")
                if isinstance(parsed_content_type, str) and parsed_content_type.strip():
                    content_type_value = parsed_content_type

                parsed_local_path = parsed.get("local_path")
                if parsed_local_path is None or isinstance(parsed_local_path, str):
                    local_path_value = parsed_local_path

                parsed_encoding = parsed.get("encoding")
                if isinstance(parsed_encoding, str) and parsed_encoding.strip():
                    encoding_value = parsed_encoding

    if not key_value or key_value in {"output.txt", "summary.txt", "outputs/summary.txt"}:
        # Generate a descriptive filename from content preview
        first_line = content_value.strip().split("\n")[0][:40]
        slug = "".join(c if c.isalnum() else "_" for c in first_line).strip("_").lower()
        if not slug:
            slug = "agent_summary"
        key_value = f"outputs/{slug}.txt"

    if content_value == "":
        content_value = "No content provided."

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(key_value).name
    local_file_path = outputs_dir / filename
    local_file_path.write_text(content_value, encoding=encoding_value)
    abs_local_path = str(local_file_path.resolve())


    payload = content_value.encode(encoding_value)

    if not _has_s3_credentials():
        print(f"AWS S3 not configured; saved file locally to '{abs_local_path}'")
        return {
            "source": "local",
            "local_path": abs_local_path,
            "bytes_written": len(payload),
            "message": f"File successfully saved locally to {abs_local_path}",
        }

    try:
        bucket = get_s3_bucket_name()
        get_s3_client().put_object(
            Bucket=bucket,
            Key=key_value,
            Body=payload,
            ContentType=content_type_value,
        )
        return {
            "source": "s3",
            "bucket": bucket,
            "key": key_value,
            "bytes_written": len(payload),
            "local_path": abs_local_path,
            "message": f"File saved to S3 and locally to {abs_local_path}",
        }
    except Exception as exc:
        print(f"S3 upload failed ({exc}); falling back to local file: '{abs_local_path}'")
        return {
            "source": "local_fallback",
            "local_path": abs_local_path,
            "bytes_written": len(payload),
            "warning": f"S3 error ({exc}); saved locally instead.",
            "message": f"File successfully saved locally to {abs_local_path}",
        }



class ReadFileInput(BaseModel):
    path_or_key: str = Field(..., description="Local file path, S3 key, or s3://bucket/key")
    source: str = Field(default="auto", description="auto, local, or s3")
    encoding: str = Field(default="utf-8", description="Text encoding")


class WriteFileInput(BaseModel):
    content: str = Field(..., description="Text content to save")
    s3_key: str = Field(default="outputs/summary.txt", description="S3 object key (e.g. outputs/summary.txt)")
    content_type: str = Field(default="text/plain", description="S3 content type")
    local_path: str | None = Field(default=None, description="Optional local backup file path")
    encoding: str = Field(default="utf-8", description="Text encoding")


    @model_validator(mode="before")
    @classmethod
    def _recover_from_nested_json_string(cls, data):
        if not isinstance(data, dict):
            return data

        if data.get("s3_key"):
            return data

        raw_content = data.get("content")
        if not isinstance(raw_content, str):
            return data

        stripped = raw_content.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return data

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return data

        if not isinstance(parsed, dict):
            return data

        recovered = dict(data)
        parsed_content = parsed.get("content")
        if isinstance(parsed_content, str) and parsed_content.strip():
            recovered["content"] = parsed_content

        for key in ("content", "s3_key", "content_type", "local_path", "encoding"):
            value = recovered.get(key)
            if (value is None or value == "") and key in parsed:
                recovered[key] = parsed[key]
        return recovered


def get_read_file_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="read_file",
        description=(
            "Read a text file from local storage or S3. "
            "Accepts local path, S3 key, or s3://bucket/key."
        ),
        func=read_file,
        args_schema=ReadFileInput,
    )


def get_write_file_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="write_file",
        description="Write text content to S3 using boto3. Optionally save a local backup file.",
        func=write_file,
        args_schema=WriteFileInput,
    )
