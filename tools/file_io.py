import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

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


def read_file(path_or_key: str, source: str = "auto", encoding: str = "utf-8") -> dict:
    mode = source.strip().lower()
    if mode not in {"auto", "local", "s3"}:
        raise ValueError("source must be one of: auto, local, s3")

    normalized_input = path_or_key.strip()
    if not normalized_input:
        raise ValueError("path_or_key is required")

    local_path = Path(normalized_input)
    if mode in {"auto", "local"} and local_path.exists() and local_path.is_dir():
        raise IsADirectoryError(f"Expected a file but found directory: {local_path}")

    if mode in {"auto", "local"} and local_path.exists() and local_path.is_file():
        content = local_path.read_text(encoding=encoding)
        return {
            "source": "local",
            "path": str(local_path.resolve()),
            "content": content,
        }

    if mode == "local":
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if mode == "auto" and not _has_s3_credentials():
        raise FileNotFoundError(
            f"Local file not found: '{normalized_input}' and no AWS credentials configured for S3 fallback."
        )

    bucket_from_uri, key = _parse_s3_uri(normalized_input)
    bucket = bucket_from_uri or get_s3_bucket_name()
    response = get_s3_client().get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode(encoding)
    return {
        "source": "s3",
        "bucket": bucket,
        "key": key,
        "content": content,
    }


def write_file(
    content: str,
    s3_key: str,
    content_type: str = "text/plain",
    local_path: str | None = None,
    encoding: str = "utf-8",
) -> dict:
    key_value = s3_key.strip()
    if not key_value:
        raise ValueError("s3_key is required")
    if content == "":
        raise ValueError("content cannot be empty for write_file")

    payload = content.encode(encoding)
    bucket = get_s3_bucket_name()
    get_s3_client().put_object(
        Bucket=bucket,
        Key=key_value,
        Body=payload,
        ContentType=content_type,
    )

    local_written = None
    if local_path:
        output = Path(local_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding=encoding)
        local_written = str(output.resolve())

    return {
        "bucket": bucket,
        "key": key_value,
        "bytes_written": len(payload),
        "local_path": local_written,
    }


class ReadFileInput(BaseModel):
    path_or_key: str = Field(..., description="Local file path, S3 key, or s3://bucket/key")
    source: str = Field(default="auto", description="auto, local, or s3")
    encoding: str = Field(default="utf-8", description="Text encoding")


class WriteFileInput(BaseModel):
    content: str = Field(..., description="Text content to save")
    s3_key: str = Field(..., description="S3 object key (e.g. outputs/summary.txt)")
    content_type: str = Field(default="text/plain", description="S3 content type")
    local_path: str | None = Field(default=None, description="Optional local backup file path")
    encoding: str = Field(default="utf-8", description="Text encoding")


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
