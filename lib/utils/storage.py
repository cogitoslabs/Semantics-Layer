"""
Storage adapters for streaming PDFs from local disk or S3.

Each adapter yields (filename, local_path, is_temporary) tuples.
Temporary files (remote sources) are cleaned up by the caller after processing.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple, Optional

import boto3

from .config import StorageConfig

DocStream = Generator[Tuple[str, str, bool], None, None]


class StorageAdapter(ABC):
    @abstractmethod
    def stream_documents(self) -> DocStream:
        """Yields (filename, local_path, is_temporary) for every supported document in the source."""


class LocalStorageAdapter(StorageAdapter):
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def stream_documents(self) -> DocStream:
        for path in self.directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".pdf", ".txt", ".html"):
                yield path.name, str(path), False


def make_temp_file(suffix: str) -> str:
    """Creates a named temp file with specific suffix and returns its path. Caller is responsible for deletion."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    return tmp.name


class S3StorageAdapter(StorageAdapter):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3")

    def stream_documents(self) -> DocStream:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                suffix = Path(key).suffix.lower()
                if suffix in (".pdf", ".txt", ".html"):
                    filename = Path(key).name
                    tmp_path = make_temp_file(suffix)
                    self.s3.download_file(self.bucket, key, tmp_path)
                    yield filename, tmp_path, True


def get_adapter(
    cfg: StorageConfig,
) -> StorageAdapter:
    """Instantiate the correct adapter using passed configuration."""
    target = cfg.storage_target.lower()

    if target in ("local", "windows"):
        return LocalStorageAdapter(cfg.local_directory_path or ".")
    if target == "s3":
        if not cfg.aws_bucket_name:
            raise ValueError("AWS_BUCKET_NAME must be provided for S3 storage target.")
        return S3StorageAdapter(
            bucket=cfg.aws_bucket_name,
            prefix=cfg.aws_prefix or "",
        )

    raise ValueError(
        f"Unknown STORAGE_TARGET '{target}'. Expected: local or s3."
    )
