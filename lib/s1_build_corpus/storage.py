"""
Storage adapters for streaming PDFs from local disk, S3, or Google Drive.

Each adapter yields (filename, local_path, is_temporary) tuples.
Temporary files (remote sources) are cleaned up by the caller after processing.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple, Optional

import boto3
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


PdfStream = Generator[Tuple[str, str, bool], None, None]


class StorageAdapter(ABC):
    @abstractmethod
    def stream_pdfs(self) -> PdfStream:
        """Yields (filename, local_path, is_temporary) for every PDF in the source."""


class LocalStorageAdapter(StorageAdapter):
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def stream_pdfs(self) -> PdfStream:
        for path in self.directory.rglob("*.pdf"):
            yield path.name, str(path), False


class S3StorageAdapter(StorageAdapter):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3")

    def stream_pdfs(self) -> PdfStream:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    filename = Path(key).name
                    tmp_path = _make_temp_pdf()
                    self.s3.download_file(self.bucket, key, tmp_path)
                    yield filename, tmp_path, True


class GoogleDriveAdapter(StorageAdapter):
    def __init__(self, folder_id: str):
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(gauth)
        self.folder_id = folder_id

    def stream_pdfs(self) -> PdfStream:
        query = (
            f"'{self.folder_id}' in parents "
            "and mimeType='application/pdf' "
            "and trashed=false"
        )
        for file in self.drive.ListFile({"q": query}).GetList():
            tmp_path = _make_temp_pdf()
            file.GetContentFile(tmp_path)
            yield file["title"], tmp_path, True


def _make_temp_pdf() -> str:
    """Creates a named temp file and returns its path. Caller is responsible for deletion."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    return tmp.name


def get_adapter(
    target: str,
    local_directory_path: Optional[str] = None,
    aws_bucket_name: Optional[str] = None,
    aws_prefix: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None,
) -> StorageAdapter:
    """Instantiate the correct adapter using passed configuration."""
    target = target.lower()

    if target in ("local", "windows"):
        return LocalStorageAdapter(local_directory_path or ".")
    if target == "s3":
        if not aws_bucket_name:
            raise ValueError("AWS_BUCKET_NAME must be provided for S3 storage target.")
        return S3StorageAdapter(
            bucket=aws_bucket_name,
            prefix=aws_prefix or "",
        )
    if target == "gdrive":
        if not gdrive_folder_id:
            raise ValueError("GDRIVE_FOLDER_ID must be provided for gdrive storage target.")
        return GoogleDriveAdapter(gdrive_folder_id)

    raise ValueError(
        f"Unknown STORAGE_TARGET '{target}'. Expected: local, s3, or gdrive."
    )

