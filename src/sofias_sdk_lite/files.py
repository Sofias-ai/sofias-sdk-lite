"""Files for third-party agents: the turn's attachments and the files you return.

Everything goes through the platform's intermediary with the turn's short-lived
grant (the same one your connectors use). Your agent never sees storage keys or
permanent links, and the grant expires when the turn does.

The task message carries what you need::

    def prepare_input(self, task, history, role):
        for attachment in task.attachment_details:      # what the user attached
            data = task.files.download(attachment)        # bytes
        saved = task.files.upload("report.csv", csv_bytes, "text/csv")
        answer = f"Here is the report.\\n\\n{saved.artifact()}"

``task.files`` is ``None`` when the platform has no intermediary for your agent
(for instance, a local run): check it before using it.

Images for a vision model::

    part = task.files.image_part(attachment)   # OpenAI-compatible content part
    messages = [{"role": "user", "content": [{"type": "text", "text": "What is it?"}, part]}]

Limits: 25 MB per file and 10 returned files per turn.
"""

from __future__ import annotations

import base64
import json
from typing import Any, ClassVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_FILE_BYTES",
    "Attachment",
    "FileAccess",
    "FileAccessError",
    "ReturnedFile",
]

#: The intermediary rejects bigger files; checked here too, before sending.
MAX_FILE_BYTES = 25 * 1024 * 1024


class FileAccessError(RuntimeError):
    """A download or upload the intermediary refused (status in ``status_code``)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class Attachment(BaseModel):
    """A file the user attached, as listed in ``task.attachment_details``."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    filename: str
    content_type: str | None = None
    byte_size: int | None = None
    url: str


class ReturnedFile(BaseModel):
    """A file you uploaded with ``FileAccess.upload``: return it with ``artifact()``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    filename: str
    content_type: str | None = None
    byte_size: int | None = None
    url: str | None = None

    def artifact(self) -> str:
        """The tag that hands this file to the user. Put it in your answer."""
        return "[ARTIFACT]\n" + json.dumps({"file_id": self.id}) + "\n[/ARTIFACT]"


class FileAccess(BaseModel):
    """Download the turn's attachments and upload the files you return.

    Built by the SDK from the ``files`` entry of the task message; the token is
    the turn's grant and never appears in ``repr`` or logs.
    """

    model_config = ConfigDict(extra="ignore")

    url: str
    token: str = Field(repr=False)
    timeout: float = 60.0

    #: Tests swap the HTTP transport here (``httpx.MockTransport``).
    _transport: ClassVar[httpx.BaseTransport | None] = None
    _async_transport: ClassVar[httpx.AsyncBaseTransport | None] = None

    # -- download -----------------------------------------------------------

    def download(self, attachment: Attachment | str) -> bytes:
        """The bytes of an attachment (or of an intermediary file URL)."""
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.get(_url_of(attachment), headers=self._headers())
        return _content(response, "download")

    async def adownload(self, attachment: Attachment | str) -> bytes:
        """Async ``download``."""
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._async_transport
        ) as client:
            response = await client.get(_url_of(attachment), headers=self._headers())
        return _content(response, "download")

    def image_part(self, attachment: Attachment) -> dict[str, Any]:
        """An OpenAI-compatible ``image_url`` content part for a vision model."""
        return _image_part(attachment, self.download(attachment))

    async def aimage_part(self, attachment: Attachment) -> dict[str, Any]:
        """Async ``image_part``."""
        return _image_part(attachment, await self.adownload(attachment))

    # -- upload -------------------------------------------------------------

    def upload(
        self, filename: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> ReturnedFile:
        """Store a file for this turn; return it to the user with ``.artifact()``."""
        _check_size(data)
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.post(
                self._upload_url(filename), content=data, headers=self._headers(content_type)
            )
        return _returned(response)

    async def aupload(
        self, filename: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> ReturnedFile:
        """Async ``upload``."""
        _check_size(data)
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._async_transport
        ) as client:
            response = await client.post(
                self._upload_url(filename), content=data, headers=self._headers(content_type)
            )
        return _returned(response)

    # -- internals ----------------------------------------------------------

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _upload_url(self, filename: str) -> str:
        return f"{self.url.rstrip('/')}?filename={quote(filename)}"


def _url_of(attachment: Attachment | str) -> str:
    return attachment if isinstance(attachment, str) else attachment.url


def _check_size(data: bytes) -> None:
    if not data:
        raise ValueError("cannot upload an empty file")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"file too large: {len(data)} bytes (limit {MAX_FILE_BYTES})")


def _content(response: httpx.Response, action: str) -> bytes:
    if response.status_code != 200:
        raise FileAccessError(
            f"file {action} failed (HTTP {response.status_code})", response.status_code
        )
    return response.content


def _returned(response: httpx.Response) -> ReturnedFile:
    if response.status_code != 201:
        raise FileAccessError(
            f"file upload failed (HTTP {response.status_code})", response.status_code
        )
    return ReturnedFile.model_validate(response.json())


def _image_part(attachment: Attachment, data: bytes) -> dict[str, Any]:
    content_type = attachment.content_type or "image/png"
    if not content_type.startswith("image/"):
        raise ValueError(f"{attachment.filename} is not an image ({content_type})")
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{encoded}"}}
