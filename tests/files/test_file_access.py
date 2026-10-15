"""Files through the intermediary: download with the turn's grant, upload and
return with ``[ARTIFACT]``, images for a vision model, and the token kept out of
``repr``."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from sofias_sdk_lite import AgentTaskMessage, Attachment, FileAccess, FileAccessError, ReturnedFile

BASE = "https://intermediary.example/files"


@pytest.fixture
def requests_seen(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET" and request.url.path.endswith("/files/abc"):
            return httpx.Response(200, content=b"%PDF-1.4")
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(
            201,
            json={
                "id": "f-1",
                "filename": request.url.params["filename"],
                "content_type": "text/csv",
                "byte_size": len(request.content),
                "url": f"{BASE}/f-1",
            },
        )

    monkeypatch.setattr(FileAccess, "_transport", httpx.MockTransport(handler))
    return seen


def access() -> FileAccess:
    return FileAccess(url=BASE, token="grant-secret")


def attachment(**kwargs) -> Attachment:
    data = {
        "id": "abc",
        "filename": "informe.pdf",
        "content_type": "application/pdf",
        "url": f"{BASE}/abc",
    }
    return Attachment(**{**data, **kwargs})


class TestTaskMessage:
    def test_reads_attachments_and_file_access_from_the_wire(self) -> None:
        task = AgentTaskMessage.model_validate(
            {
                "content": "mira esto",
                "agent": "dev.x",
                "tenant_identifier": "t",
                "conversation_id": "c",
                "message_id": 1,
                "attachments": [f"{BASE}/abc"],
                "attachment_details": [
                    {
                        "id": "abc",
                        "filename": "informe.pdf",
                        "content_type": "application/pdf",
                        "byte_size": 8,
                        "url": f"{BASE}/abc",
                    }
                ],
                "files": {"url": BASE, "token": "grant-secret"},
            }
        )

        assert task.attachment_details[0].filename == "informe.pdf"
        assert task.files is not None and task.files.url == BASE

    def test_without_intermediary_there_is_no_file_access(self) -> None:
        task = AgentTaskMessage(
            content="hola", agent="dev.x", tenant_identifier="t", conversation_id="c", message_id=1
        )

        assert task.files is None
        assert task.attachment_details == []


class TestDownload:
    def test_downloads_with_the_turn_grant(self, requests_seen: list[httpx.Request]) -> None:
        assert access().download(attachment()) == b"%PDF-1.4"
        assert requests_seen[0].headers["Authorization"] == "Bearer grant-secret"

    def test_raises_with_the_status_when_refused(self, requests_seen: list[httpx.Request]) -> None:
        with pytest.raises(FileAccessError) as error:
            access().download(f"{BASE}/otro")

        assert error.value.status_code == 404

    def test_builds_an_image_part_for_a_vision_model(
        self, requests_seen: list[httpx.Request]
    ) -> None:
        part = access().image_part(attachment(filename="foto.png", content_type="image/png"))

        assert part["type"] == "image_url"
        assert (
            part["image_url"]["url"]
            == "data:image/png;base64," + base64.b64encode(b"%PDF-1.4").decode()
        )

    def test_refuses_an_image_part_for_a_non_image(
        self, requests_seen: list[httpx.Request]
    ) -> None:
        with pytest.raises(ValueError, match="not an image"):
            access().image_part(attachment())


class TestUpload:
    def test_uploads_the_raw_file_and_returns_its_id(
        self, requests_seen: list[httpx.Request]
    ) -> None:
        saved = access().upload("resultado final.csv", b"a,b\n", "text/csv")

        request = requests_seen[0]
        assert request.method == "POST"
        assert request.url.params["filename"] == "resultado final.csv"
        assert request.headers["Content-Type"] == "text/csv"
        assert request.content == b"a,b\n"
        assert saved == ReturnedFile(
            id="f-1",
            filename="resultado final.csv",
            content_type="text/csv",
            byte_size=4,
            url=f"{BASE}/f-1",
        )

    def test_returns_the_file_with_an_artifact_tag(self) -> None:
        tag = ReturnedFile(id="f-1", filename="x.csv").artifact()

        assert tag.startswith("[ARTIFACT]\n") and tag.endswith("\n[/ARTIFACT]")
        assert json.loads(tag.splitlines()[1]) == {"file_id": "f-1"}

    def test_refuses_an_empty_file_before_sending(self, requests_seen: list[httpx.Request]) -> None:
        with pytest.raises(ValueError, match="empty"):
            access().upload("x.txt", b"")

        assert requests_seen == []

    def test_refuses_a_file_over_the_limit_before_sending(
        self, requests_seen: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sofias_sdk_lite.files.MAX_FILE_BYTES", 3)

        with pytest.raises(ValueError, match="too large"):
            access().upload("x.txt", b"abcd")

        assert requests_seen == []


class TestAsync:
    async def test_async_download_and_upload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, content=b"data")
            return httpx.Response(201, json={"id": "f-2", "filename": "y.txt"})

        monkeypatch.setattr(FileAccess, "_async_transport", httpx.MockTransport(handler))

        assert await access().adownload(attachment()) == b"data"
        assert (await access().aupload("y.txt", b"y")).id == "f-2"


def test_the_grant_never_shows_in_repr() -> None:
    assert "grant-secret" not in repr(access())
