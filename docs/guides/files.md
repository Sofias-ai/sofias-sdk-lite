# Files and images

Your agent can read what the user attaches and hand files back, and send images
to a vision model. Everything goes through the platform's intermediary with the
turn's short-lived grant, the same one your connectors use: your agent never
sees a storage key or a permanent link, and the grant expires with the turn.

Install version 0.4.0 or later: `pip install "sofias-sdk-lite>=0.4.0"`.

## What your agent receives

On every turn, the task message carries:

- `attachment_details`: what the user attached (`filename`, `content_type`,
  `byte_size`, `url`), as `Attachment` objects. The previous turns' attachments
  are in `messages` (the history), with the same kind of URL.
- `files`: a `FileAccess` to download those attachments and upload the files you
  return. It is `None` when the platform sends no intermediary, for instance in a
  local run: check it before using it.

To receive attachments, enable "Can receive file attachments" in the portal.

## Reading an attachment

```python
def prepare_input(self, task, history, role):
    texts = []
    for attachment in task.attachment_details:
        data = task.files.download(attachment)          # bytes
        if attachment.content_type == "text/csv":
            texts.append(data.decode("utf-8"))
    ...
```

`adownload` is the async version. A refused download raises `FileAccessError`
with the HTTP status in `status_code` (for instance, 401 once the turn's grant
has expired).

## Returning a file

Upload it with the turn's grant, then put its tag in your answer:

```python
saved = task.files.upload("report.csv", csv_bytes, "text/csv")
answer = f"Here is the report.\n\n{saved.artifact()}"
```

The platform replaces the tag with the file, as a download in the user's chat.
It only accepts files uploaded (or attached) in a turn of the same conversation
and by the same agent. Limits: 25 MB per file and 10 returned files per turn.

## Images for a vision model

`image_part` downloads an image attachment and builds an OpenAI-compatible
content part:

```python
part = task.files.image_part(attachment)
messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"}, part]}]
```

Select a vision model for your agent in the portal. It reaches the container in
`SOFIAS_VISION_MODEL`, with its gateway in `SOFIAS_VISION_BASE_URL` and
`SOFIAS_VISION_API_KEY`:

```python
import os
from sofias_sdk_lite.llm.gateway import GatewayLLM

vision = GatewayLLM(
    base_url=os.environ["SOFIAS_VISION_BASE_URL"],
    api_key=os.environ["SOFIAS_VISION_API_KEY"],
    model=os.environ["SOFIAS_VISION_MODEL"],
)
```

## Tool calls in the sandbox logs

Every tool the model calls, yours or a connector's, leaves one line on standard
output:

```text
SOFIAS_EVENT {"type":"tool_call","tool":"search","server":"notion","ok":true,"duration_ms":412,...}
```

The portal's sandbox shows them as a timeline of the tools your agent called.
They carry names, timings and outcome only, never arguments or results. Set
`SOFIAS_EVENTS=off` to silence them.
