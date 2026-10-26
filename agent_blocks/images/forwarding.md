{% if agent_cli == "codex" %}
Camera images are tool-response attachments, not MCP resources; direct MCP calls
deliver them automatically. In `functions.exec`, forward selected image blocks with
`image(block)` and text blocks with `text(block.text)`; never stringify a whole reply
or its image bytes, call `resources/list` for cameras, or resize inline images. View
saved frames selectively with native image viewing rather than loading many at once.
{% else %}
Camera images are tool-result attachments, not MCP resources; you see them directly
in the tool result. Never print or stringify image bytes, call `resources/list` for
cameras, or resize inline images. View saved frames selectively with your image-capable file tool
rather than loading many at once.
{% endif %}
