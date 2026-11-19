"""Merge the capture proxy's wire bodies with Claude Code's stream into events.

The trajectory's model turns carry the genuine request/response bytes recorded by
the proxy; the tool calls and their results come from the parsed stream. Turns
and calls are paired in order. Nothing here invents a model body: if the proxy
recorded no successful model call, :func:`build_trajectory_events` raises, so a
run whose real turns were not captured fails the solve instead of uploading a
reconstructed trajectory.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .proxy import ModelCall
from .runner import StreamResult

# A single tool result is capped so one huge command output cannot blow the
# trajectory slot. The proxy's model-turn request bodies, which repeat the whole
# history, are trimmed separately by the caller to fit the slot.
_MAX_TOOL_OUTPUT_BYTES = 256 * 1024


class NoModelTurns(RuntimeError):
    """The proxy recorded no usable model call, so there is nothing genuine to upload."""


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _turn_text(blocks: list) -> str:
    return "".join(b.data.get("text", "") for b in blocks if b.type == "text")


def build_trajectory_events(
    stream: StreamResult, calls: list[ModelCall]
) -> tuple[list[dict[str, Any]], str]:
    """Return ``(events, output_text)`` ready for the trajectory.

    ``events`` is model turns interleaved with matched tool call/result pairs, in
    run order, without sequence numbers (the caller assigns those and appends the
    final submission). ``output_text`` is the model's last text, for the solve's
    own bookkeeping.
    """
    successful = [c for c in calls if 200 <= c.status_code < 300 and c.response_body]
    if not successful:
        raise NoModelTurns("the proxy recorded no successful model call")

    events: list[dict[str, Any]] = []
    last_text = ""
    call_index = 0
    for blocks in stream.assistant_turns:
        text = _turn_text(blocks)
        if text:
            last_text = text
        if call_index < len(successful):
            call = successful[call_index]
            call_index += 1
            events.append(
                {
                    "event_type": "model_turn",
                    "request_body_b64": _b64(call.request_body),
                    "response_body_b64": _b64(call.response_body),
                    # Thinking is redacted in the captured body, so we do not claim
                    # per-token evidence; tokens stay empty and generated_bytes is the text.
                    "generated_bytes_b64": _b64(text.encode("utf-8")),
                    "reasoning": "",
                    "output": text,
                    "tokens": [],
                }
            )
        for b in blocks:
            if b.type != "tool_use" or not b.data.get("id"):
                continue
            call_id = b.data["id"]
            if call_id not in stream.tool_results:
                continue  # every tool call must have exactly one result; drop the unmatched
            output, is_error = stream.tool_results[call_id]
            events.append(
                {
                    "event_type": "tool_call",
                    "call_id": call_id,
                    "tool_name": str(b.data.get("name") or "tool")[:256],
                    "input_body_b64": _b64(
                        json.dumps(b.data.get("input", {}), separators=(",", ":")).encode("utf-8")
                    ),
                }
            )
            events.append(
                {
                    "event_type": "tool_result",
                    "call_id": call_id,
                    "output_body_b64": _b64(output[:_MAX_TOOL_OUTPUT_BYTES]),
                    "is_error": is_error,
                }
            )

    if not any(e["event_type"] == "model_turn" for e in events):
        # All assistant turns were empty of text but calls exist: emit the genuine
        # first call as a model turn so the trajectory has one.
        call = successful[0]
        events.insert(
            0,
            {
                "event_type": "model_turn",
                "request_body_b64": _b64(call.request_body),
                "response_body_b64": _b64(call.response_body),
                "generated_bytes_b64": _b64(b""),
                "reasoning": "",
                "output": "",
                "tokens": [],
            },
        )
    return events, last_text
