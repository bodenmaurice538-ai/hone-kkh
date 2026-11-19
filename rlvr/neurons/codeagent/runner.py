"""Run Claude Code headless over one task and parse its stream.

This drives ``claude -p --output-format stream-json`` in the task workspace,
pointed at the capture proxy, and parses the stream into the turns and tool
calls the trajectory needs. It never reads or sets a model credential: the
subscription login lives in the Claude Code config on the host, and the only
auth-related thing done here is to *remove* ``ANTHROPIC_API_KEY`` and
``ANTHROPIC_AUTH_TOKEN`` from the child environment, since either would outrank
the subscription (authentication precedence, verified against the Claude Code
docs). ``--bare`` is never used, as it skips the OAuth credential.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

# How much the subprocess stream reader may buffer; stream-json lines can be large.
_STDOUT_LINE_LIMIT = 64 * 1024 * 1024


@dataclass
class ContentBlock:
    type: str
    # text for a text block; for tool_use: name/id/input; for tool_result: content/is_error
    data: dict[str, Any]


@dataclass
class StreamResult:
    """What one headless run produced, parsed from its stream-json output."""

    # Each assistant message's content blocks, in order (one entry per assistant turn).
    assistant_turns: list[list[ContentBlock]] = field(default_factory=list)
    # call_id -> (output_text bytes, is_error), from user tool_result messages.
    tool_results: dict[str, tuple[bytes, bool]] = field(default_factory=dict)
    # The order tool_use ids first appeared, so events can be emitted in run order.
    tool_order: list[str] = field(default_factory=list)
    result_text: str = ""
    is_error: bool = False
    cost_usd: float | None = None
    num_turns: int | None = None
    session_id: str = ""
    exit_code: int | None = None
    raw_lines: int = 0


def build_argv(
    *,
    prompt: str,
    model: str,
    workdir: str,
    max_turns: int,
    max_budget_usd: float | None,
    extra_dirs: tuple[str, ...] = (),
    allowed_tools: tuple[str, ...] = (
        "Read",
        "Edit",
        "Write",
        "Bash",
        "Grep",
        "Glob",
        "LS",
    ),
    skip_permissions: bool = True,
    timeout_s: float | None = None,
    claude_binary: str = "claude",
    timeout_binary: str = "timeout",
) -> list[str]:
    """The exact command line for one headless solve.

    When ``timeout_s`` is set the command is wrapped in coreutils ``timeout`` with
    a SIGINT first (which ends the current turn cleanly) and a SIGTERM fallback,
    so an overrunning run stops without leaving a turn half-recorded.
    """
    argv: list[str] = []
    if timeout_s is not None:
        # SIGINT ends the turn cleanly; -k adds a hard SIGTERM 15s later.
        argv += [timeout_binary, "-s", "INT", "-k", "15", f"{max(1, int(timeout_s))}"]
    argv += [
        claude_binary,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",  # required with stream-json
        "--model",
        model,
        "--max-turns",
        str(int(max_turns)),
    ]
    if max_budget_usd is not None:
        argv += ["--max-budget-usd", f"{max_budget_usd:g}"]
    if skip_permissions:
        argv += ["--dangerously-skip-permissions"]
    else:
        # Deny anything that would prompt, rather than hang, on an unattended run.
        argv += ["--permission-prompts", "none", "--allowedTools", ",".join(allowed_tools)]
    for directory in extra_dirs:
        argv += ["--add-dir", directory]
    return argv


def child_env(base_url: str, *, home: str | None = None) -> dict[str, str]:
    """The environment for the child: point it at the proxy, drop API-key auth.

    ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN`` are removed so the
    subscription login is used. ``home`` overrides ``HOME`` when the Claude Code
    config (the saved login) lives somewhere other than the calling user's home.
    """
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = base_url
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    if home is not None:
        env["HOME"] = home
    return env


def _block(raw: dict[str, Any]) -> ContentBlock:
    kind = raw.get("type", "")
    if kind == "text":
        return ContentBlock("text", {"text": raw.get("text", "")})
    if kind == "tool_use":
        return ContentBlock(
            "tool_use",
            {"id": raw.get("id", ""), "name": raw.get("name", ""), "input": raw.get("input", {})},
        )
    if kind == "tool_result":
        return ContentBlock(
            "tool_result",
            {
                "tool_use_id": raw.get("tool_use_id", ""),
                "content": raw.get("content", ""),
                "is_error": bool(raw.get("is_error", False)),
            },
        )
    return ContentBlock(kind or "unknown", dict(raw))


def _tool_result_bytes(content: Any) -> bytes:
    """A tool_result's content is a string or a list of blocks; render it to bytes."""
    if isinstance(content, str):
        return content.encode("utf-8", "replace")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") if item.get("type") == "text" else json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts).encode("utf-8", "replace")
    return json.dumps(content).encode("utf-8", "replace")


def parse_stream(lines: list[str]) -> StreamResult:
    """Parse stream-json output lines into a :class:`StreamResult`.

    Unrecognized line types are ignored, so a newer Claude Code that adds event
    types still parses. A malformed line is skipped.
    """
    result = StreamResult()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        result.raw_lines += 1
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "assistant":
            message = event.get("message") or {}
            blocks = [_block(b) for b in message.get("content", []) if isinstance(b, dict)]
            result.assistant_turns.append(blocks)
            for b in blocks:
                if b.type == "tool_use" and b.data["id"]:
                    if b.data["id"] not in result.tool_order:
                        result.tool_order.append(b.data["id"])
        elif kind == "user":
            message = event.get("message") or {}
            content = message.get("content", [])
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        call_id = b.get("tool_use_id", "")
                        if call_id:
                            result.tool_results[call_id] = (
                                _tool_result_bytes(b.get("content", "")),
                                bool(b.get("is_error", False)),
                            )
        elif kind == "result":
            result.result_text = event.get("result", "") or ""
            result.is_error = bool(event.get("is_error", False)) or event.get("subtype") not in (None, "success")
            cost = event.get("total_cost_usd", event.get("cost_usd"))
            result.cost_usd = float(cost) if isinstance(cost, (int, float)) else None
            turns = event.get("num_turns")
            result.num_turns = int(turns) if isinstance(turns, int) else None
            result.session_id = event.get("session_id", "") or ""
    return result


async def run_claude(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    hard_deadline_s: float,
) -> StreamResult:
    """Run the headless command, collect its stream, and parse it.

    ``hard_deadline_s`` is a wall-clock ceiling enforced here in addition to the
    ``timeout`` wrapper in ``argv``; on overrun the process is interrupted then
    killed, and whatever streamed so far is still parsed.
    """
    if shutil.which(argv[0]) is None:
        raise FileNotFoundError(f"{argv[0]} is not on PATH")
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # stream-json emits one JSON object per line; a line can carry a whole
        # assistant message or a large tool result, far over asyncio's 64 KiB
        # default, so give the reader a generous line limit.
        limit=_STDOUT_LINE_LIMIT,
    )
    lines: list[str] = []

    async def drain() -> None:
        # Read raw chunks and split lines ourselves, so no single oversized
        # stream-json line can raise the reader's line-length limit.
        assert process.stdout is not None
        buffer = bytearray()
        while True:
            chunk = await process.stdout.read(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                lines.append(bytes(buffer[:newline]).decode("utf-8", "replace"))
                del buffer[: newline + 1]
        if buffer:
            lines.append(bytes(buffer).decode("utf-8", "replace"))

    try:
        await asyncio.wait_for(asyncio.gather(drain(), process.wait()), timeout=hard_deadline_s)
    except (asyncio.TimeoutError, TimeoutError):
        process.send_signal(2)  # SIGINT: let the current turn end
        try:
            await asyncio.wait_for(process.wait(), timeout=20.0)
        except (asyncio.TimeoutError, TimeoutError):
            process.kill()
            await process.wait()
    parsed = parse_stream(lines)
    parsed.exit_code = process.returncode
    return parsed
