"""Solve one task with Claude Code and return trajectory events + submission.

This is the whole code-agent backend: copy the workspace, run Claude Code
headless in it through the capture proxy, read off the resulting diff or script,
and build the trajectory events from the genuine wire bodies the proxy recorded.
It replaces the OpenRouter solving loop; the acceptance-suite author, adjudicator
and auditor layers are not used, because Claude Code does its own iterating.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .events import build_trajectory_events
from .proxy import CaptureProxy
from .runner import build_argv, child_env, run_claude
from .submission import TERMINAL_SCRIPT_NAME, changed_files, git_diff, read_terminal_script

HARNESS_NAME = "claude-code"
HARNESS_VERSION = "1"
MODEL_PROVIDER = "anthropic"

# Kept back from the solve deadline for building the diff and uploading.
_FINISH_RESERVE_S = 25.0


@dataclass
class CodeAgentResult:
    events: list[dict[str, Any]]
    submission: bytes
    cost_usd: float | None
    num_turns: int | None
    session_id: str
    exit_code: int | None


def _repository_prompt(instruction: str) -> str:
    return (
        f"{instruction}\n\n"
        "You are working inside a code repository. Investigate the reported problem, "
        "then make the smallest change that makes the code behave as documented. "
        "Run the project's own build and tests to confirm your fix, and edit only the "
        "source needed. Do not add dependencies or unrelated changes. When the fix is "
        "verified, stop."
    )


def _terminal_prompt(instruction: str, script_name: str) -> str:
    return (
        f"{instruction}\n\n"
        f"Write a single self-contained bash script to ./{script_name} at the root of this "
        "workspace. When that script is run from the result directory in a clean copy of this "
        "workspace, it must produce exactly the required output files. You may run the script "
        "yourself to test it, but your final answer is the script file itself: leave it at "
        f"./{script_name} and stop. Do not rely on anything you created outside the script."
    )


async def solve_code_agent(
    *,
    settings: Any,
    request: Any,
    workspace_root: Path,
    deadline: float,
    model_name: str,
    log: Callable[[str], Awaitable[None]] | None = None,
) -> CodeAgentResult:
    """Run one headless solve. ``workspace_root`` is the pristine extracted workspace."""
    identity = request.identity
    terminal = identity.task_type != "repository_patch_v1"
    working_directory = identity.result_tree_path if terminal else identity.working_directory

    async def say(message: str) -> None:
        if log is not None:
            await log(message)

    edited = Path(tempfile.mkdtemp(prefix="hone-codeagent-", dir=str(workspace_root.parent)))
    edited_root = edited / "work"
    try:
        await asyncio.to_thread(shutil.copytree, workspace_root, edited_root, symlinks=True)
        wd = working_directory.strip().strip("/")
        cwd = edited_root if wd in ("", ".") else edited_root / wd
        cwd.mkdir(parents=True, exist_ok=True)

        prompt = (
            _terminal_prompt(identity.instruction, TERMINAL_SCRIPT_NAME)
            if terminal
            else _repository_prompt(identity.instruction)
        )
        budget = settings.miner_code_agent_budget_usd
        timeout_s = max(30.0, deadline - time.monotonic() - _FINISH_RESERVE_S)

        with CaptureProxy(upstream=settings.miner_code_agent_upstream) as proxy:
            argv = build_argv(
                prompt=prompt,
                model=model_name,
                workdir=str(cwd),
                max_turns=settings.miner_code_agent_max_turns,
                max_budget_usd=budget if budget > 0 else None,
                extra_dirs=(str(edited_root),) if cwd != edited_root else (),
                skip_permissions=settings.miner_code_agent_skip_permissions,
                timeout_s=timeout_s,
                claude_binary=settings.miner_code_agent_binary,
            )
            env = child_env(
                proxy.base_url,
                home=settings.miner_code_agent_home or None,
            )
            await say(f"[code-agent] running Claude Code on {request.task_id[:16]} (model {model_name})")
            stream = await run_claude(
                argv, cwd=str(cwd), env=env, hard_deadline_s=timeout_s + 30.0
            )
            calls = proxy.turns

        await say(
            f"[code-agent] turns={stream.num_turns} calls={len(calls)} cost=${stream.cost_usd} "
            f"exit={stream.exit_code} error={stream.is_error}"
        )

        if terminal:
            submission = read_terminal_script(edited_root)
        else:
            touched = await asyncio.to_thread(changed_files, workspace_root, edited_root, working_directory)
            submission = await asyncio.to_thread(
                git_diff, workspace_root, edited_root, touched, working_directory, scratch_dir=edited
            )

        events, _output = build_trajectory_events(stream, calls)
        return CodeAgentResult(
            events=events,
            submission=submission,
            cost_usd=stream.cost_usd,
            num_turns=stream.num_turns,
            session_id=stream.session_id,
            exit_code=stream.exit_code,
        )
    finally:
        await asyncio.to_thread(shutil.rmtree, edited, True)
