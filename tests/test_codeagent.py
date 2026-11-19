"""Offline tests for the Claude Code (code_agent) solver backend.

None of these run Claude Code or touch a model: they exercise the capture proxy
against a fake upstream, the stream parser, the trajectory builder (round-tripped
through the real validator), and the diff builder against the yaml-cpp fixture.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from rlvr.neurons.codeagent.events import NoModelTurns, build_trajectory_events
from rlvr.neurons.codeagent.proxy import CaptureProxy, ModelCall
from rlvr.neurons.codeagent.runner import build_argv, child_env, parse_stream
from rlvr.neurons.codeagent.submission import changed_files, git_diff
from rlvr.v3.trajectory import parse_trajectory, serialize_trajectory, Trajectory

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "v3-yamlcpp-e6225d04"


class _Echo(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # noqa: N802
        return

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        req = self.rfile.read(n)
        assert self.headers.get("Authorization") == "Bearer SECRET"
        body = json.dumps({"path": self.path, "echo": json.loads(req)}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _fake_upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_proxy_forwards_and_records_only_messages():
    server, url = _fake_upstream()
    try:
        with CaptureProxy(upstream=url) as proxy:
            r = httpx.post(
                proxy.base_url + "/v1/messages",
                json={"x": 42},
                headers={"Authorization": "Bearer SECRET"},
                timeout=10,
            )
            assert r.status_code == 200 and r.json()["echo"]["x"] == 42
            # a token-count call is forwarded but NOT recorded as a model turn
            httpx.post(
                proxy.base_url + "/v1/messages/count_tokens",
                json={"x": 1},
                headers={"Authorization": "Bearer SECRET"},
                timeout=10,
            )
            turns = proxy.turns
        assert len(turns) == 1
        assert json.loads(turns[0].request_body)["x"] == 42
        assert b"SECRET" not in turns[0].request_body and b"SECRET" not in turns[0].response_body
    finally:
        server.shutdown()
        server.server_close()


def _stream_lines():
    return [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file_path": "a"}}]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "data", "is_error": False}]}}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Fixed."},
            {"type": "tool_use", "id": "tu2", "name": "Edit", "input": {"p": 1}}]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu2", "content": "ok", "is_error": False}]}}),
        json.dumps({"type": "result", "subtype": "success", "is_error": False,
                    "total_cost_usd": 0.12, "num_turns": 2, "session_id": "s1"}),
    ]


def test_parse_stream():
    st = parse_stream(_stream_lines())
    assert len(st.assistant_turns) == 2
    assert st.cost_usd == 0.12 and st.num_turns == 2 and not st.is_error
    assert set(st.tool_results) == {"tu1", "tu2"}


def test_build_events_make_a_valid_trajectory():
    st = parse_stream(_stream_lines())
    calls = [
        ModelCall(json.dumps({"model": "m", "messages": []}).encode(),
                  json.dumps({"id": "msg_1", "content": [{"type": "text", "text": "Looking."}]}).encode(), 200, "application/json"),
        ModelCall(json.dumps({"model": "m", "messages": []}).encode(),
                  json.dumps({"id": "msg_2", "content": [{"type": "text", "text": "Fixed."}]}).encode(), 200, "application/json"),
    ]
    events, output = build_trajectory_events(st, calls)
    assert output == "Fixed."
    assert [e["event_type"] for e in events][:3] == ["model_turn", "tool_call", "tool_result"]

    submission = b"diff --git a/x b/x\n"
    sha = hashlib.sha256(submission).hexdigest()
    b64 = lambda x: base64.b64encode(x).decode()  # noqa: E731
    events += [
        {"event_type": "tool_call", "call_id": "submission-validation-0",
         "tool_name": "validate_submission", "input_body_b64": b64(submission)},
        {"event_type": "tool_result", "call_id": "submission-validation-0",
         "output_body_b64": b64(b"{}"), "is_error": False},
        {"event_type": "final_submission", "submission_sha256": sha},
    ]
    for i, e in enumerate(events):
        e["sequence"] = i
    traj = Trajectory(schema_version=1, task_id="a" * 64, challenge_id="c", miner_hotkey="5F",
                      submission_sha256=sha, harness_name="claude-code", harness_version="1",
                      model_provider="anthropic", model_name="opus", events=events)
    # the strict server-side validator must accept it
    assert parse_trajectory(serialize_trajectory(traj)).submission_sha256 == sha


def test_build_events_rejects_no_captured_calls():
    st = parse_stream(_stream_lines())
    with pytest.raises(NoModelTurns):
        build_trajectory_events(st, [ModelCall(b"{}", b"", 500, "")])


def test_build_argv_and_env():
    argv = build_argv(prompt="go", model="opus", workdir="/w", max_turns=40,
                      max_budget_usd=2.0, timeout_s=600)
    assert argv[:5] == ["timeout", "-s", "INT", "-k", "15"]
    assert "--dangerously-skip-permissions" in argv and "stream-json" in argv
    env = child_env("http://127.0.0.1:9", home="/home/hone")
    assert env["ANTHROPIC_BASE_URL"].endswith(":9") and env["HOME"] == "/home/hone"
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env


@pytest.mark.skipif(not (FIXTURE / "workspace.tar.zst").exists() or shutil.which("git") is None,
                    reason="needs the yaml-cpp fixture and git")
def test_diff_builder_matches_git_apply(tmp_path):
    import zstandard

    pristine = tmp_path / "pristine"
    pristine.mkdir()
    with open(FIXTURE / "workspace.tar.zst", "rb") as fh:
        data = zstandard.ZstdDecompressor().stream_reader(fh).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(pristine, filter="data")
    edited = tmp_path / "edited"
    shutil.copytree(pristine, edited, symlinks=True)
    ref = (FIXTURE / "reference.diff").read_bytes()
    assert subprocess.run(["git", "-C", str(edited), "apply", "-p1", "--whitespace=nowarn"],
                          input=ref, capture_output=True).returncode == 0

    touched = changed_files(pristine, edited, ".")
    assert touched  # the reference fix changed at least one file
    diff = git_diff(pristine, edited, touched, ".", scratch_dir=tmp_path)
    assert diff.startswith(b"diff --git")

    check = tmp_path / "check"
    shutil.copytree(pristine, check, symlinks=True)
    assert subprocess.run(["git", "-C", str(check), "apply", "--check", "-p1"],
                          input=diff, capture_output=True).returncode == 0
