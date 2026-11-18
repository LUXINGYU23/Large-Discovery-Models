"""Linux/H20 integration test. A scripted Responses fixture exercises the real Pi sidecar."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
SKILL_PATH = "/workspace/.ldm-resources/skills/0/cdrh3-design/SKILL.md"


def object_in(value, key):
    if isinstance(value, str):
        try:
            value, _ = json.JSONDecoder().raw_decode(value.lstrip())
        except ValueError:
            return None
    if isinstance(value, dict):
        if key in value:
            return value
        value = list(value.values())
    if isinstance(value, list):
        for item in reversed(value):
            match = object_in(item, key)
            if match is not None:
                return match
    return None


def action(request):
    messages = request["input"]
    last = messages[-1]
    tools = {t.get("name") for t in request["tools"]}
    assert {"design_h3", "sequence_h3", "read", "submit_designs"} <= tools
    if last.get("role") == "user":
        turn = object_in(last, "measured_history")
        if not turn:
            raise ValueError("Missing campaign turn in provider input")
        if not turn["measured_history"]:
            assert SKILL_PATH in json.dumps(request)
            return "read", {"path": SKILL_PATH}
        row = turn["measured_history"][-1]
        assert row["oracle"] == "mock" and isinstance(row["energy"], float)
        return "sequence_h3", {"backbone_id": row["backbone_id"], "count": 16, "temperature": 0.7}
    call = next(m for m in reversed(messages) if m.get("type") == "function_call")
    if call["name"] == "read":
        assert "WGGDGFYAMDY" in json.dumps(last)
        return "design_h3", {"count": 16, "temperature": 0.5}
    if call["name"] == "submit_designs":
        return None, {}
    result = object_in(last, "candidates")
    turn = object_in([m for m in messages if m.get("role") == "user"], "required_candidates")
    if result is None:
        raise ValueError(f"No generation result after {call['name']}: {str(last)[:320]}")
    excluded = {row["sequence"] for row in turn["measured_history"]}
    ids = []
    for row in result["candidates"]:
        if not row["violations"] and row["sequence"] not in excluded:
            ids.append(row["design_id"])
            excluded.add(row["sequence"])
    count = turn["required_candidates"]
    if len(ids) < count:
        return "design_h3", {"count": 16, "temperature": 0.5}
    return "submit_designs", {"design_ids": ids[:count]}


def events(index, name, params):
    response = {"id": f"resp_{index}", "status": "completed"}
    if name:
        item = {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                "name": name, "arguments": json.dumps(params), "status": "completed"}
        initial = dict(item, arguments="")
    else:
        item = {"type": "message", "role": "assistant", "id": f"msg_{index}", "status": "completed",
                "content": [{"type": "output_text", "text": "Submitted.", "annotations": []}]}
        initial = dict(item, content=[])
    yield {"type": "response.created", "response": dict(response, status="in_progress", output=[])}
    yield {"type": "response.output_item.added", "output_index": 0, "item": initial}
    if name:
        yield {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": item["arguments"]}
    yield {"type": "response.output_item.done", "output_index": 0, "item": item}
    yield {"type": "response.completed", "response": dict(response, output=[item],
          usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                 "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--method", choices=["harness", "ldm_harness"], default="ldm_harness")
    parser.add_argument("--initialization-run", type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit("Smoke requires a fresh output directory")
    gateway = subprocess.check_output(["docker", "network", "inspect", "bridge", "--format",
                                       "{{(index .IPAM.Config 0).Gateway}}"], text=True).strip()
    key = secrets.token_hex(24)
    actions = []

    class Fixture(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {key}":
                self.send_error(403)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not request.get("stream"):
                response = {"model": request["model"], "status": "completed",
                            "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
                body, mime = json.dumps(response), "application/json"
            else:
                try:
                    name, params = action(request)
                    actions.append(name)
                    if len(actions) > 16:
                        raise ValueError("Exceeded integration fixture call budget")
                except Exception as exc:
                    print(f"Scripted provider assertion: {exc!r}", file=sys.stderr, flush=True)
                    self.send_error(400)
                    return
                body = "".join("data: " + json.dumps(event) + "\n\n" for event in events(len(actions), name, params))
                body += "data: [DONE]\n\n"
                mime = "text/event-stream"
            body = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            pass

    with HTTPServer((gateway, 0), Fixture) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = dict(os.environ, LLM_BASE_URL=f"http://{gateway}:{server.server_port}/v1",
                       LLM_MODEL_NAME="scripted-pi-fixture", LLM_API_KEY=key,
                       NO_PROXY=f"{gateway},localhost,127.0.0.1", no_proxy=f"{gateway},localhost,127.0.0.1")
            cmd = [sys.executable, "scripts/run_ldm_tts.py", f"config/antibody_rfd3/{args.method}.yaml",
                   "--set", f"args.out-dir={args.out_dir.resolve()}"]
            if args.initialization_run:
                cmd.extend(["--set", f"args.initialization-run={args.initialization_run.resolve()}"])
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        finally:
            server.shutdown()
            thread.join()
    assert "read" in actions and "sequence_h3" in actions and actions.count("submit_designs") == 2
    print(json.dumps({"provider": "scripted-pi-fixture", "actions": actions}))


if __name__ == "__main__":
    main()
