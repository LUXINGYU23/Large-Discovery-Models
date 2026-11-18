"""Run the pinned H20 asynchronous runtime in its own Python process."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from ldm_tts.engine.run_store import atomic_json_write


REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_PIN = REPOSITORY / "config" / "async_ldm" / "runtime.json"


def read_runtime_pin():
    return json.loads(RUNTIME_PIN.read_text(encoding="utf-8"))


def run_fixture(program: str, directory: Path, arguments: list[str]) -> int:
    pin = read_runtime_pin()
    archive = Path(pin["archive"]).read_bytes()
    if hashlib.sha256(archive).hexdigest() != pin["sha256"]:
        raise ValueError(f"H20 async runtime archive does not match pin {pin['commit']}")

    atomic_json_write(directory / "async-runtime.json", pin)
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=directory) as source:
        with ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(source)
        env = {
            **os.environ,
            "PYTHONPATH": source,
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        return subprocess.run(
            [sys.executable, str(REPOSITORY / "scripts" / program), *arguments],
            cwd=source,
            env=env,
            check=False,
        ).returncode
