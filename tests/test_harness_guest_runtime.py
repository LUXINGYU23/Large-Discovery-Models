from __future__ import annotations

import json
from pathlib import Path

import pytest

from ldm_tts.harness import load_harness_guest_runtime


def _write_recipe(root: Path, *, smoke_script: str = "smoke.sh") -> None:
    (root / "lock").mkdir(parents=True)
    (root / "Dockerfile").write_text("FROM example@sha256:" + "a" * 64 + "\n")
    (root / "smoke.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "lock" / "requirements.lock").write_text("package==1\n")
    (root / "guest-image.json").write_text(json.dumps({
        "schemaVersion": 1,
        "imageId": "fixture-research",
        "rootfsSize": "4G",
        "installPolicy": "session_overlay",
        "recipeFiles": ["Dockerfile", "smoke.sh", "lock/requirements.lock"],
        "smokeScript": smoke_script,
    }))


def test_recipe_digest_changes_with_a_listed_file(tmp_path: Path) -> None:
    _write_recipe(tmp_path)
    first = load_harness_guest_runtime("fixture", tmp_path)
    (tmp_path / "lock" / "requirements.lock").write_text("package==2\n")
    second = load_harness_guest_runtime("fixture", tmp_path)

    assert first.recipe_sha256 != second.recipe_sha256
    assert first.image_ref != second.image_ref


def test_recipe_rejects_a_smoke_path_outside_the_recipe(tmp_path: Path) -> None:
    _write_recipe(tmp_path, smoke_script="../smoke.sh")

    with pytest.raises(ValueError, match="image directory"):
        load_harness_guest_runtime("fixture", tmp_path)
