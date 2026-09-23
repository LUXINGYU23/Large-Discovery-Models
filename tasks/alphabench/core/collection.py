"""Publish complete paired IR/SFT generations from immutable accepted actions."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ldm_tts.data import DataCollectionSink
from ldm_tts.data.collection import read_jsonl
from ldm_tts.data.ir import validate_ir_record
from ldm_tts.engine.run_store import atomic_json_write
from .protocol import digest
from .receipts import Receipts


class AcceptedActions:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.journal = Receipts(self.run_dir / "accepted_actions")
        self.options = DataCollectionSink.from_env(default_root=self.run_dir / "ldm_data")

    def accept(self, action_id, ir, provenance):
        validate_ir_record(ir)
        self.journal.accept(action_id, {"action_id": action_id, "ir": ir, "provenance": provenance})
        self.export()

    def export(self):
        options = self.options
        if not options.enabled:
            return None
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.journal.root.glob("*.json"))]
        generation = digest(records)
        root = options.root_dir / "alphabench" / digest(str(self.run_dir.resolve()))
        root.mkdir(parents=True, exist_ok=True)
        destination = root / generation
        expected_ids = [record["action_id"] for record in records]
        expected_files = {name for name in (options.ir_filename, options.sft_filename,
            options.dataset_info_filename if options.sft_filename else None) if name}
        if not destination.exists():
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
            try:
                sink = DataCollectionSink(root_dir=staging, enabled=True,
                    ir_filename=options.ir_filename, sft_filename=options.sft_filename,
                    dataset_info_filename=options.dataset_info_filename,
                    render_mode=options.render_mode, include_parent_artifact=options.include_parent_artifact)
                for record in records:
                    sink.append(record["ir"], provenance=record["provenance"])
                paths = sink.paths
                paths.ir_path.touch(exist_ok=True)
                if paths.sft_path:
                    paths.sft_path.touch(exist_ok=True)
                files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in staging.iterdir() if path.is_file()}
                atomic_json_write(staging / "manifest.json", {"generation": generation, "count": len(records),
                                  "action_ids": expected_ids, "files": files})
                self._verify_generation(staging, generation, expected_ids, expected_files)
                os.replace(staging, destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        self._verify_generation(destination, generation, expected_ids, expected_files)
        atomic_json_write(root / "current.json", {"generation": generation, "count": len(records)})
        return destination

    def _verify_generation(self, directory, generation, action_ids, expected_files):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        files = manifest.get("files")
        if (manifest.get("generation") != generation or manifest.get("count") != len(action_ids)
                or manifest.get("action_ids") != action_ids or not isinstance(files, dict)
                or set(files) != expected_files
                or {path.name for path in directory.iterdir() if path.is_file()} != expected_files | {"manifest.json"}):
            raise ValueError("accepted-action export manifest differs from the journal")
        for name, expected_hash in files.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"accepted-action export hash mismatch: {name}")
        if (len(read_jsonl(directory / self.options.ir_filename)) != len(action_ids)
                or (self.options.sft_filename is not None
                    and len(read_jsonl(directory / self.options.sft_filename)) != len(action_ids))):
            raise ValueError("accepted-action export IR/SFT row count mismatch")
