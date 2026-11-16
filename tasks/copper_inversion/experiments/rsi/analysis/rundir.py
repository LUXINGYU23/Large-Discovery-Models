"""Read-only view of one archived run directory (public side).

Both the prototype layout (proposals/round_tt/branch_b.json, config inside the
manifest) and the contracts layout (proposals/round_tt.jsonl, contracts §2.3
manifest) are accepted. All format knowledge lives here.
"""

import hashlib
import json
import re
from functools import cached_property
from pathlib import Path

from ....rsi.ledger import read as read_ledger
from ....rsi.render import stripped_source

LOOP_DEFAULTS = {"rounds": 20, "reset_period": 5, "proposals_per_branch": 8, "batch": 16, "branches": 4}
CONTINUATION = re.compile(r"ck(?P<seed>\d+)-(?P<mode>conversation|history|best_program)"
                          r"(?P<planted>-planted)?-r(?P<replicate>\d+)$")
ROUND_FILE = re.compile(r"round_(\d+)")


class Run:
    def __init__(self, path, archive_root=None, loop_defaults=None):
        self.path = Path(path)
        self.archive_root = Path(archive_root) if archive_root else self.path.parent.parent
        self.digests = {}
        self._sources, self._stripped = {}, {}
        self._loop_defaults = {**LOOP_DEFAULTS, **(loop_defaults or {})}

    # -- file access; every read is hashed for provenance
    def _bytes(self, relative):
        body = (self.path / relative).read_bytes()
        self.digests[str(relative)] = hashlib.sha256(body).hexdigest()
        return body

    def _json(self, relative):
        return json.loads(self._bytes(relative))

    def exists(self, relative):
        return (self.path / relative).is_file()

    # -- identity
    @cached_property
    def manifest(self):
        return self._json("manifest.json")

    @property
    def run_id(self):
        return self.manifest.get("run_id") or self.path.name

    @property
    def experiment(self):
        return self.manifest.get("experiment") or self.path.parent.name

    def _field(self, name, default=None):
        value = self.manifest.get(name)
        if value is None:
            value = self.manifest.get("config", {}).get(name, default)
        return value

    @property
    def arm(self):
        return self._field("arm")

    @property
    def seed(self):
        return self._field("seed")

    @property
    def tier(self):
        return self.manifest.get("tier", "S")

    @property
    def judge(self):
        return self.manifest.get("judge", "standard")

    @cached_property
    def loop(self):
        config = self.manifest.get("config", {})
        loop = {k: config.get(k, v) for k, v in self._loop_defaults.items()}
        if "branches" not in config:
            loop["branches"] = self._loop_defaults["branches"]
        return loop

    @property
    def rounds(self):
        return self.loop["rounds"]

    @property
    def reset_period(self):
        return self.loop["reset_period"]

    @cached_property
    def continuation(self):
        value = self.manifest.get("continuation")
        match = CONTINUATION.fullmatch(self.run_id)
        if value is None and match:
            value = {"source_run": f"accumulated-s{match['seed']}", "round": 10,
                     "mode": match["mode"], "replicate": int(match["replicate"])}
        if value is not None and "checkpoint" not in value:
            seed = re.search(r"s(\d+)$", value["source_run"])
            value = {**value, "checkpoint": f"ck{seed.group(1)}" if seed else value["source_run"]}
        return value

    @property
    def intervention(self):
        return self.manifest.get("intervention")

    # -- records and receipts
    @cached_property
    def ledger(self):
        self._bytes("ledger.jsonl")
        return read_ledger(self.path / "ledger.jsonl")

    @cached_property
    def result(self):
        return self._json("result.json") if self.exists("result.json") else {}

    @property
    def stop_reason(self):
        return self.result.get("stop_reason")

    @property
    def status(self):
        return self.result.get("status", "incomplete")

    def _round_files(self, folder, suffix):
        root = self.path / folder
        if not root.is_dir():
            return {}
        found = {}
        for path in sorted(root.iterdir()):
            match = ROUND_FILE.fullmatch(path.name.removesuffix(suffix))
            if match:
                found[int(match.group(1))] = path
        return found

    @cached_property
    def proposals(self):
        """round -> proposals (program_id, branch, parent and stated interval), branch order."""
        out = {}
        for round_index, path in self._round_files("proposals", ".jsonl").items():
            rel = path.relative_to(self.path)
            if path.is_dir():
                rows = []
                for receipt in sorted(path.glob("branch_*.json"), key=lambda p: int(p.stem[7:])):
                    rows.extend(self._json(receipt.relative_to(self.path))["proposals"])
            else:
                rows = [json.loads(line) for line in self._bytes(rel).decode().splitlines() if line.strip()]
            out[round_index] = sorted(rows, key=lambda r: r["branch"])
        return out

    @cached_property
    def selection(self):
        """round -> {program_id: prediction with mean, std, acquisition}."""
        out = {}
        for round_index, path in self._round_files("selection", ".json").items():
            data = self._json(path.relative_to(self.path))
            out[round_index] = {p["program_id"]: p for p in data.get("predictions", [])}
        return out

    @cached_property
    def usage(self):
        """round -> {branch: usage}; profile keys are mapped to branch numbers."""
        out = {}
        for round_index, path in self._round_files("usage", ".json").items():
            data = self._json(path.relative_to(self.path))
            rows = {}
            for index, (key, value) in enumerate(sorted(data.get("by_profile", {}).items())):
                digits = re.search(r"(\d+)$", key)
                rows[int(digits.group(1)) if digits else index] = value
            out[round_index] = rows
        return out

    # -- sources and features
    def source(self, program_id):
        if program_id not in self._sources:
            self._sources[program_id] = self._bytes(f"candidates/{program_id}/solver.py")
        return self._sources[program_id]

    def stripped(self, program_id):
        if program_id not in self._stripped:
            self._stripped[program_id] = stripped_source(self.source(program_id))
        return self._stripped[program_id]

    def encoding(self, program_id):
        rel = f"encodings/{program_id}.json"
        return self._json(rel) if self.exists(rel) else None

    def provenance(self):
        """Digest over every file this run contributed to the analysis."""
        listing = "".join(f"{k}\t{v}\n" for k, v in sorted(self.digests.items()))
        return {"files": len(self.digests), "sha256": hashlib.sha256(listing.encode()).hexdigest(),
                "ledger_sha256": self.digests.get("ledger.jsonl"),
                "manifest_sha256": self.digests.get("manifest.json")}


def discover(archive):
    """Every run directory under <archive>/<exp>/<run_id>/ (a manifest and a ledger)."""
    archive = Path(archive)
    runs = []
    for manifest in sorted(archive.glob("*/*/manifest.json")):
        if (manifest.parent / "ledger.jsonl").is_file():
            runs.append(Run(manifest.parent, archive))
    return runs
