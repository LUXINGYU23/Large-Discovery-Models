"""Stage a native executable and explicitly selected precomputed scoring assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from tasks.antibody_rfd3.core.experiments import _assets
from tasks.antibody_rfd3.repro.h20.launch_utils import ACTIVE_ROOT, save_summary
from tasks.antibody_rfd3.repro.h20.run_full_ldm import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antigen", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--asset", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ACTIVE_ROOT not in output.parents or output.exists():
        parser.error("Use a new directory below /Aidata/shared/ldm")
    sources = {f.name: f for f in args.asset}
    if len(sources) != len(args.asset) or "assets.json" in sources:
        parser.error("Asset basenames must be unique and cannot include assets.json")
    sources["src/bin/Absolut"] = args.binary
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "inputs"
        records = []
        for name, source in sources.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "--reflink=auto", "--dereference", str(source.resolve()), str(destination)], check=True)
            destination.chmod(0o555 if name == "src/bin/Absolut" else 0o444)
            records.append({"name": name, "bytes": destination.stat().st_size, "sha256": digest(destination)})
        save_summary(stage / "assets.json", {"schema_version": 1, "antigen": args.antigen, "assets": records})
        (stage / "assets.json").chmod(0o444)
        _assets(stage, args.antigen)
        stage.rename(output)
    print(output)


if __name__ == "__main__":
    main()
