"""AT-09 driver (H20 phase): run rsi_guest_probe.mjs in the real sidecar with PiAgent's mounts.

python -m tasks.copper_inversion.tests.rsi_sidecar_probe --image <sidecar ref> --cache-dir <cache> --work <empty dir>
Prints the docker command with --dry-run. Needs Docker and a built guest image in the cache.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import load_pi_guest_runtime
from tasks.copper_inversion.rsi.agent import RESOURCES, container_name, sidecar_mounts


def command(image, cache, work):
    run = Path(work).resolve() / "probe-run"
    own, sibling = (run / "harness/epoch_00" / f"branch_{b}" for b in (0, 1))
    for root, text in ((own, "own-view-canary"), (sibling, "sibling-view-canary")):
        (root / "public/evidence").mkdir(parents=True, exist_ok=True)
        (root / "public/evidence/history.txt").write_text(text)
    probe = Path(work).resolve() / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("rsi_guest_probe.mjs"), probe / "rsi_guest_probe.mjs")
    runtime = load_pi_guest_runtime("copper_inversion", RESOURCES / "harness/image")
    (probe / "runtime.json").write_text(json.dumps(runtime.to_dict()))
    base = pi_sidecar_command(own, image, cache_root=Path(cache),
                              environment={"PROBE_SIBLING": str(sibling), "PROBE_RUN_DIR": str(run)},
                              mounts=(*sidecar_mounts(), (probe, "/probe", True)))
    at = base.index("run") + 1
    base = base[:at] + ["--name", container_name(own, "researcher_0", run.name, 0)] + base[at:]
    return base[:-1] + ["--entrypoint", "node", base[-1], "/probe/rsi_guest_probe.mjs"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    argv = command(args.image, args.cache_dir, args.work)
    if args.dry_run:
        print(json.dumps(argv))
        return 0
    return subprocess.run(argv, stdin=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    sys.exit(main())
