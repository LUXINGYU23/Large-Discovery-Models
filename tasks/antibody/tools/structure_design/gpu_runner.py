"""Apply an explicit PyTorch allocator cap before executing an upstream CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-gib", type=float, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module")
    target.add_argument("--script")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = args.limit_gib * 1024**3 / total
    if not 0 < fraction < 1:
        parser.error("Allocator limit must be positive and below physical GPU memory")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [args.module or args.script, *arguments]
    if args.module:
        runpy.run_module(args.module, run_name="__main__")
    else:
        # run_path does not add the script's directory as `python script.py`
        # does; original ProteinMPNN imports its sibling utility module.
        sys.path.insert(0, str(Path(args.script).resolve().parent))
        runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
