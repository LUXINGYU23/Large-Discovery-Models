"""Route LDM to the streaming campaign and baselines to synchronous evaluation."""
import argparse

from tasks.antibody_rfd3.core import workflow as streaming
from tasks.antibody_rfd3.core.synchronous import workflow as synchronous


def backend(method):
    return streaming if method == "ldm_harness" else synchronous


def parse_args(argv=None):
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--method", choices=streaming.METHODS)
    probe.add_argument("--mock", action="store_true")
    selection, _ = probe.parse_known_args(argv)
    method = selection.method or ("fixed" if selection.mock else "ldm_harness")
    return backend(method).parse_args(argv)


def describe_ldm_task(args):
    return backend(args.method).describe_ldm_task(args)


def main(argv=None):
    args = parse_args(argv)
    return backend(args.method).run(args)


if __name__ == "__main__":
    raise SystemExit(main())
