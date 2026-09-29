"""Stable shared-runner adapter; the implementation lives in core."""

from tasks.researchgym.core import workflow


def parse_args(argv=None):
    return workflow.parse_args(argv)


def describe_ldm_task(*args, **kwargs):
    return workflow.describe_ldm_task(*args, **kwargs)


def main(argv=None):
    return workflow.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
