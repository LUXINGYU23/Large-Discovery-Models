"""Stable shared-runner adapter; scientific implementation lives in core."""

from tasks.reasyn.core import workflow


def parse_args(argv=None):
    return workflow.parse_args(argv)


def describe_ldm_task(*args, **kwargs):
    return workflow.describe_ldm_task(*args, **kwargs)


def main(argv=None):
    return workflow.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
