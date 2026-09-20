"""Stable manifest-driven shared-runner entrypoint."""

from tasks.atomworld.core import workflow


def parse_args(argv=None):
    return workflow.parse_args(argv)


def describe_ldm_task(args=None):
    return workflow.describe_ldm_task(args)


def main(argv=None):
    return workflow.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
