"""Shared-runner entry point."""

from tasks.nucleobench.core.workflow import describe_ldm_task, main
from tasks.nucleobench.core.workflow_args import parse_args


if __name__ == "__main__":
    raise SystemExit(main())
