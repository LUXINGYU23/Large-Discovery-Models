"""Shared-runner entry point."""

from tasks.reasyn.core.task_spec import describe_ldm_task
from tasks.reasyn.core.workflow import main
from tasks.reasyn.core.workflow_args import parse_args


if __name__ == "__main__":
    raise SystemExit(main())
