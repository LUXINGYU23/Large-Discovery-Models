"""Shared-runner entry point."""

from tasks.atomworld.core.workflow import main
from tasks.atomworld.core.task_spec import describe_ldm_task
from tasks.atomworld.core.workflow_args import parse_args


if __name__ == "__main__":
    raise SystemExit(main())
