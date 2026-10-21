"""Shared-runner entry point."""

from tasks.atomworld.core.workflow import describe_ldm_task, main, parse_args


if __name__ == "__main__":
    raise SystemExit(main())
