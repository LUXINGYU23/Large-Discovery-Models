"""Shared runner entry point; all domain implementation lives in core."""

from tasks.copper_inversion.core.workflow import main, parse_args, describe_ldm_task

if __name__ == "__main__":
    raise SystemExit(main())
