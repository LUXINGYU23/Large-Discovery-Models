"""Shared-runner entry point."""

from tasks.antibody.core.workflow import (
    configure_llm_environment,
    describe_ldm_task,
    main,
    parse_args,
)


if __name__ == "__main__":
    raise SystemExit(main())
