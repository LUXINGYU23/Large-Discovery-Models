"""Shared-runner entry point."""

from tasks.nanogpt.core.workflow import (
    describe_ldm_task,
    main,
    parse_args,
    validate_feature_payload,
    validate_generator_action,
)


if __name__ == "__main__":
    raise SystemExit(main())
