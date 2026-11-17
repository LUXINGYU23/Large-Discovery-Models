"""Standard LDM task entry point."""
from tasks.antibody_rfd3.core.workflow import describe_ldm_task, parse_args, run


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
