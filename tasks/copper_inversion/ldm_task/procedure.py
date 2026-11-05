"""Registered copper v2 entry point: cross-case, fusion-only discovery."""

from tasks.copper_inversion.v2.workflow import main, parse_args
from tasks.copper_inversion.v2.task_spec import describe_task
from tasks.copper_inversion.v2.public_catalog import catalog_digest


def describe_ldm_task(args):
    return describe_task(catalog_digest(args.public_catalog))

if __name__ == "__main__":
    raise SystemExit(main())
