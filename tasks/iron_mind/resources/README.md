# Iron Mind Resources

This directory contains versioned source contracts and reaction schemas required
by the task. The pinned Olympus data stays under `/mnt/data1/ldm-for-sci/data/`
and is verified against `upstream_contract.json` before loading.

`mock_oracle.csv` is a deterministic, task-owned mock fixture. It is not a
scientific benchmark dataset and must not be substituted for the source-pinned
Olympus reaction tables in a real campaign.

Do not commit runtime outputs, downloaded datasets, model checkpoints, or
credentials to this directory.
