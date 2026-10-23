# Runtime readiness probe

Only verify the environment; do not design an inversion algorithm.
Use bash to import numpy, scipy, skfmm and discretize. Confirm that writing to
/workspace/.ldm-resources/AGENTS.md fails with OSError. Then write exactly ready
to /workspace/probe.txt and call submit_readiness with ready=true.
