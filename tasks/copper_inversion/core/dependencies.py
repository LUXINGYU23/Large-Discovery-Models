import importlib.util
import os
import shutil
from pathlib import Path
from ldm_tts.registration.dependencies import plan_check_context, ok, fail, skip


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    checks = []
    for name in ("numpy", "scipy", "skfmm", "discretize"):
        checks.append(
            (ok if importlib.util.find_spec(name) else fail)(
                task, name, "Python dependency availability"
            )
        )
    if args.get("encoding", "ast") != "ast":
        for name in ("sentence_transformers", "torch", "transformers"):
            checks.append((ok if importlib.util.find_spec(name) else fail)(task, name, "Encoding dependency"))
        model = Path(args.get("embedding-model-path", ""))
        checks.append((ok if (model / "provenance.json").is_file() and (model / "model.safetensors").is_file() else fail)(
            task, "embedding model", "Frozen local Qwen model and provenance required"))
    if mode == "mock" or args.get("mock"):
        checks.append(skip(task, "docker/provider", "Mock is service-free"))
    else:
        checks.append(
            (ok if shutil.which("docker") else fail)(
                task, "docker", "Docker CLI required"
            )
        )
        checks.append(
            (ok if os.path.exists("/dev/kvm") else fail)(task, "kvm", "Pi requires KVM")
        )
        try:
            membership = Path("/proc/self/cgroup").read_text()
            relative = next(line[3:] for line in membership.splitlines() if line.startswith("0::"))
            peak = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.peak"
            available = peak.is_file()
        except (OSError, StopIteration):
            available = False
        checks.append((ok if available else fail)(
            task, "memory.peak", "Lifetime memory diagnostics require a non-root cgroup v2 memory.peak counter"))
    return checks
