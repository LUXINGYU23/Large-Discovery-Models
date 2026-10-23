import importlib.util
import os
import shutil
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
    return checks
