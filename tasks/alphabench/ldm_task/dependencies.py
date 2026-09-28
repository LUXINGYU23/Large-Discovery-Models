from ldm_tts.registration.dependencies import arg_value, fail, ok, plan_check_context, resolve_task_path


def check_dependencies(plan, *, include_optional=True):
    task, args, env, cwd, mode = plan_check_context(plan)
    checks = [ok(task, "adapter", "AlphaBench T3 adapter imports", str(cwd))]
    required = [] if mode == "mock" else ["data-manifest", "upstream-root", "protocol-file"]
    if (arg_value(args, "method") or "").startswith("alphabench_"):
        required.extend(name for name in ("upstream-root", "protocol-file") if name not in required)
    for name in required:
        path = resolve_task_path(arg_value(args, name), cwd)
        checks.append(ok(task, name, "Available", str(path)) if path and path.exists()
                      else fail(task, name, f"A verified {name} is required"))
    return checks
