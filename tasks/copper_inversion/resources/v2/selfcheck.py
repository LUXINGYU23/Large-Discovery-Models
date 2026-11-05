"""Public convenience check; the controller independently checks sealed code."""
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.geometry import canonicalize
spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for problem in fixtures():
    for array in [problem["domain_mask"], problem["forbidden_mask"],
                  *[o[k] for o in problem["observations"] for k in ("points", "labels", "groups")]]:
        array.flags.writeable = False
    field = module.solve(problem, dict(wall_time_seconds=5, cpu_count=1, memory_mb=2048), 47)
    canonicalize(problem, field)
print("Six public v2 interface checks passed; not a formal geometry score.")
