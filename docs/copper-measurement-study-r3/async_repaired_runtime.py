import hashlib,importlib.util,runpy,sys
from pathlib import Path
p=Path('/Aidata/shared/ldm/copper-permanent-repair-20261025/async_transport_qualified.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='cc295e18f1723bb314226925d4b6cbe2534a6c5bda2ad9e3a43cbf5fde0c70f5'
name='tasks.copper_inversion.v2.async_transport'
spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);m.install()
runpy.run_path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3/controller-epochs/admission-deadline-r1/repaired_runtime.py',run_name='__main__')
