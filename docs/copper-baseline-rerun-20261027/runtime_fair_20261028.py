"""Frozen baseline-only runtime: qualified async transport and interface gate."""
import hashlib,importlib.util,json,runpy,sys
from pathlib import Path
O=Path(__file__).resolve().parent
for path,sha in json.loads((O/'ack-frozen-20261028.json').read_text()).items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,'ack component changed'
for path,sha in json.loads((O/'fair-frozen-20261028.json').read_text()).items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,'fair component changed'
manifest=json.loads((O/'frozen-files.json').read_text())
for name,sha in manifest.items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,'frozen component changed: '+name
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
transport=load('tasks.copper_inversion.v2.async_transport',O/'async_transport.py')
gate=load('tasks.copper_inversion.v2.concurrent_interfaces',O/'interface_checks.py')
from tasks.copper_inversion.v2 import interface_checks,service
interface_checks.InterfaceChecks=gate.InterfaceChecks;service.InterfaceChecks=gate.InterfaceChecks
from tasks.copper_inversion.experiments.v2 import parallel_runtime
fair=load('fair_queue_20261028',O/'fair_queue_20261028.py')
parallel_runtime.JobQueue=fair.JobQueue
if sys.argv[1]=='load':
 sys.argv=[str(O/'qualify_fair_20261028.py'),*sys.argv[2:]]
 runpy.run_path(str(O/'qualify_fair_20261028.py'),run_name='__main__')
elif sys.argv[1]=='qualify':
 policy=json.loads(Path(sys.argv[2]).read_text());parallel_runtime.install(policy)
 sys.argv=[sys.argv[0],*sys.argv[3:]]
 runpy.run_module('tasks.copper_inversion.experiments.v2.qualify_async_protocol',run_name='__main__')
else:
 ack=load('acknowledged_failure_20261028',O/'acknowledged_failure_20261028.py')
 ack.install(json.loads((O/'acknowledged-failure-receipts-20261028.json').read_text()))
 transport.install();parallel_runtime.main()
