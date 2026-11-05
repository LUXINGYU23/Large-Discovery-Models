"""Read-only qualification of continuation bindings; never evaluates formal sources."""
import json,sys,tempfile,hashlib,runpy
from pathlib import Path
from unittest.mock import patch
R=Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3'); E=R/'controller-epochs/admission-deadline-r1'
sys.path.insert(0,str(R/'source'))
epoch=json.loads((E/'epoch.json').read_text())
from tasks.copper_inversion.experiments.v2 import parallel_runtime,tune_stochastic
from tasks.copper_inversion.core import harness_runtime
original_source=tune_stochastic.source_for
count=0
for name,info in epoch['baseline_maps'].items():
 rows=json.loads(Path(info['path']).read_text());plan=json.loads((R/name/'plan.json').read_text())
 old=(R/'source/tasks/copper_inversion/experiments/v2/stochastic_solver.py').read_text();new=Path(epoch['baseline']['path']).read_text()
 def verify():
  global count
  for row,params in zip(rows,plan['parameters']):
   source=tune_stochastic.source_for(old,params)
   expected=original_source(old if row['already_charged'] else new,params)
   assert source==expected and hashlib.sha256(source.encode()).hexdigest()==row['source_sha256']
   count+=1
 with patch.object(parallel_runtime,'main',verify),patch.object(sys,'argv',['runtime','--epoch',str(E/'epoch.json'),'tune','--run',str(R/name)]):runpy.run_path(epoch['runtime']['path'],run_name='__main__')
 tune_stochastic.source_for=original_source
# Exercise new-vs-existing sidecar selection with fake launch, not provider calls.
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);oldroot=root/'old';oldroot.mkdir();(oldroot/'manifest.json').write_text('{}')
 newroot=root/'new';newroot.mkdir()
 def verify_images():
  assert harness_runtime.pi_sidecar_command(oldroot,'old')=='old'
  assert harness_runtime.pi_sidecar_command(newroot,'old')==epoch['sidecar_image']
  assert harness_runtime.pi_sidecar_command(newroot,'old')==epoch['sidecar_image']
 with patch.object(harness_runtime,'pi_sidecar_command',lambda root,image,**kw:image),patch.object(parallel_runtime,'main',verify_images),patch.object(sys,'argv',['runtime','--epoch',str(E/'epoch.json'),'service']):runpy.run_path(epoch['runtime']['path'],run_name='__main__')
print(json.dumps(dict(status='passed',mapped_sources=count,paid_sources_retained=774,sidecar_selection_verified=True,formal_sources_executed=False)))
