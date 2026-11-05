"""Run inside the same baseline image on captured public development proposals."""
import ast,json
from pathlib import Path
import numpy as np,skfmm
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.v2.data import read_problem
root=Path('/audit')
tree=ast.parse((root/'stochastic_solver_fixed.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='search_distance')
ns=dict(np=np,skfmm=skfmm,canonicalize=canonicalize)
exec(compile(ast.Module(body=[node],type_ignores=[]),'helper','exec'),ns)
checks=[]
for f in (root/'numeric-capture').rglob('drift.npz'):
 a=np.load(f);field=a['field'];sp=a['spacing']
 # Unit-scaled first order must retain signs without altering occupancy.
 for scale in [.001,1.,1000.]:
  d=skfmm.distance(field*scale,dx=tuple(sp*scale),order=1)
  assert np.array_equal(d>0,field>0)
  checks.append(dict(job=f.parent.parent.name,scale=scale,signs_preserved=True))
# Exercise actual helper on fixtures: legal subvoxel motion and hard rejection.
p=next(fixtures());p['forbidden_mask']=np.zeros(p['shape'],bool);p['open_boundary_faces']=['x-','x+','y-','y+','z-','z+']
f=np.broadcast_to((np.arange(p['shape'][0])-3.5)[:,None,None],p['shape']).copy()*p['spacing'][0];initial=f>0
for _ in range(10):f=ns['search_distance'](p,f-.1*p['spacing'][0])
assert np.any((f>0)!=initial);checks.append({'subvoxel_motion':True})
p['forbidden_mask'][-1]=True
try:ns['search_distance'](p,f)
except ValueError:checks.append({'illegal_geometry_rejected':True})
else:raise AssertionError('illegal geometry accepted')
(root/'numeric-regression.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2));print('passed',len(checks))
