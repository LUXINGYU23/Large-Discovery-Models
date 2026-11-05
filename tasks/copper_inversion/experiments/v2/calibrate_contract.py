"""Prespecified analytic acceptance checks; no benchmark outcomes are inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.objective import score
from .measurement_policy import scales


def check():
    # Contact error criterion is a declared empirical tolerance, not an SDF theorem.
    criteria = dict(max_contact_error_in_effective_scale=2.,
                    max_clean_label_error_away_from_interface=.01,
                    unit_conversion_absolute_tolerance=1e-9,
                    forbidden_positive_must_reject=True)
    rows=[]
    rng=np.random.default_rng(739102)
    for spacing in ([2.,2.,2.],[1.,1.,1.],[1.,1.5,2.]):
        h=np.asarray(spacing);shape=np.ceil(40/h).astype(int);origin=-shape*h/2
        xyz=np.stack(np.meshgrid(*[origin[k]+(np.arange(n)+.5)*h[k] for k,n in enumerate(shape)],indexing='ij'),axis=-1)
        for family in ('sphere','thin_slab','oblique_slab','faulted_slab','open_boundary','terrain_clipped'):
            samples=rng.uniform(-12,12,(300,3))
            if family=='sphere':
                field=10-np.linalg.norm(xyz,axis=-1)
                contacts=samples/np.linalg.norm(samples,axis=1)[:,None]*10
                analytic=10-np.linalg.norm(samples,axis=1)
            else:
                n=np.array([.6,0,.8]) if family=='oblique_slab' else np.array([0.,0.,1.])
                # Thin body has physical thickness 6, at least 3 coarsest voxels.
                half=3. if family=='thin_slab' else 5.
                offset=17. if family=='open_boundary' else 0.
                shift=np.where(xyz[...,0]>=0,4.,-4.) if family=='faulted_slab' else 0.
                field=half-np.abs(xyz@n-offset-shift)
                contacts=samples.copy();contacts[:,2]=0
                side=rng.choice([-1.,1.],len(samples))
                displacement=np.where(contacts[:,0]>=0,4.,-4.) if family=='faulted_slab' else 0.
                # The high contact of an open body is outside the domain, not an observed interface.
                if family=='open_boundary':side[:]=-1
                contacts[:,2]=(offset+displacement+half*side-contacts[:,0]*n[0])/n[2]
                if family=='faulted_slab':contacts=contacts[np.abs(contacts[:,0])>3*h.max()]
                analytic=half-np.abs(samples@n-offset-(np.where(samples[:,0]>=0,4.,-4.) if family=='faulted_slab' else 0.))
            forbidden=np.zeros(shape,bool)
            if family=='terrain_clipped':
                forbidden=xyz[...,2]>(2.+.3*xyz[...,0])
                field=np.where(forbidden,-1.,field)
                contacts=contacts[contacts[:,2]<(2.+.3*contacts[:,0])-2*h.max()]
            scale=scales(h,False)['effective_scale']
            obs=dict(name='analytic',points=contacts,labels=np.full(len(contacts),.5),groups=np.arange(len(contacts)),weight=1.,transition_scale=scale,contact_sigma=scale,label_error=0.)
            p=dict(schema_version='problem_schema_v2',geometry_version='occupancy_v2',objective_version='objective_v2',case_id='analytic-contract',shape=shape.tolist(),origin=origin.tolist(),spacing=h.tolist(),domain_mask=np.ones(shape,bool),forbidden_mask=forbidden,open_boundary_faces=['x-','x+','y-','y+','z-','z+'],observations=[obs],priors=dict(area_weight=0.,volume_weight=0.,volume_bounds=None,fragment_weight=0.,fragment_volume=0.,hard_volume_bounds=None,max_components=None))
            _,distance=canonicalize(p,field.astype(float))
            residual=map_coordinates(distance,((contacts-origin)/h-.5).T,order=1,mode='nearest')
            clean=map_coordinates(distance,((samples-origin)/h-.5).T,order=1,mode='nearest')>0
            away=np.abs(analytic)>2*np.linalg.norm(h)
            if family=='faulted_slab':away &= np.abs(samples[:,0])>3*h.max()
            if family=='terrain_clipped':away &= samples[:,2]<(2.+.3*samples[:,0])-2*h.max()
            label_error=float(np.mean(clean[away]!=(analytic[away]>0))) if away.any() else 0.
            loss=score(p,field.astype(float))['loss_total']
            scaled={**p,'origin':(origin*1000).tolist(),'spacing':(h*1000).tolist(),'observations':[{**obs,'points':contacts*1000,'transition_scale':scale*1000,'contact_sigma':scale*1000}]}
            unit_error=abs(score(scaled,field.astype(float))['loss_total']-loss)
            illegal=field.astype(float).copy();probe={**p,'forbidden_mask':forbidden.copy()}
            probe['forbidden_mask'].flat[0]=True;illegal.flat[0]=1.
            try:canonicalize(probe,illegal)
            except ValueError:rejected=True
            else:rejected=False
            ratio=float(np.max(abs(residual))/scale)
            passed=(ratio<=criteria['max_contact_error_in_effective_scale'] and label_error<=.01 and unit_error<1e-9 and rejected)
            rows.append(dict(family=family,spacing=h.tolist(),contacts=len(contacts),max_contact_error_in_effective_scale=ratio,clean_label_error_away_from_interface=label_error,unit_conversion_error=unit_error,forbidden_positive_rejected=rejected,passed=passed))
    return dict(schema='copper-contract-calibration-v1',status='passed' if all(r['passed'] for r in rows) else 'failed',criteria=criteria,rows=rows,limitations=['Empirical numerical contract checks, not geological validation','Thin bodies below three coarse voxels are not qualified','Axis-face area is excluded from the objective; no isotropic surface-area claim','Independent population and hidden-data audits remain required'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check()
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
