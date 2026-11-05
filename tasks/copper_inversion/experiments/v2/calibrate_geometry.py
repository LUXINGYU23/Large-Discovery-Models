"""Analytic numerical calibration, never a benchmark geology generator."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.objective import score
from .measurement_policy import scales


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();rng=np.random.default_rng(20261024)
    directions=rng.normal(size=(240,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    contacts=directions*10
    binary=directions*np.linspace(6,14,len(directions))[:,None]
    points=np.concatenate([binary,contacts]);labels=np.r_[(np.linalg.norm(binary,axis=1)<10).astype(float),np.full(len(contacts),.5)]
    rows=[]
    for spacing in ([4.,4.,4.],[2.,2.,2.],[1.,1.,1.],[1.,1.5,2.]):
        h=np.asarray(spacing);shape=np.ceil(32/h).astype(int);origin=-shape*h/2
        xyz=np.stack(np.meshgrid(*[origin[k]+(np.arange(n)+.5)*h[k] for k,n in enumerate(shape)],indexing="ij"),axis=-1)
        phi=10-np.linalg.norm(xyz,axis=-1)
        problem=dict(schema_version="problem_schema_v2",geometry_version="occupancy_v2",objective_version="objective_v2",case_id="analytic-calibration",
            shape=shape.tolist(),origin=origin.tolist(),spacing=h.tolist(),domain_mask=np.ones(shape,bool),forbidden_mask=np.zeros(shape,bool),
            open_boundary_faces=["x-","x+","y-","y+","z-","z+"],observations=[dict(name="analytic",points=points,labels=labels,groups=np.arange(len(points)),weight=1.,transition_scale=.25*h.min(),contact_sigma=.25*h.min(),label_error=0.)],
            priors=dict(area_weight=.02,volume_weight=0.,volume_bounds=None,fragment_weight=0.,fragment_volume=0.,hard_volume_bounds=None,max_components=None))
        _,distance=canonicalize(problem,phi)
        reconstructed=map_coordinates(distance,((contacts-origin)/h-.5).T,order=1,mode="nearest")
        metrics=score(problem,phi)
        scaled={**problem,"origin":(origin*1000).tolist(),"spacing":(h*1000).tolist(),
            "observations":[{**problem["observations"][0],"points":points*1000,"transition_scale":250*h.min(),"contact_sigma":250*h.min()}]}
        scaled_metrics=score(scaled,phi)
        if not np.isclose(metrics["loss_total"],scaled_metrics["loss_total"],rtol=1e-10,atol=1e-10):
            raise AssertionError("public objective changes under unit conversion")
        rows.append(dict(spacing=h.tolist(),contact_rmse=float(np.sqrt(np.mean(reconstructed**2))),
            proposed_scale=scales(h,False),
            max_contact_error_over_proposed_scale=float(abs(reconstructed).max()/scales(h,False)["effective_scale"]),
            contact_rmse_in_sigma=float(np.sqrt(np.mean(reconstructed**2))/(.25*h.min())),
            max_contact_distance=float(abs(reconstructed).max()),public_loss=metrics["loss_total"],
            relative_volume_error=abs(metrics["positive_volume"]/(4*np.pi*10**3/3)-1),
            voxel_area_ratio=metrics["internal_area"]/(4*np.pi*10**2),unit_conversion_loss_difference=scaled_metrics["loss_total"]-metrics["loss_total"]))
    # Rotation sensitivity is assessed on slab geometry, with no domain padding.
    rotation=[]
    for degrees in (0,15,30,45,60,75,90):
        angle=np.deg2rad(degrees);normal=np.array([np.cos(angle),np.sin(angle),0.])
        slab=3.-np.abs(xyz@normal)
        # Only geometry diagnostic; sphere observations do not fit this slab.
        from tasks.copper_inversion.v2.geometry import diagnostics
        geometry=diagnostics(problem,slab>0)
        extent=shape*h
        corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*extent[:2]/2
        analytic=0.
        for offset in (-3.,3.):
            crossings=[]
            for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                da,db=a@normal[:2]-offset,b@normal[:2]-offset
                if abs(da)<1e-10:crossings.append(a)
                if da*db<0:crossings.append(a+(b-a)*da/(da-db))
            crossings=np.unique(np.round(crossings,10),axis=0)
            if len(crossings)!=2:raise ValueError("analytic slab clipping failed")
            analytic+=float(np.linalg.norm(crossings[1]-crossings[0])*extent[2])
        rotation.append(dict(degrees=degrees,area=geometry["internal_area"],analytic_area=analytic,
            voxel_area_ratio=geometry["internal_area"]/analytic,volume=geometry["positive_volume"]))
    result=dict(scope="analytic discretization diagnostics only; not external validation",sphere=rows,slab_rotation=rotation,
        limitations="Axis-face area is an anisotropic voxel perimeter, not Euclidean surface area. Pilot prior/noise calibration remains open.")
    with args.output.open("x") as file:json.dump(result,file,indent=2)
    print(json.dumps(result))


if __name__=="__main__":main()
