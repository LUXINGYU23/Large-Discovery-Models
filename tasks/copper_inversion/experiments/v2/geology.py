"""Trusted GemPy geology generation. Never mount this directory in solver containers.

Development recipes cover folds, closed lenses, multiple thin bodies and faults.
Population quality gates and independent OOD recipes still require qualification.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import gempy as gp

RECIPE = "gempy-folded-layer-development-v3"


def folded_parent(seed):
    geometry_stream, terrain_stream = np.random.SeedSequence(seed).spawn(2)
    rng, terrain_rng = np.random.default_rng(geometry_stream), np.random.default_rng(terrain_stream)
    shape = tuple(int(v) for v in rng.integers([24, 24, 20], [37, 37, 31]))
    spacing = rng.uniform([1.5, 2., 1.], [4., 5., 3.])
    origin = rng.uniform([-200., -200., -150.], [200., 200., -50.])
    extent = np.asarray(shape) * spacing
    center = rng.uniform(.35, .6)
    thickness = rng.uniform(.12, .22)
    amplitude = rng.uniform(.05, .15)
    angle = rng.uniform(0, 2*np.pi)
    phase = rng.uniform(-np.pi, np.pi)
    slope = rng.uniform(-.12, .12, 2)
    x, y = np.meshgrid(np.linspace(.08,.92,5), np.linspace(.08,.92,5), indexing="ij")
    u = (x-.5)*np.cos(angle)+(y-.5)*np.sin(angle)
    z = center + amplitude*np.cos(2*np.pi*u+phase) + slope[0]*(x-.5)+slope[1]*(y-.5)
    dx = -amplitude*2*np.pi*np.sin(2*np.pi*u+phase)*np.cos(angle)+slope[0]
    dy = -amplitude*2*np.pi*np.sin(2*np.pi*u+phase)*np.sin(angle)+slope[1]
    controls = []
    for offset in (thickness/2, -thickness/2):
        controls.append(origin + np.stack([x.ravel(),y.ravel(),(z+offset).ravel()],axis=1)*extent)
    poles = np.stack([-dx.ravel()*extent[2]/extent[0],
                       -dy.ravel()*extent[2]/extent[1], np.ones(x.size)],axis=1)
    poles /= np.linalg.norm(poles,axis=1)[:,None]
    metadata=dict(recipe=RECIPE,family="folded",ore_lithology_id=2,sigmoid_slope=1_000_000_000,transform="GlobalAnisotropy.NONE",seed=int(seed),shape=shape,spacing=spacing.tolist(),origin=origin.tolist(),
        control_points=[p.tolist() for p in controls],orientation_poles=poles[[0,4,12,20,24]].tolist(),
        orientation_indices=[0,4,12,20,24],
        versions={name:importlib.metadata.version(name) for name in ("gempy","gempy-engine","numpy")})
    model = model_from_metadata(metadata)
    xyz = np.stack(np.meshgrid(*[origin[i]+(np.arange(n)+.5)*spacing[i] for i,n in enumerate(shape)],indexing="ij"),axis=-1)
    gp.set_custom_grid(model.grid,xyz.reshape(-1,3),reset=True)
    gp.compute_model(model)
    lithology = np.asarray(model.solutions.raw_arrays.custom)
    if lithology.shape != (int(np.prod(shape)),) or not np.isfinite(lithology).all() or not np.allclose(lithology,np.rint(lithology)):
        raise ValueError(f"GemPy custom-grid lithology is not discrete: shape={lithology.shape}, max residual={np.max(np.abs(lithology-np.rint(lithology)))}")
    lithology=np.rint(lithology).astype(np.int16).reshape(shape)
    normalized=(xyz-origin)/extent
    terrain_parameters=terrain_rng.uniform(-1,1,4)
    surface = .77 + .09*terrain_parameters[0]*np.sin(2*np.pi*normalized[:,:,0,0]+terrain_parameters[1]) + .06*terrain_parameters[2]*(normalized[:,:,0,1]-.5)
    forbidden=normalized[:,:,:,2] > surface[:,:,None]
    truth=(lithology==2)&~forbidden
    fraction=float(truth.sum()/(~forbidden).sum())
    if not .02 <= fraction <= .65:
        raise ValueError("predeclared geology quality gate: ore fraction outside [0.02,0.65]")
    metadata.update(terrain_parameters=terrain_parameters.tolist(),positive_fraction=fraction)
    return metadata,truth,forbidden


def model_from_metadata(metadata):
    """Reconstruct the same implicit parent for arbitrary off-grid measurements."""
    shape=metadata["shape"]; origin=np.asarray(metadata["origin"])
    extent=np.asarray(shape)*metadata["spacing"]
    controls=[np.asarray(p) for p in metadata["control_points"]]
    frame=gp.data.StructuralFrame.initialize_default_structure()
    model=gp.create_geomodel(project_name="private-parent",extent=np.column_stack([origin,origin+extent]).ravel().tolist(),
        resolution=list(shape),structural_frame=frame)
    p=controls[0]; ids=np.asarray(metadata["orientation_indices"])
    gp.add_surface_points(geo_model=model,x=p[:,0],y=p[:,1],z=p[:,2],elements_names="surface1")
    gp.add_orientations(geo_model=model,x=p[ids,0],y=p[ids,1],z=p[ids,2],
        elements_names=["surface1"]*len(ids),pole_vector=np.asarray(metadata["orientation_poles"]))
    for i,q in enumerate(controls[1:],start=2):
        model.structural_frame.structural_groups[0].append_element(gp.data.StructuralElement(
            name=f"surface{i}",color="#884422",
            surface_points=gp.data.SurfacePointsTable.from_arrays(x=q[:,0],y=q[:,1],z=q[:,2],names=f"surface{i}"),
            orientations=gp.data.OrientationsTable.initialize_empty()))
    if "fault" in metadata:
        fault=metadata["fault"]; p=np.asarray(fault["points"]); normal=np.asarray(fault["normal"])
        element=gp.data.StructuralElement(name="fault",color="#777777",
            surface_points=gp.data.SurfacePointsTable.from_arrays(x=p[:,0],y=p[:,1],z=p[:,2],names="fault"),
            orientations=gp.data.OrientationsTable.from_arrays(x=p[:1,0],y=p[:1,1],z=p[:1,2],
                G_x=normal[:1],G_y=normal[1:2],G_z=normal[2:],names="fault"))
        group=gp.data.StructuralGroup(name="fault_group",elements=[element],
            structural_relation=gp.data.StackRelationType.FAULT,
            fault_relations=gp.data.FaultsRelationSpecialCase.OFFSET_ALL)
        model.structural_frame.insert_group(0,group)
    model.update_transform(gp.data.GlobalAnisotropy.NONE)
    model.interpolation_options.mesh_extraction=False
    model.interpolation_options.sigmoid_slope=metadata["sigmoid_slope"]
    return model


def query_ore(model, points, *, target_id):
    """Classify the implicit parent at physical points, without a voxel SDF."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError("nonempty physical query points required")
    gp.set_custom_grid(model.grid,points,reset=True)
    gp.compute_model(model)
    values=np.asarray(model.solutions.raw_arrays.custom)
    if values.shape!=(len(points),) or not np.isfinite(values).all() or np.any((values<target_id-1-1e-6)|(values>target_id+1+1e-6)):
        raise ValueError(f"invalid GemPy query classifications: range={values.min(),values.max()}, unique={np.unique(values)[:12]}")
    # Lithology indices include preceding fault elements. The target index is
    # part of the private structural recipe, never a public prior. Fractional
    # sigmoid transitions use their midpoint boundary, never scorer distances.
    return (values>=target_id-.5)&(values<target_id+.5)


def terrain_height(metadata, xy):
    origin=np.asarray(metadata["origin"]); extent=np.asarray(metadata["shape"])*metadata["spacing"]
    xy=(np.asarray(xy)-origin[:2])/extent[:2]
    a=metadata["terrain_parameters"]
    z=.77+.09*a[0]*np.sin(2*np.pi*xy[:,0]+a[1])+.06*a[2]*(xy[:,1]-.5)
    return origin[2]+z*extent[2]


def structured_parent(seed,family):
    if family=="folded":
        return folded_parent(seed)
    # The same public coordinate distribution across families prevents shape or
    # units from serving as a hidden family label.
    streams=np.random.SeedSequence(seed).spawn(2)
    rng,terrain_rng=map(np.random.default_rng,streams)
    shape=tuple(int(v) for v in rng.integers([24,24,20],[37,37,31]))
    spacing=rng.uniform([1.5,2.,1.],[4.,5.,3.]);origin=rng.uniform([-200.,-200.,-150.],[200.,200.,-50.])
    extent=np.asarray(shape)*spacing
    if family in {"lens","multiple"}:
        controls=[];normals=[]
        count=1 if family=="lens" else 2
        azimuth=rng.uniform(0,2*np.pi)
        for j in range(count):
            center=rng.uniform([.38,.38,.35],[.62,.62,.55])
            if count==2:center[:2]=.5+(-.23+.46*j)*np.array([np.cos(azimuth),np.sin(azimuth)])
            radii=rng.uniform([.18,.19,.13],[.29,.32,.22])
            if count==2:
                radii[0]=rng.uniform(.13,.18)
                radii[2]=rng.uniform(.045,.075)
            # Non-axis-aligned ellipsoid boundary constraints, interpolated by
            # GemPy. The analytic surface is not evaluated as the ground truth.
            angle=rng.uniform(-.5,.5) if count==1 else rng.choice([-1,1])*rng.uniform(.4,.9)
            tilt=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
            strike=np.array([[np.cos(azimuth),-np.sin(azimuth),0],[np.sin(azimuth),np.cos(azimuth),0],[0,0,1.]])
            rotation=strike@tilt
            i=np.arange(42);z=1-2*(i+.5)/42;theta=i*np.pi*(3-np.sqrt(5))
            directions=np.column_stack([np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z])
            points=center+(directions*radii)@rotation.T
            poles=(directions/radii)@rotation.T/extent
            poles/=np.linalg.norm(poles,axis=1)[:,None]
            controls.extend(origin+points*extent);normals.extend(poles)
        metadata=dict(control_points=[np.asarray(controls).tolist()],orientation_indices=list(range(len(controls))),
            orientation_poles=np.asarray(normals).tolist())
    elif family in {"faulted", "faulted_lens"}:
        metadata,_,_=(folded_parent(seed) if family=="faulted" else structured_parent(seed,"lens"))
        points=np.asarray(metadata["control_points"])
        angle=rng.uniform(0,2*np.pi)
        normal=np.array([np.cos(angle),np.sin(angle),0.])
        tangent=np.array([-np.sin(angle),np.cos(angle),0.])
        center=np.array([*rng.uniform(.42,.58,2),.5])
        throw=rng.uniform(.1,.18)*extent[2]
        side=np.sum(((points-origin)/extent-center)*normal,axis=-1)>0
        points[:,:,2]+=np.where(side,throw/2,-throw/2)
        metadata["control_points"]=points.tolist()
        plane=np.stack([center-.4*tangent+[0,0,-.35],center+.4*tangent+[0,0,-.35],center+[0,0,.35]])
        physical_normal=normal/extent;physical_normal/=np.linalg.norm(physical_normal)
        metadata["fault"]=dict(points=(origin+plane*extent).tolist(),normal=physical_normal.tolist(),throw=throw,strike_radians=angle)
    else:
        raise ValueError("unknown geology family")
    metadata.update(recipe=f"gempy-{family}-development-v3",family=family,ore_lithology_id=3 if "fault" in metadata else 2,seed=int(seed),shape=shape,
        spacing=spacing.tolist(),origin=origin.tolist(),sigmoid_slope=1_000_000_000,transform="GlobalAnisotropy.NONE",
        terrain_parameters=terrain_rng.uniform(-1,1,4).tolist(),
        versions={name:importlib.metadata.version(name) for name in ("gempy","gempy-engine","numpy")})
    model=model_from_metadata(metadata)
    xyz=np.stack(np.meshgrid(*[origin[i]+(np.arange(n)+.5)*spacing[i] for i,n in enumerate(shape)],indexing="ij"),axis=-1)
    ore=query_ore(model,xyz.reshape(-1,3),target_id=metadata["ore_lithology_id"]).reshape(shape)
    terrain=terrain_height(metadata,xyz[:,:,0,:2].reshape(-1,2)).reshape(shape[:2])
    forbidden=xyz[:,:,:,2]>terrain[:,:,None]
    truth=ore&~forbidden
    fraction=float(truth.sum()/(~forbidden).sum())
    if not .02<=fraction<=.65:
        raise ValueError(f"predeclared geology quality gate failed: {fraction}")
    metadata["positive_fraction"]=fraction
    return metadata,truth,forbidden


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",choices=["folded","lens","multiple","faulted","faulted_lens"],default="folded")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    # New parent only: never overwrite previously inspected/generated evidence.
    args.output.mkdir(parents=True,exist_ok=False)
    metadata,truth,forbidden=structured_parent(args.seed,args.family)
    np.savez(args.output/"geology.npz",truth=truth,forbidden_mask=forbidden)
    metadata["geometry_sha256"]=hashlib.sha256((args.output/"geology.npz").read_bytes()).hexdigest()
    (args.output/"private-parent.json").write_text(json.dumps(metadata,sort_keys=True,indent=2))
    print(json.dumps({k:metadata[k] for k in ("recipe","shape","positive_fraction","geometry_sha256")}))


if __name__=="__main__":
    main()
