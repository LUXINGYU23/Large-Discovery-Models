"""Remote generator API qualification; this is not a benchmark dataset."""
import json
import numpy as np
import gempy as gp


def main():
    model = gp.create_geomodel(project_name="api-qualification", extent=[0, 100, 0, 100, 0, 100],
        resolution=[12, 10, 8], structural_frame=gp.data.StructuralFrame.initialize_default_structure())
    gp.add_surface_points(geo_model=model, x=[10, 50, 90, 50], y=[50, 50, 50, 90],
        z=[55, 65, 55, 60], elements_names="surface1")
    gp.add_orientations(geo_model=model, x=[50], y=[50], z=[65],
        elements_names=["surface1"], pole_vector=[[0, 0, 1.]])
    element = gp.data.StructuralElement(name="surface2", color="#884422",
        surface_points=gp.data.SurfacePointsTable.from_arrays(x=np.array([10.,50.,90.,50.]),
            y=np.array([50.,50.,50.,90.]), z=np.array([30.,40.,30.,35.]), names="surface2"),
        orientations=gp.data.OrientationsTable.initialize_empty())
    model.structural_frame.structural_groups[0].append_element(element)
    model.interpolation_options.mesh_extraction = False
    xyz = np.stack(np.meshgrid(np.arange(12)*100/12+50/12,
        np.arange(10)*10+5, np.arange(8)*12.5+6.25, indexing="ij"), axis=-1)
    gp.set_custom_grid(model.grid, xyz.reshape(-1,3), reset=True)
    gp.compute_model(model)
    raw = model.solutions.raw_arrays
    print(json.dumps({"raw_arrays": {k: list(v.shape) for k,v in vars(raw).items() if isinstance(v,np.ndarray)},
        "custom_type": str(type(raw.custom)), "custom_shape": str(np.shape(raw.custom))}))
    print("custom_values", np.unique(raw.custom))


if __name__ == "__main__":
    main()
