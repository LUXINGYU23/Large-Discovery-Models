"""Fixed public interface inputs; no benchmark parent, truth, or held-out data."""
import numpy as np
from .contract import validate
from .geological_policy import public_priors

FIXTURE_VERSION = "copper-v2-interface-2"


def fixtures():
    for index, contacts in enumerate((0, 1, 3, 4, 7, 2)):
        shape = (9 + index, 8, 7 + index % 2)
        scale = 1000. if index == 4 else 1.
        spacing = np.asarray([1.2, 2.1, .8]) * scale
        origin = np.asarray([-30., 40., -70.]) * scale
        forbidden = np.zeros(shape, dtype=bool)
        forbidden[:, :, -1] = True
        faces = ["x-", "x+", "y-", "y+", "z-"]
        if index == 5:
            for axis in range(3):
                for side in (0, -1):
                    view = [slice(None)] * 3
                    view[axis] = side
                    forbidden[tuple(view)] = True
            faces = []
        # Integer grid-center coordinates in physical space; no exactly-four
        # contact assumption, no unit spacing, no zero-origin assumption.
        points = [[2, 2, 2], [shape[0] - 3, 5, 2]]
        labels = [1., 0.]
        groups = [0, 1]
        for j in range(contacts):
            points.append([2 + j % (shape[0] - 4), 3, 3])
            labels.append(.5)
            groups.append(j % 3)
        problem = dict(
            schema_version="problem_schema_v2", geometry_version="occupancy_v2",
            objective_version="objective_v2", case_id=f"public-interface-{index}",
            shape=shape, origin=origin.tolist(), spacing=spacing.tolist(),
            domain_mask=np.ones(shape, dtype=bool), forbidden_mask=forbidden,
            open_boundary_faces=faces,
            observations=[dict(name="drill", points=origin + (np.asarray(points) + .5) * spacing,
                labels=np.asarray(labels), groups=np.asarray(groups), weight=1.,
                transition_scale=scale, contact_sigma=scale, label_error=0.)],
            priors=dict(area_weight=.02, volume_weight=0., volume_bounds=None,
                fragment_weight=0., fragment_volume=0., hard_volume_bounds=None, max_components=None),
        )
        if index in (0,5):
            problem['priors']=public_priors(problem['domain_mask']&~problem['forbidden_mask'],spacing)
        yield validate(problem)
