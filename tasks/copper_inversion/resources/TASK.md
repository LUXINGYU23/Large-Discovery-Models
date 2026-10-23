# Public geological inversion task — objective_v1

Submit solver.py defining solve(problem, budget, seed) -> floating numpy ndarray.
Public files: problem.json metadata, problem.npz arrays, objective.py and geometry.py scoring formula, selfcheck.py.
For local experiments, add /workspace/.ldm-resources to sys.path and import
copper.objective; the controller still recomputes all official scores independently.
The input dict contains shape, axis_names, origin, spacing, initial_level_set,
observations (name, values, mask, weight), sketch or None, active_mask or None.
NaN means unknown. Values 0/1 are outside/inside, 0.5 marks contact.
2D axes x/z, 3D x/y/z. Coordinates are cell centers: origin+(index+.5)*spacing.
For 3D the UBC decoder reverses its storage z ordering to ascending physical z.
Geometry is the output zero isosurface. Positive means interior. Host reinitializes
with skfmm.distance and then computes the fixed objective; arbitrary amplitude
cannot be used to lower loss. Preserve both signs and finite numbers.

Loss is weighted logistic plus squared mean and mean squared contact distance.
Contact points ALSO contribute as positives to logistic, matching upstream.
2D has an additional weight-5 normalized Procrustes shape difference with the sketch;
see exact ordering and formula in objective.py. 3D weights are 1 drillhole and 0.1
outcrop. active_mask is diagnostic only in v1. Initial geometry is a weak baseline.

Formal libraries: numpy 1.26.4, scipy 1.14.1, scikit-fmm 2025.6.23, discretize 0.11.3.
Formal budget: 2 CPU/4 GiB; 2D 60s, 3D 120s; seed 47. There is no network or API key.
Use algorithms rather than a hardcoded target array: small 2D/3D fixture grids are
part of mandatory admission. The public selfcheck uses only NumPy and is a development
aid; controller validates the sealed source independently, outside candidate Python.
The official score is recomputed in a separate process with no solver source mounted.

history.json contains every measured source and feedback. Successful evaluations include
model_slices: sampled central-plane sign maps with axis/index/physical coordinates,
model digest and seed. These previews are not full-resolution geometry or scores.
Resource diagnostics explicitly distinguish sampled memory/CPU from lifetime totals. Lower loss_total is better.
Valid weak scores are observations. Unmeasured/invalid is not a zero score. Refer to
exact parent digests and explain what changed and why. Research resources and tests
are read-only. Work in /workspace/candidates/0/solver.py; submit via submit_solver.
The submission is {"candidates": [{"artifact_path": "candidates/0/solver.py",
"parent_sha256": null, "change_summary": "...", "rationale": "..."}]}.
Submit exactly one candidate per researcher; use a measured source digest instead
of null when improving that source. The flat object format is not accepted.

Geometry diagnostics: positive_area (2D) or positive_volume (3D) equals positive
cell count times the product of physical grid spacing; this is a voxel estimate.
Where active_mask exists, air_positive_voxels and air_positive_area/volume count
positive cells outside it; air_fraction_of_positive uses all positive cells as
the denominator. Missing active_mask means air diagnostics are unavailable and
omitted, never assumed zero. These diagnostics do not change loss_total.
