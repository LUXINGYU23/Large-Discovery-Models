import time
import numpy as np
import skfmm


def _distance(mask, spacing):
    return skfmm.distance(np.where(mask, 1.0, -1.0), dx=tuple(spacing))


def _observed(problem):
    out = []
    for ob in problem.get("observations", []):
        a = np.asarray(ob.get("values"))
        good = np.isfinite(a)
        if np.any(good):
            out.append((a, good, float(ob.get("weight", 1.0))))
    return out


def _data_loss(phi, observations):
    total = 0.0
    for a, good, weight in observations:
        y = a[good]
        d = phi[good]
        total += weight * (np.logaddexp(0.0, d[y == 0]).sum()
                           + np.logaddexp(0.0, -d[y > 0]).sum())
        c = phi[a == 0.5]
        if c.size:
            total += weight * (float(c.mean()) ** 2 + float(np.mean(c * c)))
    return float(total)


def _sketch_loss(mask, reference):
    if reference is None or np.asarray(reference).shape != mask.shape:
        return 0.0
    grids = np.meshgrid(*[np.arange(n, dtype=float) for n in mask.shape], indexing="ij")
    coords = np.stack([g.reshape(-1, order="F") for g in grids], axis=1)
    aa = coords[np.asarray(reference).reshape(-1, order="F") > 0.5]
    bb = coords[mask.reshape(-1, order="F") > 0.0]
    if len(aa) < 2 or len(bb) < 2:
        return 1.0e6
    aa -= aa.mean(0)
    bb -= bb.mean(0)
    na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
    if not na or not nb:
        return 1.0e6
    aa, bb = aa / na, bb / nb
    n = max(len(aa), len(bb))
    aa = np.pad(aa, ((0, n - len(aa)), (0, 0)))
    bb = np.pad(bb, ((0, n - len(bb)), (0, 0)))
    singular = np.linalg.svd(aa.T @ bb, compute_uv=False)
    return float(5.0 * (1.0 - float(singular.sum()) ** 2))


def _objective(mask, spacing, observations, reference):
    phi = _distance(mask, spacing)
    return _data_loss(phi, observations) + _sketch_loss(phi > 0.0, reference), phi


def _shifted(source, shift):
    ans = np.zeros_like(source, dtype=bool)
    src = np.argwhere(source)
    q = src + np.asarray(shift, dtype=int)
    ok = np.all((q >= 0) & (q < np.asarray(source.shape)), axis=1)
    if np.any(ok):
        q = q[ok]
        ans[tuple(q.T)] = True
    return ans


def _translation_search(sketch, spacing, observations, budget_end):
    """Search translations and a few small rotations of the sketch."""
    source = np.asarray(sketch) > 0.5
    pts = np.argwhere(source).astype(float)
    if len(pts) < 2:
        return []
    shape = np.asarray(source.shape)
    top = []

    def add(mask):
        if time.perf_counter() > budget_end or np.count_nonzero(mask) < 2:
            return
        val, _ = _objective(mask, spacing, observations, sketch)
        top.append((val, mask))
        if len(top) > 8:
            top.sort(key=lambda x: x[0])
            del top[8:]

    def translate(base):
        bp = np.argwhere(base)
        lo, hi = -bp.min(axis=0), shape - 1 - bp.max(axis=0)
        ranges = []
        for k in range(source.ndim):
            vals = np.arange(lo[k], hi[k] + 1, dtype=int)
            if len(vals) > 81:
                vals = np.arange(-40, 41, dtype=int)
                vals = vals[(vals >= lo[k]) & (vals <= hi[k])]
            ranges.append(vals)
        for index in np.ndindex(*(len(x) for x in ranges)):
            if time.perf_counter() > budget_end:
                return
            sh = tuple(int(ranges[k][index[k]]) for k in range(source.ndim))
            add(_shifted(base, sh))

    translate(source)
    # Procrustes is also invariant to rigid rotation.  Rasterizing a small
    # angular family lets the observations choose a better boundary slope.
    if source.ndim == 2 and time.perf_counter() < budget_end:
        sp = np.asarray(spacing, dtype=float)
        pc = pts * sp
        center = pc.mean(axis=0)
        for angle in (-0.12, -0.08, -0.04, 0.04, 0.08, 0.12):
            co, si = np.cos(angle), np.sin(angle)
            rot = np.array(((co, -si), (si, co)))
            qq = (pc - center) @ rot.T + center
            ij = np.rint(qq / sp).astype(int)
            good = np.all((ij >= 0) & (ij < shape), axis=1)
            base = np.zeros_like(source)
            if np.count_nonzero(good) < 0.8 * len(ij):
                continue
            base[tuple(ij[good].T)] = True
            translate(base)
    top.sort(key=lambda x: x[0])
    return [m for _, m in top]


def _boundary(mask):
    b = np.zeros_like(mask, dtype=bool)
    for ax in range(mask.ndim):
        for side in (-1, 1):
            q = np.roll(mask, side, axis=ax)
            sl = [slice(None)] * mask.ndim
            sl[ax] = 0 if side == -1 else -1
            q[tuple(sl)] = mask[tuple(sl)]
            b |= q != mask
    return b


def _near_candidates(mask, coords, radius=8):
    candidates = np.argwhere(_boundary(mask))
    if len(candidates):
        near = np.any(np.max(np.abs(candidates[:, None, :] - coords[None, :, :]), axis=2) <= radius, axis=1)
        candidates = candidates[near]
    return candidates


def _local_refinement(mask, spacing, observations, reference, budget_end):
    if not observations:
        return mask
    coords = []
    for a, good, _ in observations:
        coords.extend(np.argwhere(good).tolist())
    if not coords:
        return mask
    coords = np.asarray(coords, dtype=int)
    best = mask.copy()
    best_value, _ = _objective(best, spacing, observations, reference)
    for _ in range(7):
        if time.perf_counter() > budget_end:
            break
        candidates = _near_candidates(best, coords)
        if len(candidates) == 0:
            break
        if len(candidates) > 700:
            candidates = candidates[:700]
        chosen, chosen_value = None, best_value
        for q in candidates:
            if time.perf_counter() > budget_end:
                break
            trial = best.copy()
            trial[tuple(q)] = ~trial[tuple(q)]
            value, _ = _objective(trial, spacing, observations, reference)
            if value + 1.0e-10 < chosen_value:
                chosen, chosen_value = trial, value
        if chosen is None:
            break
        best, best_value = chosen, chosen_value
    return best


def _pair_refinement(mask, spacing, observations, reference, budget_end):
    """Escape one-pixel local minima with bounded two-pixel moves."""
    if reference is None or not observations:
        return mask
    coords = []
    for a, good, _ in observations:
        coords.extend(np.argwhere(good).tolist())
    if not coords:
        return mask
    coords = np.asarray(coords, dtype=int)
    best = mask.copy()
    best_value, _ = _objective(best, spacing, observations, reference)
    for _ in range(8):
        if time.perf_counter() > budget_end:
            break
        candidates = _near_candidates(best, coords)
        if len(candidates) > 130:
            candidates = candidates[:130]
        chosen, chosen_value = None, best_value
        for i in range(len(candidates)):
            if time.perf_counter() > budget_end:
                break
            for j in range(i + 1, len(candidates)):
                if time.perf_counter() > budget_end:
                    break
                trial = best.copy()
                trial[tuple(candidates[i])] = ~trial[tuple(candidates[i])]
                trial[tuple(candidates[j])] = ~trial[tuple(candidates[j])]
                value, _ = _objective(trial, spacing, observations, reference)
                if value + 1.0e-10 < chosen_value:
                    chosen, chosen_value = trial, value
        if chosen is None:
            break
        best, best_value = chosen, chosen_value
    return best


def solve(problem, budget, seed):
    spacing = tuple(float(x) for x in problem["spacing"])
    initial = np.asarray(problem["initial_level_set"], dtype=float)
    observations = _observed(problem)
    reference = problem.get("sketch", None)
    wall = float((budget or {}).get("wall_time_seconds", 60.0))
    end = time.perf_counter() + max(1.0, min(wall * 0.80, wall - 0.25))
    mask = initial > 0.0
    if reference is not None and np.count_nonzero(np.asarray(reference) > 0.5) >= 2:
        candidates = _translation_search(np.asarray(reference), spacing, observations, end)
        if candidates:
            best, best_value = candidates[0], np.inf
            for candidate in candidates:
                if time.perf_counter() >= end:
                    break
                refined = _local_refinement(candidate, spacing, observations, reference, end)
                value, _ = _objective(refined, spacing, observations, reference)
                if value < best_value:
                    best, best_value = refined, value
            mask = best
            if time.perf_counter() < end:
                mask = _pair_refinement(mask, spacing, observations, reference, end)
    elif observations:
        # Unknown labels are not negatives; only measured cells are changed.
        for a, good, _ in observations:
            for q in np.argwhere(good):
                mask[tuple(q)] = bool(a[tuple(q)] > 0.0)
    if observations and time.perf_counter() < end and reference is None:
        mask = _local_refinement(mask, spacing, observations, reference, end)
    if not np.any(mask):
        mask.flat[np.argmin(initial)] = True
    if np.all(mask):
        mask.flat[np.argmin(initial)] = False
    return np.where(mask, 1.0, -1.0).astype(np.float64)
