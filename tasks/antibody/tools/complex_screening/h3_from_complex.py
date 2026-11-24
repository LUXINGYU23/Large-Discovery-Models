"""Extract an H3 only after checking the complete fixed heavy/light framework."""
import argparse
import json
from pathlib import Path


def ca_chains(path):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.Data.PDBData import protein_letters_3to1
    d = MMCIF2Dict(str(path))
    chains = {}
    for chain, number, residue, atom in zip(d['_atom_site.label_asym_id'],
            d['_atom_site.label_seq_id'], d['_atom_site.label_comp_id'], d['_atom_site.label_atom_id']):
        if atom != 'CA' or not number.isdigit():
            continue
        number = int(number)
        records = chains.setdefault(chain, {})
        if number in records:
            raise ValueError('Duplicate CA in chain ' + chain)
        if residue not in protein_letters_3to1:
            raise ValueError('Noncanonical protein residue ' + residue)
        records[number] = protein_letters_3to1[residue]
    return chains


def extract(structure, scaffold, heavy='B', light='A', heavy_stop=120, light_stop=107,
            h3_start=99, h3_stop=109):
    if not 1 <= h3_start <= h3_stop <= heavy_stop or light_stop < 1 or heavy == light:
        raise ValueError('Invalid explicit framework/H3 mapping')
    native = ca_chains(scaffold)
    try:
        expected_heavy = ''.join(native[heavy][i] for i in range(1, heavy_stop + 1))
        expected_light = ''.join(native[light][i] for i in range(1, light_stop + 1))
    except KeyError as exc:
        raise ValueError('Scaffold is missing an explicitly required framework residue') from exc
    start, stop = h3_start - 1, h3_stop
    candidates, lights = [], []
    for chain, residues in ca_chains(structure).items():
        # BoltzGen retains numbering gaps around replaced loops. Match the complete
        # framework by sequence ordinal, then report the actual output residue IDs.
        if list(residues) != sorted(residues):
            continue
        seq = ''.join(residues.values())
        if seq == expected_light:
            lights.append(chain)
        if len(seq) == heavy_stop and seq[:start] == expected_heavy[:start] and seq[stop:] == expected_heavy[stop:]:
            candidates.append((chain, seq[start:stop], list(residues)[start:stop]))
    if len(candidates) != 1 or len(lights) != 1 or candidates[0][0] == lights[0]:
        raise ValueError('Cannot uniquely identify an unchanged complete heavy/light framework')
    return {'h3': candidates[0][1], 'heavy_chain': candidates[0][0], 'light_chain': lights[0],
            'h3_label_seq_ids': candidates[0][2],
            'framework_sequence_unchanged': True, 'required_h3_length': stop - start,
            'note': 'Sequence/mapping check only; coordinate preservation and binding require separate validation.'}


def _framework_alignment(structure, scaffold, mapping, heavy='B', light='A', heavy_stop=120,
                   light_stop=107, h3_start=99, h3_stop=109):
    import numpy as np
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    def coordinates(path):
        d = MMCIF2Dict(str(path))
        chains = {}
        for chain, number, atom, x, y, z in zip(d['_atom_site.label_asym_id'],
                d['_atom_site.label_seq_id'], d['_atom_site.label_atom_id'],
                d['_atom_site.Cartn_x'], d['_atom_site.Cartn_y'], d['_atom_site.Cartn_z']):
            if atom == 'CA' and number.isdigit():
                chains.setdefault(chain, []).append([float(x), float(y), float(z)])
        return {k: np.asarray(v) for k, v in chains.items()}

    native, predicted = coordinates(scaffold), coordinates(structure)
    fixed = np.r_[0:h3_start-1, h3_stop:heavy_stop]
    x = np.concatenate([predicted[mapping['heavy_chain']][fixed], predicted[mapping['light_chain']]])
    y = np.concatenate([native[heavy][:heavy_stop][fixed], native[light][:light_stop]])
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid fixed-framework coordinates')
    x_mean, y_mean = x.mean(0), y.mean(0)
    x, y = x - x_mean, y - y_mean
    if np.linalg.matrix_rank(x) < 3 or np.linalg.matrix_rank(y) < 3:
        raise ValueError('Degenerate fixed-framework geometry')
    u, _, v = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ v)
    rotation = u @ v
    return rotation, y_mean - x_mean @ rotation, float(np.sqrt(np.mean(np.sum((x @ rotation - y)**2, axis=1))))


def framework_rmsd(*args, **kwargs):
    return _framework_alignment(*args, **kwargs)[2]


def project_fixed_framework(structure, scaffold, output, mapping, heavy='B', light='A',
                            heavy_stop=120, light_stop=107, h3_start=99, h3_stop=109):
    """Keep raw generation separately; enforce the fixed template on proposal coordinates.

    This is for generated proposals, never for predictor structure/PAE pairs.
    """
    import numpy as np
    import gemmi
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    rotation, translation, raw_rmsd = _framework_alignment(
        structure, scaffold, mapping, heavy, light, heavy_stop, light_stop, h3_start, h3_stop)
    d, native = MMCIF2Dict(str(structure)), MMCIF2Dict(str(scaffold))

    def atoms(data):
        return list(zip(data['_atom_site.label_asym_id'], data['_atom_site.label_seq_id'],
                        data['_atom_site.label_atom_id']))

    def xyz(data):
        return np.asarray(list(zip(data['_atom_site.Cartn_x'], data['_atom_site.Cartn_y'],
                                   data['_atom_site.Cartn_z'])), dtype=float)

    native_coords = {}
    for key, coord in zip(atoms(native), xyz(native)):
        native_coords.setdefault(key, coord)
    keys = atoms(d)
    coordinates = xyz(d) @ rotation + translation
    heavy_ids = [n for chain, n, atom in keys if chain == mapping['heavy_chain'] and atom == 'CA']
    light_ids = [n for chain, n, atom in keys if chain == mapping['light_chain'] and atom == 'CA']
    fixed = {(mapping['heavy_chain'], n): (heavy, str(i)) for i, n in enumerate(heavy_ids, 1)
             if not h3_start <= i <= h3_stop}
    fixed.update({(mapping['light_chain'], n): (light, str(i)) for i, n in enumerate(light_ids, 1)})
    restored = 0
    for i, (chain, number, atom) in enumerate(keys):
        ref = fixed.get((chain, number))
        if ref is None:
            continue
        key = (*ref, atom)
        if key in native_coords:
            coordinates[i] = native_coords[key]
            restored += 1
        elif atom in ('N', 'CA', 'C', 'O'):
            raise ValueError('Missing required native fixed-backbone atom')
    indexed = {key: coordinates[i] for i, key in enumerate(keys)}
    distances = []
    for left, right in ((h3_start - 2, h3_start - 1), (h3_stop - 1, h3_stop)):
        if left < 0 or right >= len(heavy_ids):
            continue
        c = indexed[(mapping['heavy_chain'], heavy_ids[left], 'C')]
        n = indexed[(mapping['heavy_chain'], heavy_ids[right], 'N')]
        distance = float(np.linalg.norm(c - n))
        if not 1.0 <= distance <= 1.8:
            raise ValueError('Invalid H3/framework peptide boundary after projection')
        distances.append(distance)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve native metadata, including valid quoted empty values that
    # Biopython's dictionary writer cannot round-trip.
    document = gemmi.cif.read_file(str(structure))
    block = document.sole_block()
    for axis, column in enumerate(('Cartn_x', 'Cartn_y', 'Cartn_z')):
        values = block.find_values('_atom_site.' + column)
        for i, value in enumerate(coordinates[:, axis]):
            values[i] = f'{value:.3f}'
    document.write_file(str(output))
    return dict(raw_framework_ca_rmsd_angstrom=raw_rmsd, restored_atom_count=restored,
                peptide_boundary_c_n_distances_angstrom=distances,
                method='Rigidly align raw complex to template, then restore matching fixed-framework atom coordinates. H3 coordinates and sequence are retained; raw model output remains separate.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('structure', type=Path)
    p.add_argument('scaffold', type=Path)
    p.add_argument('--heavy', default='B'); p.add_argument('--light', default='A')
    p.add_argument('--heavy-stop', type=int, default=120); p.add_argument('--light-stop', type=int, default=107)
    p.add_argument('--h3-start', type=int, default=99); p.add_argument('--h3-stop', type=int, default=109)
    a = p.parse_args()
    print(json.dumps(extract(a.structure, a.scaffold, a.heavy, a.light, a.heavy_stop,
                             a.light_stop, a.h3_start, a.h3_stop), indent=2))
