"""Run BoltzGen 0.3.2 H3 generation independently of its ranking pipeline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def snapshot_inputs(spec, scaffold, directory):
    """Freeze nested native YAML/CIF references, so later host edits cannot change a run."""
    directory.mkdir()
    seen, visiting, manifest = {}, set(), []

    def visit(source):
        source = Path(source).resolve()
        if source in visiting:
            raise ValueError('Cyclic native design specification')
        if source in seen:
            return seen[source]
        if len(seen) >= 64:
            raise ValueError('Too many native input references')
        data = source.read_bytes()
        destination = directory / f'{len(seen):03d}-{source.name}'
        seen[source] = destination
        visiting.add(source)
        if source.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            native = yaml.safe_load(data)
            if not isinstance(native, dict):
                raise ValueError('Native design specification must be a mapping')

            def rewrite(node):
                if isinstance(node, list):
                    return [rewrite(item) for item in node]
                if not isinstance(node, dict):
                    return node
                result = {}
                for key, value in node.items():
                    if key == 'path':
                        paths = value if isinstance(value, list) else [value]
                        copied = [str(visit(Path(p) if Path(p).is_absolute() else source.parent / p)) for p in paths]
                        result[key] = copied if isinstance(value, list) else copied[0]
                    else:
                        result[key] = rewrite(value)
                return result

            destination.write_text(yaml.safe_dump(rewrite(native), sort_keys=False))
        else:
            destination.write_bytes(data)
        visiting.remove(source)
        manifest.append(dict(source=str(source), source_sha256=hashlib.sha256(data).hexdigest(),
                             snapshot=str(destination.relative_to(directory.parent)),
                             snapshot_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        return destination

    return visit(spec), visit(scaffold), manifest


def worker(job):
    import importlib
    from importlib.metadata import version
    if version('boltzgen') != '0.3.2':
        raise RuntimeError('This adapter is pinned to BoltzGen 0.3.2')
    if job.get('memory_limit_gib') is not None:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, job['memory_limit_gib'] * 1024**3 / total), 0)
    from pytorch_lightning import seed_everything
    from omegaconf import OmegaConf
    entry = importlib.import_module('boltzgen.resources.main')
    seed_everything(job['seed'], workers=True)
    base = Path(entry.__file__).parent / 'config/design.yaml'
    config = OmegaConf.merge(OmegaConf.load(base), OmegaConf.from_dotlist(job['overrides']))
    (Path(job['output']) / 'resolved-design.yaml').write_text(OmegaConf.to_yaml(config))
    entry.main(str(base), job['overrides'])
    seed_everything(job['seed'], workers=True)
    inverse_base = base.with_name('inverse_fold.yaml')
    from boltzgen.data import const
    inverse_overrides = [*job['inverse_overrides'],
                         f'override.inverse_fold_args.inverse_fold_restriction=[{const.prot_letter_to_token["C"]}]']
    inverse_config = OmegaConf.merge(OmegaConf.load(inverse_base), OmegaConf.from_dotlist(inverse_overrides))
    (Path(job['output']) / 'resolved-inverse-fold.yaml').write_text(OmegaConf.to_yaml(inverse_config))
    entry.main(str(inverse_base), inverse_overrides)
    import torch
    runtime = dict(boltzgen=version('boltzgen'), torch=torch.__version__,
                   cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   max_gpu_memory_reserved_gib=torch.cuda.max_memory_reserved(0) / 1024**3)
    (Path(job['output']) / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')


def generate(a):
    from .h3_from_complex import extract, framework_rmsd, project_fixed_framework
    output = a.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    if not 0 <= a.seed < 2**32 or a.samples < 1 or a.gpu < 0:
        raise ValueError('Invalid seed, GPU, or sample count')
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    memory_limit = getattr(a, 'memory_limit_gib', None)
    if memory_limit is not None and (not math.isfinite(memory_limit) or memory_limit <= 0):
        raise ValueError('GPU memory limit must be positive and finite')
    max_rmsd = getattr(a, 'max_framework_rmsd', 0.25)
    if not math.isfinite(max_rmsd) or max_rmsd <= 0:
        raise ValueError('Framework RMSD limit must be positive and finite')
    inputs = {name: getattr(a, name).resolve() for name in ('spec', 'scaffold', 'checkpoint', 'inverse_fold_checkpoint', 'moldir')}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    frozen_spec, frozen_scaffold, source_graph = snapshot_inputs(inputs['spec'], inputs['scaffold'], output / 'inputs')
    designs = output / 'intermediate_designs'
    overrides = [f'output={designs}', f'data.cfg.output_dir={designs}',
                 'data.cfg.yaml_path=' + json.dumps([str(frozen_spec)]),
                 'data.cfg.multiplicity=1', 'data.num_workers=0',
                 'diffusion_samples=1', 'trainer.devices=1', 'override.use_kernels=false',
                 f'checkpoint={inputs["checkpoint"]}', f'data.cfg.moldir={inputs["moldir"]}']
    # Same inverse-folding defaults as upstream antibody-anything, including C avoidance.
    from_path = output / 'intermediate_designs_inverse_folded'
    inverse_overrides = [f'output={from_path}', f'data.design_dir={designs}',
                         f'data.cfg.multiplicity={a.samples}', 'data.cfg.num_workers=0',
                         'data.skip_existing=false', 'data.skip_existing_kind=inverse_fold',
                         'override.use_kernels=false', f'checkpoint={inputs["inverse_fold_checkpoint"]}',
                         f'data.cfg.moldir={inputs["moldir"]}', 'trainer.devices=1']
    job = dict(seed=a.seed, memory_limit_gib=memory_limit, output=str(output), overrides=overrides, inverse_overrides=inverse_overrides)
    (output / 'job.json').write_text(json.dumps(job, indent=2))
    # Resolving a venv's Python symlink loses its environment and selects system Python.
    command = [str(a.python.absolute()), '-m', __package__ + '.boltzgen_design',
               '--worker', str(output / 'job.json')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    root = str(Path(__file__).resolve().parents[4])
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    record = dict(tool='boltzgen_h3', version='0.3.2', status='running', seed=a.seed,
                  samples=a.samples, backbones=1, memory_limit_gib=memory_limit, command=command, steps=['design', 'inverse_folding'],
                  inverse_fold_avoid='C',
                  max_framework_rmsd_angstrom=max_rmsd,
                  hard_filter_eligible=False, ranking_pipeline_validated=False,
                  input_snapshot=source_graph,
                  inputs={k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for k, p in inputs.items()})
    start = time.monotonic()
    try:
        with (output / 'tool.log').open('wb') as log:
            subprocess.run(command, cwd=inputs['spec'].parent, env=env, stdout=log,
                           stderr=subprocess.STDOUT, timeout=a.timeout, check=True)
        if len(list(designs.glob('*.cif'))) != 1:
            raise RuntimeError('Expected one generated BoltzGen backbone')
        structures = sorted(from_path.glob('*.cif'))
        if len(structures) != a.samples:
            raise RuntimeError(f'Expected {a.samples} generated structures, received {len(structures)}')
        candidates = []
        for structure in structures:
            candidate = extract(structure, frozen_scaffold, a.heavy, a.light,
                                a.heavy_stop, a.light_stop, a.h3_start, a.h3_stop)
            rmsd = framework_rmsd(structure, frozen_scaffold, candidate, a.heavy, a.light,
                                 a.heavy_stop, a.light_stop, a.h3_start, a.h3_stop)
            if rmsd > max_rmsd:
                raise RuntimeError('Generated fixed framework exceeds coordinate RMSD limit')
            projected = output / 'projected_designs' / structure.name
            candidate['framework_projection'] = project_fixed_framework(
                structure, frozen_scaffold, projected, candidate, a.heavy, a.light,
                a.heavy_stop, a.light_stop, a.h3_start, a.h3_stop)
            final_rmsd = framework_rmsd(projected, frozen_scaffold, candidate, a.heavy, a.light,
                                        a.heavy_stop, a.light_stop, a.h3_start, a.h3_stop)
            if final_rmsd > 0.05:
                raise RuntimeError('Fixed framework projection failed coordinate verification')
            candidate['framework_ca_rmsd_angstrom'] = final_rmsd
            candidate['note'] = 'Template fixed-framework coordinates enforced; raw generation retained. Binding requires separate evaluation.'
            candidate['raw_structure'] = str(structure.relative_to(output))
            candidate['structure'] = str(projected.relative_to(output))
            candidates.append(candidate)
        (output / 'candidates.json').write_text(json.dumps(candidates, indent=2) + '\n')
        record.update(status='succeeded', candidates=candidates)
        if (output / 'runtime.json').is_file():
            record['runtime'] = json.loads((output / 'runtime.json').read_text())
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        record['seconds'] = time.monotonic() - start
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(json.loads(Path(sys.argv[2]).read_text()))
        return
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'scaffold', 'checkpoint', 'inverse-fold-checkpoint', 'moldir', 'out', 'python'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=19); p.add_argument('--samples', type=int, default=1)
    p.add_argument('--memory-limit-gib', type=float, help='Optional per-process GPU allocator cap')
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--max-framework-rmsd', type=float, default=0.25,
                   help='Maximum raw fixed-framework CA RMSD before source-coordinate projection, in angstrom')
    p.add_argument('--heavy', default='B'); p.add_argument('--light', default='A')
    p.add_argument('--heavy-stop', type=int, default=120); p.add_argument('--light-stop', type=int, default=107)
    p.add_argument('--h3-start', type=int, default=99); p.add_argument('--h3-stop', type=int, default=109)
    print(json.dumps(generate(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
