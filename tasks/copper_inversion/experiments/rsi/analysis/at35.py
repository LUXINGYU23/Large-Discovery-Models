"""Prepare and score independent rater labels; sweep rehearsal sources for AT-07."""
import argparse
import ast
import csv
import dis
import hashlib
import io
import json
import random
import tokenize
import types
from collections import Counter
from pathlib import Path

from .tagger import Tagger, load_rules, cohen_kappa, calibrate_threshold, cluster_labels
from .rundir import Run
from .measures import reference_vectors
from ....core.encoding import FEATURE_VERSION
from ....rsi.render import stripped_source


def sources(runs):
    result = {}
    for run in sorted(map(lambda p: Path(p).resolve(), runs), key=str):
        for path in sorted((run / 'candidates').glob('*/solver.py')):
            data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
            if sha != path.parent.name: raise ValueError('source identity differs: ' + str(path))
            result.setdefault(sha, (run, data))
    return result


def sample(runs, seed, count, out, rules=None):
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1: raise ValueError('invalid sample')
    rules = load_rules(rules); items = sources(runs)
    if any(not r.get("description") for r in rules["rules"]): raise ValueError("every sample rule needs a description")
    if len(items) < count: raise ValueError('not enough unique candidate programs')
    chosen = sorted(random.Random(seed).sample(sorted(items), count))
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    for sha in chosen:
        (out / (sha + '.py')).write_text(stripped_source(items[sha][1]))
    with (out / 'labels.csv').open('x', newline='') as f:
        w = csv.writer(f); names = [r['name'] for r in rules['rules']]
        w.writerow(['program_id', *names])
        w.writerows([[sha, *([''] * len(names))] for sha in chosen])
    (out / 'definitions.json').write_text(json.dumps({r['name']: r['description'] for r in rules['rules']}, indent=2, sort_keys=True)+'\n')
    record = {'schema':'rsi-at35-sample-v1', 'seed':seed, 'n':count, 'program_ids':chosen,
              'run_paths':sorted(str(Path(p).resolve()) for p in runs),
              'stripped_sha256':{sha:hashlib.sha256((out / (sha+'.py')).read_bytes()).hexdigest() for sha in chosen},
              'rules_sha256': Tagger(rules).sha256}
    (out/'sampling.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    return record


def read_labels(path, ids, names):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['program_id', *names]: raise ValueError('label columns differ from frozen rules')
        found = {}
        for row in reader:
            sha = row['program_id']
            if sha in found or sha not in ids or any(row[n] not in ('0','1') for n in names):
                raise ValueError('invalid or duplicate rater labels')
            found[sha] = {n:int(row[n]) for n in names}
    if set(found) != set(ids): raise ValueError('rater label sheet is incomplete')
    return found


def agreement(a, b, names, ids):
    rows, eligible = {}, []
    for name in names:
        left = [a[sha][name] for sha in ids]; right = [b[sha][name] for sha in ids]
        support = [sum(left), sum(right)]
        constant = len(set(left)) == 1 and len(set(right)) == 1
        kappa = None if constant else cohen_kappa(left, right)
        included = not constant and max(support) >= 5
        if included: eligible.append(kappa)
        rows[name] = {'rater_positive':support[0], 'other_positive':support[1], 'kappa':kappa,
                      'excluded_constant':constant, 'included_primary':included}
    return {'per_tag':rows, 'primary_kappa':sum(eligible)/len(eligible) if eligible else None,
            'excluded_constant':[n for n in names if rows[n]['excluded_constant']]}


def score(packet, labels, rules_path, runs, rater_id, rater2=None):
    if not isinstance(rater_id, str) or not rater_id.strip(): raise ValueError('rater-id is required')
    packet = Path(packet); record = json.loads((packet/'sampling.json').read_text()); rules = load_rules(rules_path); t = Tagger(rules)
    if record['rules_sha256'] != t.sha256: raise ValueError('rules differ from the blinded packet')
    if sorted(str(Path(p).resolve()) for p in runs) != record['run_paths']: raise ValueError('runs differ from sample')
    ids = record['program_ids']; names = [r['name'] for r in rules['rules']]
    human = read_labels(labels, ids, names); automatic = {}
    for sha in ids:
        body = (packet/(sha+'.py')).read_bytes()
        if hashlib.sha256(body).hexdigest() != record['stripped_sha256'][sha]: raise ValueError('packet source changed')
        tags = set(t.tags(body)); automatic[sha] = {n:int(n in tags) for n in names}
    compare = agreement(human, automatic, names, ids)
    comparisons = {'rater_vs_tagger':compare}
    if rater2: comparisons['rater_vs_rater2'] = agreement(human, read_labels(rater2, ids, names), names, ids)
    counts, vectors_by_run, all_rows, versions = Counter(), [], [], []
    for path in runs:
        run = Run(path); proposals = [(r, p['program_id']) for r, rows in sorted(run.proposals.items()) for p in rows]
        if not proposals: raise ValueError('run has no proposals')
        vectors, version = reference_vectors(run, sorted({sha for _,sha in proposals}), FEATURE_VERSION + '+recomputed'); versions.append(version)
        rows = []
        for r, sha in proposals:
            tags = t.tags(stripped_source(run.source(sha))); counts[t.family(tags)] += 1
            vec = vectors[sha]; rows.append((r, vec)); all_rows.append((vec, tags))
        vectors_by_run.append(rows)
    share = max(counts.values())/sum(counts.values()); kappa = compare['primary_kappa']
    mode = 'tags' if kappa is not None and kappa >= .7 and share <= .9 else 'clusters'
    result = {'schema':'rsi-at35-score-v1', 'rater_id':rater_id,
              'agreement_label':'model-rater kappa' if rater_id.startswith('model:') else 'rater kappa',
              'rules_sha256':t.sha256, 'comparisons':comparisons,
              'family_counts':dict(counts), 'largest_family_share':share, 'mode':mode}
    if mode == 'clusters':
        if len(set(versions)) != 1: raise ValueError('rehearsal feature versions differ')
        threshold, medians = calibrate_threshold(vectors_by_run)
        vectors = [v.tolist() for v,_ in all_rows]; cluster = cluster_labels(vectors, threshold)
        ref = {'schema':'rsi-cluster-reference-v1', 'threshold':threshold, 'feature_version':versions[0],
               'vectors':vectors, 'labels':cluster, 'tags':[tags for _,tags in all_rows], 'rules_sha256':t.sha256}
        path = packet.parent / (packet.name+'-cluster-reference.json')
        with path.open('x') as f: json.dump(ref, f, sort_keys=True, allow_nan=False); f.write('\n')
        result['cluster_reference'] = {'path':str(path.resolve()), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        result['threshold'] = threshold; result['threshold_medians'] = medians
    return result


def _program(code):
    jumps = set(dis.hasjabs) | set(dis.hasjrel)
    ins = [i for i in dis.get_instructions(code) if i.opname != 'NOP']; offsets = [i.offset for i in ins]; out=[]
    for item in ins:
        if item.opcode in jumps: arg = next(k for k, offset in enumerate(offsets) if offset >= item.argval)
        elif isinstance(item.argval, types.CodeType): arg = _program(item.argval)
        else: arg = item.argrepr
        out.append((item.opname, arg))
    return (code.co_name, tuple(out), tuple(c for c in code.co_consts if not isinstance(c,types.CodeType)),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)


def sweep(runs):
    items = sources(runs); failures = []
    if not items: raise ValueError('no candidate sources')
    for sha, (run, body) in sorted(items.items()):
        try:
            text = body.decode(); shown = stripped_source(text); tree = ast.parse(shown)
            if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(shown).readline)): raise ValueError('comment remains')
            if any(ast.get_docstring(n,clean=False) is not None for n in ast.walk(tree) if isinstance(n,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))): raise ValueError('docstring remains')
            if _program(compile(text,'solver.py','exec',optimize=2)) != _program(compile(shown,'solver.py','exec',optimize=2)): raise ValueError('bytecode differs')
        except Exception as exc: failures.append({'program_id':sha,'run':str(run),'error':str(exc)})
    return {'programs':len(items), 'failures':failures, 'status':'failed' if failures else 'passed'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('sample'); a.add_argument('--run', action='append', required=True); a.add_argument('--seed', type=int, required=True); a.add_argument('--n', type=int, default=60); a.add_argument('--out', required=True); a.add_argument('--rules')
    b = sub.add_parser('score'); b.add_argument('--packet', required=True); b.add_argument('--labels', required=True); b.add_argument('--rules', required=True); b.add_argument('--run', action='append', required=True); b.add_argument('--rater2'); b.add_argument('--rater-id', required=True)
    c = sub.add_parser('sweep'); c.add_argument('--run', action='append', required=True)
    args = parser.parse_args(argv)
    if args.command == 'sample': result = sample(args.run,args.seed,args.n,args.out,args.rules)
    elif args.command == 'score': result = score(args.packet,args.labels,args.rules,args.run,args.rater_id,args.rater2)
    else: result = sweep(args.run)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result.get('status') == 'failed' else 0

if __name__ == '__main__': raise SystemExit(main())
