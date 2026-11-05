"""Fail-closed private release checks; a stage label alone is never qualification."""
import json
import os
from pathlib import Path
from .suite import file_hash
from .data import read_problem


def verify_geological_constraints(manifest):
    """Formal data must enable a calibrated geometric prior, not just air masks."""
    manifest=Path(manifest).resolve()
    cases=json.loads(manifest.read_text()).get('cases',[])
    if not cases:raise ValueError('geological constraints require nonempty cases')
    for case in cases:
        public=(manifest.parent/case['public']).resolve()
        if not public.is_relative_to(manifest.parent):raise ValueError('public path escapes release')
        from .geological_policy import public_priors
        problem=read_problem(public);priors=problem['priors']
        enabled=(priors['hard_volume_bounds'] is not None or priors['max_components'] is not None
                 or priors['area_weight']>0 or priors['volume_weight']>0
                 or (priors['fragment_weight']>0 and priors['fragment_volume']>0))
        if not enabled:raise ValueError('formal case has no enabled geological prior: '+case['case_id'])
        if priors!=public_priors(problem['domain_mask']&~problem['forbidden_mask'],problem['spacing']):
            raise ValueError('formal priors differ from the fixed public population policy')


def verify_release(manifest):
    manifest=Path(manifest).resolve()
    release=manifest.parent.parent
    receipt=release/'qualification.json'
    record=json.loads(receipt.read_text())
    if record.get('schema')!='copper-qualified-synthetic-v1' or record.get('status')!='qualified':
        raise ValueError('missing qualified synthetic release')
    if os.geteuid()!=record['evaluator_uid'] or receipt.stat().st_uid!=record['evaluator_uid']:
        raise ValueError('formal release requires its independent evaluator account')
    if file_hash(manifest)!=record['manifest_sha256'].get(str(manifest.relative_to(release))):
        raise ValueError('formal manifest not covered by qualification')
    for relative,expected in record['evidence_sha256'].items():
        path=(release/relative).resolve()
        if not path.is_relative_to(release) or file_hash(path)!=expected:
            raise ValueError('qualification evidence changed')
    for name in ('calibration','population','historical_disjointness','confirmation','isolation','geological_constraints'):
        if 'evidence/'+name+'.json' not in record['evidence_sha256']:
            raise ValueError('qualification does not seal evidence: '+name)
        evidence=json.loads((release/'evidence'/f'{name}.json').read_text())
        if evidence.get('status')!='passed':raise ValueError('incomplete qualification: '+name)
        if name=='geological_constraints' and (
                evidence.get('calibration_scope')!='independent_development'
                or evidence.get('used_evaluation_truth') is not False):
            raise ValueError('geological priors require independent calibration without evaluation truth')
        if name=='geological_constraints':
            from .geological_policy import POLICY_VERSION
            if evidence.get('policy')!=POLICY_VERSION:raise ValueError('geological calibration policy differs')
    verify_geological_constraints(manifest)
    return record
