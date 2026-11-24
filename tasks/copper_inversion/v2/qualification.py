"""Fail-closed private release checks; a stage label alone is never qualification."""
import json
import os
from pathlib import Path
from .contract import validate
from .suite import file_hash
from .data import read_problem


def verify_geological_constraints(manifest, *, require_current_objective=False):
    """Formal data must enable a calibrated geometric prior, not just air masks."""
    manifest=Path(manifest).resolve()
    cases=json.loads(manifest.read_text()).get('cases',[])
    if not cases:raise ValueError('geological constraints require nonempty cases')
    for case in cases:
        public=(manifest.parent/case['public']).resolve()
        if not public.is_relative_to(manifest.parent):raise ValueError('public path escapes release')
        from .geological_policy import public_priors
        problem=read_problem(public)
        if require_current_objective:
            validate(problem, legacy=False)
        priors=problem['priors']
        enabled=(priors['hard_volume_bounds'] is not None or priors['max_components'] is not None
                 or priors['area_weight']>0 or priors['volume_weight']>0
                 or (priors['fragment_weight']>0 and priors['fragment_volume']>0))
        if not enabled:raise ValueError('formal case has no enabled geological prior: '+case['case_id'])
        if priors!=public_priors(problem['domain_mask']&~problem['forbidden_mask'],problem['spacing']):
            raise ValueError('formal priors differ from the fixed public population policy')


TIER_KEYS={"solver_seconds","train_manifest","train_bodies","validation_manifest","sealed_manifests","slots"}
EVIDENCE_V1=('calibration','population','historical_disjointness','confirmation','isolation','geological_constraints')
# Release schema v2 (rsi-r1) adds the tier table, one-layout assignment, the
# weak-judge thinning and the whole-hole audit; v3 (rsi-r2) adds judge_validity.
EVIDENCE_V2=EVIDENCE_V1+('hole_audit','layout','weak_judge')
EVIDENCE_V3=EVIDENCE_V2+('judge_validity',)


def verify_tiers(record,release=None,*,allow_unsealed=False):
    """The `tiers` object read by the evaluator service: contracts §2.1 shape exactly."""
    tiers=record.get('tiers')
    if not isinstance(tiers,dict) or set(tiers) not in ({'S','F'},{'S','M','F'}):
        raise ValueError('release tier table requires tiers S and F')
    covered=record.get('manifest_sha256',{})
    for name,tier in tiers.items():
        if not isinstance(tier,dict) or set(tier)!=TIER_KEYS:raise ValueError('invalid tier fields: '+name)
        for key in ('solver_seconds','train_bodies','slots'):
            if type(tier[key]) is not int or tier[key]<1:raise ValueError(f'invalid tier {key}: {name}')
        manifests=[tier['train_manifest'],tier['validation_manifest'],*tier['sealed_manifests']]
        if (not tier['sealed_manifests'] and not allow_unsealed) or any(not isinstance(m,str) or m not in covered for m in manifests):
            raise ValueError('tier manifest not covered by qualification: '+name)
        if release is not None:
            cases=json.loads((Path(release)/tier['train_manifest']).read_text())['cases']
            if len({row['parent_id'] for row in cases})!=tier['train_bodies']:
                raise ValueError('tier training body count differs from its manifest: '+name)
    return tiers


def load_tiers(release):
    """Validated tier table of a release root (formal `qualification.json`)."""
    release=Path(release).resolve()
    return verify_tiers(json.loads((release/'qualification.json').read_text()),release)


def verify_release(manifest):
    manifest=Path(manifest).resolve()
    release=manifest.parent.parent
    receipt=release/'qualification.json'
    record=json.loads(receipt.read_text())
    schemas={'copper-qualified-synthetic-v1','copper-qualified-synthetic-v2','copper-qualified-synthetic-v3'}
    if record.get('schema') not in schemas or record.get('status')!='qualified':
        raise ValueError('missing qualified synthetic release')
    v2=record['schema'] in {'copper-qualified-synthetic-v2','copper-qualified-synthetic-v3'}
    v3=record['schema']=='copper-qualified-synthetic-v3'
    if os.geteuid()!=record['evaluator_uid'] or receipt.stat().st_uid!=record['evaluator_uid']:
        raise ValueError('formal release requires its independent evaluator account')
    if file_hash(manifest)!=record['manifest_sha256'].get(str(manifest.relative_to(release))):
        raise ValueError('formal manifest not covered by qualification')
    for relative,expected in record['evidence_sha256'].items():
        path=(release/relative).resolve()
        if not path.is_relative_to(release) or file_hash(path)!=expected:
            raise ValueError('qualification evidence changed')
    required=list(EVIDENCE_V3 if v3 else (EVIDENCE_V2 if v2 else EVIDENCE_V1))
    if not v3 and any(k in record['evidence_sha256'] for k in ('evidence/judge_validity.json','evidence/judge-validity.json')):
        required.append('judge_validity')
    for name in required:
        candidates=('evidence/judge_validity.json','evidence/judge-validity.json') if name=='judge_validity' else ('evidence/'+name+'.json',)
        rel=next((c for c in candidates if c in record['evidence_sha256']),None)
        if rel is None:
            raise ValueError('qualification does not seal evidence: '+name)
        evidence=json.loads((release/rel).read_text())
        if evidence.get('status')!='passed':raise ValueError('incomplete qualification: '+name)
        if name=='geological_constraints' and (
                evidence.get('calibration_scope')!='independent_development'
                or evidence.get('used_evaluation_truth') is not False):
            raise ValueError('geological priors require independent calibration without evaluation truth')
        if name=='geological_constraints':
            from .geological_policy import POLICY_VERSION
            if evidence.get('policy')!=POLICY_VERSION:raise ValueError('geological calibration policy differs')
    if v2:
        verify_tiers(record,release)
        for judge in record['judges'].values():
            for key in ('train_manifest','validation_manifest'):
                if key in judge and judge[key] not in record['manifest_sha256']:
                    raise ValueError('judge manifest not covered by qualification')
    verify_geological_constraints(manifest,require_current_objective=v3)
    return record
