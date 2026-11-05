"""Reconstruct a lost sidecar submission journal from a matching durable admission.
No provider calls, candidate generation, budget edits, history edits or turn commits.
Default is read-only; --apply writes the missing journal with exclusive creation.
"""
import argparse, hashlib, json, os
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
def reconstruct(branch, profile, turn):
    root = branch/'sessions'/profile/'turns'/turn
    inp=json.loads((root/'input.json').read_text())
    assert inp['profileId']==profile and inp['turnId']==turn
    admfile=branch/'admissions'/(inp['inputDigest']+'.json')
    adm=json.loads(admfile.read_text())
    assert adm['profile_id']==profile and adm['input_digest']==inp['inputDigest']
    calls=[]
    for f in (branch/'sessions'/profile/'pi-session').glob('*.jsonl'):
        for line in f.read_text().splitlines():
            d=json.loads(line); m=d.get('message',{})
            for c in m.get('content',[]) if isinstance(m.get('content'),list) else []:
                if c.get('type')=='toolCall' and c.get('name')=='submit_solver': calls.append(c['arguments'])
    # Deliberately narrow recovery: exactly one original attempt, eight artifacts.
    assert len(calls)==1 and len(calls[0]['candidates'])==8
    submission=calls[0]; artifacts=[]
    assert sorted(p.name for p in (root/'attempts').iterdir())==['0001']
    for i,c in enumerate(submission['candidates']):
        f=root/'attempts/0001/artifacts'/f'artifact-{i:02}.py'
        assert not f.is_symlink() and f.is_file()
        b=f.read_bytes()
        artifacts.append(dict(pathPointer=f'/candidates/{i}/artifact_path',relativePath=c['artifact_path'],snapshotPath=str(f.relative_to(branch)),sha256=sha(b),sizeBytes=len(b)))
    assert [x['sha256'] for x in artifacts]==adm['sha256s']
    text=canonical(dict(artifacts=artifacts,submission=submission))
    assert sha(text.encode())==adm['submission_digest'], 'submission metadata or artifact mismatch'
    budget=json.loads((root/'tool-budget.json').read_text())
    used,limits=budget['used'],budget['limits']
    journal=dict(submission=dict(submissionStatus='accepted',submissionId=turn+'-submission-1',submissionJson=text,submissionDigest=sha(text.encode()),submission=submission,submittedArtifacts=artifacts,validationErrors=[]),toolUsage=dict(toolCalls=used,toolBudget={k:dict(limit=v,used=used.get(k,0),remaining=max(0,v-used.get(k,0))) for k,v in limits.items()}))
    return root,journal,admfile

def main():
    p=argparse.ArgumentParser();p.add_argument('branch',type=Path);p.add_argument('profile');p.add_argument('turn');p.add_argument('--apply',action='store_true');a=p.parse_args()
    root,journal,adm=reconstruct(a.branch,a.profile,a.turn)
    target=root/'submission.json'
    if target.exists():
        assert json.loads(target.read_text())==journal
        print(json.dumps(dict(status='matches_existing',profile=a.profile)));return
    proof=dict(operation='restore_accepted_submission_journal',admission_sha256=sha(adm.read_bytes()),input_sha256=sha((root/'input.json').read_bytes()),submission_digest=journal['submission']['submissionDigest'],tool_budget_sha256=sha((root/'tool-budget.json').read_bytes()),history_preserved=True,new_provider_calls=0,new_evaluations=0)
    if a.apply:
        proof_path=root/'accepted-submission-recovery.json'
        with proof_path.open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
        # Atomic and no overwrite; original framework will verify and commit.
        temp=root/'submission.recovery.tmp'
        with temp.open('x') as f:json.dump(journal,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        os.link(temp,target);temp.unlink()
    print(json.dumps(dict(status='restored' if a.apply else 'verified_dry_run',**proof)))
if __name__=='__main__':main()
