"""Operational input identity journal, persisted before existing turn accounting."""
import dataclasses,json,re
from pathlib import Path
from ldm_tts.engine.run_store import atomic_json_write

def durable_turn(turn, root):
    root=Path(root)
    if not re.fullmatch(r'researcher_[1-4]_round_[0-9]+',turn.turn_id):
        raise ValueError('unexpected researcher turn identity')
    path=root/'operational-turn-inputs'/f'{turn.turn_id}.json'
    native=list((root/'harness').glob(f'**/sessions/{turn.profile_id}/turns/{turn.turn_id}/input.json'))
    if len(native)>1:raise ValueError('ambiguous native turn')
    records=[json.loads(p.read_text()) for p in [path,*native] if p.exists()]
    current=turn.to_dict()
    for saved in records:
        if set(saved)!=set(current):raise ValueError('input field set changed')
        if any(saved[k]!=current[k] for k in saved if k not in ('message','inputDigest')):
            raise ValueError('input metadata changed')
        if json.loads(saved['message'])!=json.loads(current['message']):
            raise ValueError('input semantics changed')
        candidate=dataclasses.replace(turn,message=saved['message'])
        if candidate.input_digest!=saved['inputDigest']:raise ValueError('input digest corrupt')
    if len(records)==2 and records[0]!=records[1]:raise ValueError('native and journal differ')
    if records:turn=dataclasses.replace(turn,message=records[0]['message'])
    if not path.exists():atomic_json_write(path,turn.to_dict())
    return turn

def install(root):
    from tasks.copper_inversion.core import harness
    original=harness.HarnessTurn
    def create(**kwargs):return durable_turn(original(**kwargs),root)
    harness.HarnessTurn=create
