"""Replay original JSON bytes only when every field is semantically identical."""
import dataclasses,json,sys
from pathlib import Path
from tasks.copper_inversion.core import harness
from tasks.copper_inversion.experiments.v2 import parallel_runtime
original=harness.HarnessTurn
root=Path(sys.argv[sys.argv.index('--out-dir')+1])
def replay_turn(**kwargs):
    turn=original(**kwargs)
    paths=list((root/'harness').glob(f'**/sessions/{turn.profile_id}/turns/{turn.turn_id}/input.json'))
    if not paths:return turn
    if len(paths)!=1:raise ValueError('ambiguous persisted turn')
    saved=json.loads(paths[0].read_text());current=turn.to_dict()
    if set(saved)!=set(current):raise ValueError('turn field set changed')
    for key in saved:
        if key not in ('message','inputDigest') and saved[key]!=current[key]:raise ValueError('turn metadata changed: '+key)
    if json.loads(saved['message'])!=json.loads(turn.message):raise ValueError('turn semantic content changed')
    result=dataclasses.replace(turn,message=saved['message'])
    if result.input_digest!=saved['inputDigest']:raise ValueError('persisted input digest corrupt')
    print(json.dumps(dict(event='exact_input_replay',turn=turn.turn_id,serialization_only=turn.input_digest!=result.input_digest)),flush=True)
    return result
harness.HarnessTurn=replay_turn
parallel_runtime.main()
