"""Resume exact persisted turn bytes only after full semantic identity checks."""
import dataclasses
import json
from pathlib import Path
import sys
from tasks.copper_inversion.core import harness
import parallel_runtime

original=harness.HarnessTurn
root=Path(sys.argv[sys.argv.index('--out-dir')+1])
def replay_turn(**kwargs):
    turn=original(**kwargs)
    paths=list((root/'harness').glob(f'**/sessions/{turn.profile_id}/turns/{turn.turn_id}/input.json'))
    if not paths:return turn
    if len(paths)!=1:raise ValueError('ambiguous persisted turn')
    saved=json.loads(paths[0].read_text());current=turn.to_dict()
    for k in saved:
        if k not in ('message','inputDigest') and saved[k]!=current[k]:raise ValueError('persisted turn metadata changed: '+k)
    old_message=json.loads(saved['message']);new_message=json.loads(turn.message)
    if old_message!=new_message:
        keys=[k for k in set(old_message)|set(new_message) if old_message.get(k)!=new_message.get(k)]
        raise ValueError('persisted turn semantic content changed: '+str(keys))
    result=dataclasses.replace(turn,message=saved['message'])
    if result.input_digest!=saved['inputDigest']:raise ValueError('persisted turn digest corrupt')
    print(json.dumps(dict(event='exact_turn_replay',turn=turn.turn_id,serialization_only=turn.input_digest!=result.input_digest)),flush=True)
    return result
harness.HarnessTurn=replay_turn
parallel_runtime.main()
