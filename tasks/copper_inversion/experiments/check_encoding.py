"""Remote real-model encoding smoke, before any provider-backed campaign."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from tasks.copper_inversion.core.encoding import CodeEncoder, MODEL_REVISION
from tasks.copper_inversion.core.candidate import Domain, seal
from ldm_tts.contracts import RawProposal

base=Path(sys.argv[1]);model=Path(sys.argv[2]);results={};vectors={}
for mode in ('ast','qwen','fusion'):
    root=base/mode
    body=b'def solve(problem, budget, seed):\n    return problem["initial_level_set"].copy()\n'
    candidate=Domain(root).admit(RawProposal(seal(root,body),'smoke'))
    encoder=CodeEncoder(root,mode,model,'cuda:0')
    start=time.monotonic();v=encoder.encode(candidate);elapsed=time.monotonic()-start
    x=np.array(v.values);assert np.isfinite(x).all() and np.isclose(np.linalg.norm(x),1.)
    assert encoder.encode(candidate)==v
    vectors[mode]=x
    results[mode]={'dimension':len(x),'norm':float(np.linalg.norm(x)),'seconds':elapsed,'version':v.version}
    if mode=='fusion':
        assert np.allclose(x[:256],vectors['ast']/np.sqrt(2))
        assert np.allclose(x[256:],vectors['qwen']/np.sqrt(2),atol=1e-6)
        try: encoder.model.encode('token '*40000)
        except ValueError as e: assert 'no truncation' in str(e)
        else: raise AssertionError('long input was silently truncated')
    del encoder
    import gc
    gc.collect()
    if mode!='ast':
        import torch
        torch.cuda.empty_cache()
(base/'verification.json').write_text(json.dumps({'status':'passed','model_revision':MODEL_REVISION,'results':results},indent=2))
print(json.dumps(results))
