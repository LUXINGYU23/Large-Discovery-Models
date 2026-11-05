"""Explicitly acknowledge immutable infrastructure-failure receipts; never retry or promote them."""
import hashlib,inspect,json
from pathlib import Path
from tasks.copper_inversion.core import execution

def install(acknowledged):
 original=execution.Execution.run
 signature=inspect.signature(original)
 def run(self,*args,**kwargs):
  values=signature.bind(self,*args,**kwargs);values.apply_defaults();v=values.arguments
  truth=v['truth'];td=hashlib.sha256((Path(truth)/'truth.npy').read_bytes()).hexdigest() if truth is not None else None
  key=execution.digest(dict(identity=v['identity'],science_module=v['science_module'],truth_digest=td,phase=v['phase'],image=self.image,timeout=v['timeout'],cpus=v['cpus'],memory_mb=v['memory_mb'],implementation=execution.source_identity()))
  receipt=self.root/key/'receipt.json';expected=acknowledged.get(str(receipt))
  if expected is None:return original(self,*args,**kwargs)
  raw=receipt.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=expected:raise execution.EvaluationPaused('acknowledged failure receipt changed')
  r=json.loads(raw)
  if r['reason']!='supervisor_timeout':raise execution.EvaluationPaused('acknowledgement must preserve infrastructure failure')
  # Returning the unmodified failure lets the existing suite emit failed status.
  # No Docker call, no new worker, and accounting remains keyed by original job.
  self._account(key,r,v['phase'])
  return receipt.parent,r
 execution.Execution.run=run
