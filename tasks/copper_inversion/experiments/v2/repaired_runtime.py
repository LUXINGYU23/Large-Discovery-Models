"""Versioned r3 operational continuation. Old science and receipts stay immutable.

The epoch pins host admission/transport modules, the new sidecar for NEW sessions,
and each baseline's exact source for every parameter index. Source changes never
rewrite a paid candidate. All budgets continue from their existing ledger.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

def load_module(name,path,expected):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('repair component changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=Path,required=True);a,rest=p.parse_known_args()
    epoch=json.loads(a.epoch.read_text())
    from tasks.copper_inversion.core.execution import source_identity
    if source_identity()!=epoch['scientific_identity']:raise ValueError('scientific runtime changed')
    from tasks.copper_inversion.experiments.v2 import parallel_runtime
    from tasks.copper_inversion.v2 import interface_checks,service
    from ldm_tts.harness import client
    modules={}
    for key,name in [('interface','tasks.copper_inversion.v2.concurrent_interfaces'),('client','ldm_tts.harness.validation_clock')]:
        component=epoch[key];modules[key]=load_module(name,component['path'],component['sha256'])
    interface_checks.InterfaceChecks=modules['interface'].InterfaceChecks
    service.InterfaceChecks=modules['interface'].InterfaceChecks
    client.HarnessClient._request=modules['client'].HarnessClient._request
    from tasks.copper_inversion.core import harness_runtime,harness
    from ldm_tts.engine.run_store import atomic_json_write
    command=harness_runtime.pi_sidecar_command
    def versioned_sidecar(root,image,**kwargs):
        root=Path(root);record=root/'runtime-epoch.json'
        if record.exists():
            choice=json.loads(record.read_text())
            if choice['epoch']!=epoch['id']:raise ValueError('sidecar epoch changed')
            image=choice['image']
        elif not (root/'manifest.json').exists():
            image=epoch['sidecar_image'];atomic_json_write(record,dict(epoch=epoch['id'],image=image))
        # Existing native session manifests keep their original image and history.
        return command(root,image,**kwargs)
    harness_runtime.pi_sidecar_command=versioned_sidecar
    if 'ldm' in rest:
        import dataclasses
        turn_class=harness.HarnessTurn;root=Path(rest[rest.index('--out-dir')+1])
        def exact_turn(**kwargs):
            turn=turn_class(**kwargs);paths=list((root/'harness').glob(f'**/sessions/{turn.profile_id}/turns/{turn.turn_id}/input.json'))
            if not paths:return turn
            if len(paths)!=1:raise ValueError('ambiguous persisted input')
            saved=json.loads(paths[0].read_text());current=turn.to_dict()
            if set(saved)!=set(current):raise ValueError('input field set changed')
            if any(saved[k]!=current[k] for k in saved if k not in ('message','inputDigest')):raise ValueError('input metadata changed')
            if json.loads(saved['message'])!=json.loads(turn.message):raise ValueError('input semantics changed')
            replay=dataclasses.replace(turn,message=saved['message'])
            if replay.input_digest!=saved['inputDigest']:raise ValueError('input digest corrupt')
            return replay
        harness.HarnessTurn=exact_turn
    if 'tune' in rest:
        from tasks.copper_inversion.experiments.v2 import tune_stochastic
        run=Path(rest[rest.index('--run')+1]);mapping_path=Path(epoch['baseline_maps'][run.name]['path'])
        if hashlib.sha256(mapping_path.read_bytes()).hexdigest()!=epoch['baseline_maps'][run.name]['sha256']:raise ValueError('source map changed')
        mapping=json.loads(mapping_path.read_text());by_config={json.dumps(row['parameters'],sort_keys=True):row for row in mapping}
        def source_for(template,parameters):
            row=by_config[json.dumps(parameters,sort_keys=True)];body=Path(row['source_file']).read_bytes()
            if hashlib.sha256(body).hexdigest()!=row['source_sha256']:raise ValueError('mapped source changed')
            return body.decode()
        tune_stochastic.source_for=source_for
    sys.argv=[sys.argv[0],*rest];parallel_runtime.main()

if __name__=='__main__':main()
