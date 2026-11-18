import json, os, sys, subprocess, signal, time, socket, secrets, urllib.request
from pathlib import Path
root=Path('/Aidata/shared/ldm/antibody-poc')
os.chdir(root/'code')
bridge='172.17.0.1'
with socket.socket() as s:
    s.bind((bridge,0))
    port=s.getsockname()[1]
url=f'http://{bridge}:{port}/v1'
key=secrets.token_hex(32)
method=sys.argv[1] if len(sys.argv)>1 else 'ldm_harness'
dest=root/'runs/antibody_rfd3'/('qwen3-'+method)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VLLM_API_KEY=key,LLM_API_KEY=key,LLM_BASE_URL=url,LLM_MODEL_NAME='Qwen3-8B',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_PROXY=bridge+',localhost,127.0.0.1',no_proxy=bridge+',localhost,127.0.0.1',LDM_GONDOLIN_CACHE=str(root/'cache/gondolin'),LDM_PI_IMAGE='ldm-copper-pi:main-3580273b',VLLM_NO_USAGE_STATS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_CACHE_ROOT=str(root/'cache/vllm'))
command=['taskset','-c','0,1','/home/zsgpu/vllm-env/bin/vllm','serve','/Aidata/shared/rsi-models/Qwen3-8B','--host',bridge,'--port',str(port),'--served-model-name','Qwen3-8B','--max-model-len','16384','--max-num-seqs','1','--max-num-batched-tokens','1024','--gpu-memory-utilization','0.27','--enforce-eager','--enable-auto-tool-choice','--tool-call-parser','hermes','--reasoning-parser','qwen3','--generation-config','vllm']
with (root/'logs/qwen-campaign-server.log').open('w') as log:
    server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(150):
            if server.poll() is not None: raise RuntimeError('vLLM exited during startup')
            try:
                with opener.open(urllib.request.Request(url+'/models',headers={'Authorization':'Bearer '+key}),timeout=3) as r:
                    print('Live model:',json.load(r)['data'][0]['id'],flush=True)
                break
            except OSError: time.sleep(2)
        else: raise RuntimeError('Model startup timed out')
        cmd=['taskset','-c','0,1',str(root/'env/antibody_campaign/bin/python'),'scripts/run_ldm_tts.py','config/antibody_rfd3/'+method+'.yaml','--set','args.share-gpu=true','--set','args.out-dir='+str(dest)]
        if len(sys.argv)>2: cmd+=['--set','args.initialization-run='+str(root/'runs/antibody_rfd3'/sys.argv[2])]
        from qwen_responses_bridge import start
        listener=start(bridge,key,url)
        env['LLM_BASE_URL']=f'http://{bridge}:{listener.server_port}/v1'
        try:
            result=subprocess.run(cmd,env=env,timeout=1800)
        finally:
            listener.shutdown()
            listener.server_close()
        sys.exit(result.returncode)
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try: server.wait(timeout=25)
            except subprocess.TimeoutExpired: os.killpg(server.pid,signal.SIGKILL);server.wait()

