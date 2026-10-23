import {readFileSync,mkdirSync,cpSync} from 'node:fs';
import {join} from 'node:path';
export default function(pi) {
  pi.on('tool_call', (event) => {
    if (event.toolName !== 'bash') return;
    const timeout = event.input.timeout;
    if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 60) {
      return {block:true, reason:'Research bash commands require an explicit timeout positive and at most 60 seconds. Use read/write tools for file operations. Keep a complete candidate and submit before the turn deadline; formal solver evaluation retains its separate task budget.'};
    }
  });
  pi.registerTool({name:'get_inversion_context',label:'Read public inversion case and measured history',
    description:'Export public data, fixed read-only selfcheck, objective source, and authoritative evaluated algorithms into guest files. Read these files before designing solver.py. Includes all measured source code and losses.',
    parameters:{type:'object',properties:{},additionalProperties:false},
    async execute(_id,_params,_signal,_update,ctx) {
      const dest=join(ctx.cwd,'.ldm-resources','copper');mkdirSync(dest,{recursive:true});
      cpSync('/artifacts/public',dest,{recursive:true});
      const history=JSON.parse(readFileSync('/artifacts/public/history.json','utf8'));
      const value={directory:'/workspace/.ldm-resources/copper',history_count:history.length,
        instructions:'Read TASK.md, history.json and objective.py. Load arrays from problem.npz and metadata from problem.json. Run python selfcheck.py /workspace/candidates/0/solver.py from the exported directory. Frozen tests are public; controller independently checks the final snapshot.'};
      return {content:[{type:'text',text:JSON.stringify(value)}],details:value};
    }});
}
