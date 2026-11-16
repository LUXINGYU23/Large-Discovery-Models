import {mkdirSync, cpSync} from 'node:fs';
import {join} from 'node:path';

export default function(pi) {
  pi.on('tool_call', (event) => {
    if (event.toolName !== 'bash') return;
    const timeout = event.input.timeout;
    if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 60) {
      return {block:true, reason:'Research shell commands require a timeout of at most 60 seconds.'};
    }
  });
  pi.registerTool({name:'get_inversion_context', label:'Read this branch task and evidence',
    description:'Export the public task, cases and this branch’s allowed measured history.',
    parameters:{type:'object',properties:{},additionalProperties:false},
    async execute(_id,_params,_signal,_update,ctx) {
      const dest = join(ctx.cwd,'.ldm-resources','copper');
      mkdirSync(dest,{recursive:true});
      cpSync('/artifacts/public',dest,{recursive:true});
      const value = {directory:'/workspace/.ldm-resources/copper',
        instructions:'Read TASK.md, cases.json, evidence/history.txt and evidence/sources/.'};
      return {content:[{type:'text',text:JSON.stringify(value)}],details:value};
    }});
}
