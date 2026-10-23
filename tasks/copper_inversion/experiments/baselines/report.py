"""Read-only comparison; missing and failed measurements are not wins."""
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1]);rows=[]
for case in ('drillholes_3d','drillholes_outcrops_3d'):
    for method in ('paper_mcmc','MCMC','ACQ_MCMC','ast','qwen','fusion'):
        results=[json.loads(p.read_text()) for p in (root/case/method).glob('result-*.json')]
        if not results:continue
        good=[r for r in results if r['status']=='succeeded']
        values=[r['metrics']['loss_total'] for r in good]
        rows.append(dict(case=case,method=method,finished=len(results),successful=len(good),
            seeds=[r['seed'] for r in results],losses=values,
            mean_loss=statistics.mean(values) if values else None,
            std_loss=statistics.stdev(values) if len(values)>1 else None,
            solver_seconds=[r['solver_receipt']['wall_seconds'] for r in results],
            objective_calls=[r.get('search',{}).get('calls') for r in results],
            native_loss=[r.get('search',{}).get('best_raw_loss') for r in results]))
(root/'comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
