"""Compare final frozen solvers with completed, resource-matched baseline receipts."""
import json
from pathlib import Path
import statistics
import sys

base = Path(sys.argv[1])
final = base / 'matched-120s-final-search'
reference = json.loads((final / 'manifest.json').read_text())
rows = []
for folder, methods in (
    ('matched-120s-v2', ('paper_mcmc', 'MCMC', 'ACQ_MCMC')),
    ('matched-120s-acq-warmup4', ('ACQ_MCMC',)),
    ('matched-120s-final-search', ('ast', 'qwen', 'fusion')),
):
    root = base / folder
    manifest = json.loads((root / 'manifest.json').read_text())
    for key in ('seconds', 'seeds', 'solver_image', 'scorer_image', 'cpus', 'memory_mb', 'objective'):
        assert manifest[key] == reference[key], (folder, key)
    for case in ('drillholes_3d', 'drillholes_outcrops_3d'):
        for method in methods:
            results = [json.loads((root / case / method / f'result-{seed}.json').read_text())
                       for seed in reference['seeds']]
            successful = [r for r in results if r['status'] == 'succeeded']
            losses = [r['metrics']['loss_total'] for r in successful]
            times = [r['solver_receipt']['wall_seconds'] for r in results]
            origin = next(s for s in manifest['sources'] if s['case'] == case and s['method'] == method)
            label = method if method != 'ACQ_MCMC' else ('ACQ_warmup4' if 'warmup4' in folder else 'ACQ_warmup30')
            rows.append(dict(case=case, method=label, root=str(root),
                             source_sha256=origin['payload']['sha256'], origin=origin['origin'],
                             seeds=reference['seeds'], successful=len(successful), attempts=len(results),
                             losses=losses, mean_loss=statistics.mean(losses) if losses else None,
                             std_loss=statistics.stdev(losses) if len(losses)>1 else None,
                             wall_seconds=times, mean_seconds=statistics.mean(times),
                             std_seconds=statistics.stdev(times)))
(final / 'baseline-comparison.json').write_text(json.dumps(rows, indent=2))
lines = ['# 最终冻结求解器：120 秒同资源复验', '',
         '每行 seeds=47/48/49；损失和时间均为均值 ± 样本标准差。时间为 solver 墙钟秒，不含设计、接口检查和独立评分。', '',
         '传统基线复用已完成的同镜像、同资源、同种子测量；本次新运行最终 LDM 的 18 次复验。', '',
         '| Case | 方法 | 搜索轮数 | 成功/尝试 | 损失 | 求解秒 |',
         '|---|---|---:|---:|---:|---:|']
for r in rows:
    loss = f"{r['mean_loss']:.6f} ± {r['std_loss']:.6f}" if r['std_loss'] is not None else str(r['mean_loss'])
    lines.append(f"| {r['case']} | {r['method']} | {r['origin'].get('selection_round', '—')} | {r['successful']}/{r['attempts']} | {loss} | {r['mean_seconds']:.2f} ± {r['std_seconds']:.2f} |")
(final / 'baseline-comparison.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows, indent=2))
