# -*- coding: utf-8 -*-
"""Read-only source/metric export and annotated discovery trajectory; remote only."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

O = Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops')
B = O / 'seed42-investor-report-20261031'
R = Path('/Aidata/shared/ldm/copper-measurement-study-20261025-r3')
P = Path('/Aidata/shared/ldm/copper-v2-evaluator/runs')
REFERENCE = 'https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/analysis.ipynb'
# These labels describe verified active code, not candidate docstring nicknames.
NODES = {
    0: ('只返回观测初始化', '直接调用共享initialize，不进行迭代搜索。', '建立搜索前对照。'),
    1: ('加入球形贪心搜索', '在可行域随机选球心，0.075D尺度翻转体素；每次修复约束，仅接受更低公开损失；工作比例0.75。', '让观测初始体持续改善，而非直接返回。'),
    4: ('省去重复规范化', 'FAST=True，评分函数已做规范化，删去评分前重复canonicalize；单尺度仍为0.075D，工作比例0.68。', '减少一次提议中的重复距离场计算。'),
    5: ('从单尺度到三尺度', '新增SCALE_MIX=(0.05,0.075,0.105)，每次随机抽取球形翻转尺度。', '同时尝试小范围边界修改与较大形状修改。'),
    6: ('增加0.13大尺度', '半径集合扩展到0.13D；增加_add/_remove，按排名补足或裁剪体积。', '扩大空间搜索步幅，并更明确处理体积约束。'),
    7: ('扩大到五种尺度', '半径集合再加入0.17D，保持0.68工作比例与贪心接受。', '进一步覆盖更大范围的形状变化。'),
    8: ('加入低温退火与快速评分', 'mode=anneal，T0=1e-5；fast_legal/fast_loss启用；先评分合法提议，不合法才修复；偏移43。', '允许少量非改进移动，并减少合法提议的重复处理。'),
    9: ('调整退火温度与随机流', 'T0由1e-5改为3e-5，随机偏移43改为11；尺度与工作比例保持不变。', '调整接受较差提议的概率和随机搜索路径。'),
    10: ('退火后增加边界精修', '实际分支仍为sphere；工作比例0.68→0.94，前0.68退火，余下做边界半径1/2立方翻转。', '把额外工作时间投入边界局部改进；收益同时包含实际算时增加。'),
    13: ('重新分配退火与精修', '退火比例0.68→0.56，工作截止仍0.94；精修随机半径改为1/2/3。', '留出更多边界精修时间，并扩展局部邻域。'),
    14: ('精修保留小／大两尺度', '精修半径1/2/3→1/3，随机偏移11→31；仍随机选一个邻域评分。', '在最小和较大邻域间分配局部搜索。'),
    19: ('改为同点双提议比较', 'best_pair对同一边界点分别评分半径1和3，再选改进提议；其best模型记录问题已披露。', '用同一边界位置比较两种邻域；该源码整体获得训练最佳。'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def constants(s):
    out = {}
    for n in ast.parse(s).body:
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if isinstance(t, ast.Name) and (t.id.isupper()):
                try:
                    out[t.id] = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    pass
    return out


def function_map(s):
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}


protected = json.loads((O / 'r3-preservation-before.json').read_text())
assert len(protected) == 29 and all(sha(p) == h for p, h in protected.items())
report_sha = sha(B / 'data/report.json')
d = json.loads((B / 'data/report.json').read_text())
tr = json.loads((O / 'training-curves-source-draft-20261031.json').read_text())
g = next(g for g in tr['groups'] if g['method'] == 'ldm' and g['seed'] == 42)
rows = g['evaluations']
assert len(rows) == 308 and len({r['source_sha256'] for r in rows}) == 308
out = dict(discovery_seed=42, reference=dict(url=REFERENCE, commit='228791fb499afffb54b46200aca536f79142f117',
    adapted_elements=['faint candidate dots', 'prominent record points', 'running-best step', 'description annotations'],
    differences=['batch-end x coordinates; no invented within-batch ordering',
                 'all 308 scored candidates included; 13 charged interface rejections have no score',
                 'record improvement is not autoresearch KEEP or evidence of direct parentage']),
    candidates=[], nodes=[], baselines={}, report_data_sha256=report_sha)
for r in rows:
    h = r['source_sha256']; p = R / 'ldm-42/candidates' / h / 'solver.py'
    assert sha(p) == h
    out['candidates'].append(dict(round=r['round_idx'], batch_end_slot=1 + 16*r['round_idx'],
                                  source_sha256=h, metrics=r['metrics'], status=r['status']))
best = float('inf'); previous_source = None; previous_id = None
node_dir = B / 'source/trajectory'; node_dir.mkdir(exist_ok=True)
for rnd in range(21):
    r = min((r for r in rows if r['round_idx'] == rnd), key=lambda r: r['metrics']['loss_total'])
    loss = r['metrics']['loss_total']
    if loss >= best:
        continue
    h = r['source_sha256']; source = (R / 'ldm-42/candidates' / h / 'solver.py').read_text()
    path = node_dir / f'r{rnd:02d}-{h[:12]}.py'; path.write_text(source); assert sha(path) == h
    title, change, mechanism = NODES[rnd]
    fm = function_map(source)
    prev = function_map(previous_source) if previous_source is not None else {}
    diff_path = None
    if previous_source is not None:
        diff_path = node_dir / f'r{rnd:02d}-vs-previous-record.diff'
        diff_path.write_text('\n'.join(difflib.unified_diff(previous_source.splitlines(),
            source.splitlines(), fromfile=previous_id, tofile=h)) + '\n')
    n = dict(id=chr(65+len(out['nodes'])), round=rnd, batch_end_slot=1+16*rnd,
        loss=loss, improvement_absolute=best-loss if np.isfinite(best) else None,
        mean_solver_seconds=r['metrics']['mean_solver_wall_seconds'],
        source_sha256=h, source_path=str(path.relative_to(B)), title=title, code_change=change,
        mechanism=mechanism, constants=constants(source),
        changed_functions=[k for k in fm if fm[k] != prev.get(k)],
        removed_functions=[k for k in prev if k not in fm],
        diff_path=str(diff_path.relative_to(B)) if diff_path else None)
    out['nodes'].append(n)
    best, previous_source, previous_id = loss, source, h
assert [n['round'] for n in out['nodes']] == list(NODES)
assert abs(best - next(t['training_best']['metrics']['loss_total'] for t in d['training'] if t['method']=='ldm')) < 1e-12
selected_id = next(v['selected_sha256'] for v in d['validation'] if v['method']=='ldm')
selected_row = next(r for r in rows if r['source_sha256']==selected_id)
out['selected'] = dict(source_sha256=selected_id, round=selected_row['round_idx'],
    batch_end_slot=1+16*selected_row['round_idx'], metrics=selected_row['metrics'],
    note='validation-selected winner; distinct from training record L')
for method in ['mcmc', 'acq']:
    group = next(v for v in d['validation'] if v['method']==method)
    root = P / 'copper-measurement-study-20261027-r4-baselines/selection' / f'{method}-42'
    frozen = json.loads((root/'frozen.json').read_text()); h = group['selected_sha256']
    c = next(c for c in frozen['candidates'] if c['sha256']==h)
    src = root/c['source_path']; assert sha(src)==h
    dst = B/'source'/f'{method}42-selected-solver.py';dst.write_bytes(src.read_bytes())
    out['baselines'][method] = dict(source_sha256=h, source_path=str(dst.relative_to(B)),
        constants=constants(src.read_text()),
        functions={n.name:dict(start=n.lineno,end=n.end_lineno) for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef)})
f1=function_map((B/'source/mcmc42-selected-solver.py').read_text())
f2=function_map((B/'source/acq42-selected-solver.py').read_text())
assert f1==f2
out['baselines_function_bodies_identical']=True
out['training_summary']=dict(charged=321, scored=308, charged_interface_rejected=13,
    record_improvements=len(out['nodes'])-1, baseline_loss=out['nodes'][0]['loss'],
    final_training_loss=best, absolute_reduction=out['nodes'][0]['loss']-best,
    relative_reduction_percent=(1-best/out['nodes'][0]['loss'])*100)
out['protected_r3_files_verified']=29
assert all(sha(p)==h for p,h in protected.items()) and sha(B/'data/report.json')==report_sha
(B/'data/method-trajectory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))

# Same visual grammar as autoresearch; metrics and batch semantics remain copper-specific.
font_path=next((O/'report-fonts').rglob('NotoSansCJK-Regular.ttc'))
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font_path)).get_name(),
    'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'savefig.facecolor':'white','axes.titleweight':'bold'})
fig, ax=plt.subplots(figsize=(10,6.2))
xx=[r['batch_end_slot'] for r in out['candidates']]; yy=[r['metrics']['loss_total'] for r in out['candidates']]
ax.scatter(xx,yy,s=16,c='#b6c0c8',alpha=.43,label='完整候选（308份，纪录另突出）',zorder=1)
curve=next(t for t in d['training'] if t['method']=='ldm')
ax.step(curve['slots'],1-np.array(curve['best_utility']),where='post',c='#238c59',lw=2.5,label='LDM历史最佳',zorder=3)
for m,c in [('mcmc','#d38b26'),('acq','#9c4d9c')]:
    cur=next(t for t in d['training'] if t['method']==m)
    ax.step(cur['slots'],1-np.array(cur['best_utility']),where='post',c=c,lw=1.7,alpha=.85,label=m.upper()+'历史最佳',zorder=2)
nodes=out['nodes']
ax.scatter([n['batch_end_slot'] for n in nodes],[n['loss'] for n in nodes],
           s=54,c='#2ecc71',edgecolors='#175735',lw=.8,zorder=5,label='批末新纪录')
ax.scatter([out['selected']['batch_end_slot']],[out['selected']['metrics']['loss_total']],
           marker='*',s=210,c='#2563a6',edgecolors='white',zorder=6,label='验证选中程序（非训练最低点）')
for n in nodes:
    ax.annotate(n['id'],(n['batch_end_slot'],n['loss']),xytext=(0,-16),textcoords='offset points',
                ha='center',fontsize=9,color='#17623e',weight='bold')
labels={0:(7,.82,'A 初始化'),1:(22,.72,'B 加入球形搜索'),5:(69,.64,'D 多尺度翻转'),
        8:(128,.77,'G 低温退火＋快评'),10:(170,.68,'I 边界精修＋更长工作时间'),
        14:(230,.59,'K 精修尺度与随机流'),19:(270,.49,'L 同点双提议比较')}
for n in nodes:
    if n['round'] not in labels:continue
    x,y,label=labels[n['round']]
    ax.annotate(label,(n['batch_end_slot'],n['loss']),xytext=(x,y),fontsize=12,color='#17623e',
        bbox=dict(boxstyle='round,pad=.35',fc='white',ec='#bedbcc',alpha=.96),
        arrowprops=dict(arrowstyle='-',color='#6da489',lw=.8),zorder=7)
ax.set(xlim=(-4,330),ylim=(.315,max(.87,max(yy)+.03)),xlabel='累计收费候选槽位（仅按批末定位，非批内串行顺序）',
       ylabel='最佳／候选训练损失 ↓',title='LDM如何从初始化走向最终算法：321收费槽位，11次批末纪录改进')
ax.grid(alpha=.16);ax.legend(loc='upper right',fontsize=9.5,framealpha=.95)
fig.tight_layout();fig.savefig(B/'assets/ldm42-annotated-trajectory.png',dpi=190);fig.savefig(B/'assets/ldm42-annotated-trajectory.svg');plt.close(fig)
# Late-stage zoom keeps small, real changes visible without changing their magnitude.
fig,ax=plt.subplots(figsize=(10,3.5))
curves=d['training']
for m,c in [('ldm','#238c59'),('mcmc','#d38b26'),('acq','#9c4d9c')]:
    cur=next(t for t in curves if t['method']==m)
    ax.step(cur['slots'],1-np.array(cur['best_utility']),where='post',lw=2,c=c,label=m.upper())
for n in nodes[1:]:
    ax.scatter(n['batch_end_slot'],n['loss'],s=35,c='#2ecc71',edgecolor='#175735',zorder=4)
    ax.annotate(f"{n['id']} · R{n['round']}",(n['batch_end_slot'],n['loss']),xytext=(3,8),
                textcoords='offset points',fontsize=9.5,color='#17623e',rotation=18)
ax.set(xlim=(12,328),ylim=(.333,.432),xlabel='累计收费候选槽位（批末）',ylabel='历史最佳训练损失 ↓',
       title='首批搜索后的放大视图：尺度 → 退火 → 边界精修 → 双提议比较')
ax.grid(alpha=.18);ax.legend();fig.tight_layout()
fig.savefig(B/'assets/ldm42-trajectory-zoom.png',dpi=190);plt.close(fig)
# Keep the report's existing training/validation comparison, now cross-referenced
# to the four major source milestones instead of leaving an unannotated curve.
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for m,c in [('ldm','#2563a6'),('mcmc','#d38b26'),('acq','#9c4d9c')]:
    cur=next(t for t in d['training'] if t['method']==m)
    axs[0].step(cur['slots'],1-np.array(cur['best_utility']),where='post',color=c,label=m.upper())
    vg=next(g for g in d['validation'] if g['method']==m)
    j=['ldm','mcmc','acq'].index(m)
    for i,v in enumerate(vg['validation_candidates']):
        selected=v['source_sha256']==vg['selected_sha256']
        axs[1].scatter(j+(i-1)*.09,v['metrics']['loss_total'],color=c,
            s=180 if selected else 55,marker='*' if selected else 'o',edgecolor='white',zorder=3)
for n in nodes:
    if n['id'] not in ['B','G','I','L']:continue
    axs[0].annotate(n['id'],(n['batch_end_slot'],n['loss']),xytext=(-9 if n['id']=='L' else 5,12),
                    textcoords='offset points',color='#2563a6',fontsize=10,weight='bold')
axs[0].set(xlabel='累计收费槽位（批末）',ylabel='历史最佳训练损失 ↓',title='训练搜索：B / G / I / L为源码节点',xlim=(1,321))
axs[0].legend();axs[0].grid(alpha=.2)
axs[1].set(xticks=range(3),xticklabels=['LDM','MCMC','ACQ'],ylabel='验证损失 ↓',title='冻结验证：星标为选中程序')
axs[1].grid(axis='y',alpha=.2)
fig.savefig(B/'assets/training-validation.png',dpi=170);plt.close(fig)
print(json.dumps(dict(nodes=[{k:n[k] for k in ['id','round','loss','mean_solver_seconds','title']} for n in nodes],
    selected_round=out['selected']['round'],training=out['training_summary'],
    loss_range=[min(yy),max(yy)],formal_evaluations_rerun=0),ensure_ascii=False),flush=True)
