"""Render a training-only, charged-prefix snapshot. Execute remotely."""
import argparse,json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
d=json.loads(a.snapshot.read_text());a.output.mkdir(parents=True,exist_ok=True)
methods=['ldm','mcmc','acq'];seeds=[42,43,3407];colors=['#1768ac','#d76b27','#319477'];n=d['charged_prefix']
stamp=datetime.fromtimestamp(d['time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True);axes=axes.ravel()
for i,(m,c) in enumerate(zip(methods,colors)):
 for ax,field in [(axes[0],'curves'),(axes[1],'solver_cost_curves')]:
  v=np.array([d[field][f'{m}-{s}'] for s in seeds]);mean=v.mean(0);sd=v.std(0,ddof=1);x=np.arange(1,n+1)
  ax.plot(x,mean,color=c,label=m.upper());ax.fill_between(x,mean-sd,mean+sd,color=c,alpha=.13)
 for j,s in enumerate(seeds):
  g=d['groups'][f'{m}-{s}'];xx=i+(j-1)*.13
  for ax,key in [(axes[2],'best_loss'),(axes[3],'best_mean_solver_seconds')]:
   ax.scatter(xx,g[key],color=c,marker=['o','s','^'][j],s=55);ax.annotate(str(s),(xx,g[key]),xytext=(5,5),textcoords='offset points',fontsize=8)
axes[0].set(title='Best training measurement loss: mean ± sample SD',xlabel='Charged source budget (rejections included)',ylabel='L / (1 + L), lower is better');axes[0].legend(frameon=False)
axes[1].set(title='Sum of completed solver-job times: mean ± sample SD',xlabel='Charged source budget',ylabel='Aggregate solver hours per discovery seed');axes[1].legend(frameon=False)
axes[2].set(title=f'Three discovery seeds at budget {n}',ylabel='Best training measurement loss');axes[3].set(title='Winner solve time at the same budget',ylabel='Mean solver seconds per training case')
for ax in axes[2:]:ax.set_xticks(range(3),[m.upper() for m in methods])
for ax in axes:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle(f'Copper inversion r3 — INTERIM TRAINING RESULTS — {stamp} CST',fontsize=14)
for ext in ['png','pdf']:fig.savefig(a.output/f'training-prefix-{n}.{ext}',dpi=180)
s=d['method_summary'];worse=s['ldm']['mean']>min(s['mcmc']['mean'],s['acq']['mean'])
conclusion=('当前 LDM 平均损失仍高于基线，三个种子的波动也明显更大，尚无稳定优势。' if worse else '当前前缀的训练均值有改善，但仅三个种子及未完成独立测试不足以确认稳定泛化优势。')
rows=''.join(f'<tr><td>{m.upper()}</td><td>{s[m]["mean"]:.6f} ± {s[m]["sample_sd"]:.6f}</td><td>{s[m]["mean_solver_seconds"]:.2f} ± {s[m]["solver_sample_sd"]:.2f}</td><td>{s[m]["mean_cumulative_completed_solver_hours"]:.2f}</td></tr>' for m in methods)
seedrows=''.join(f'<tr><td>{m.upper()}</td><td>{seed}</td><td>{d["groups"][f"{m}-{seed}"]["best_loss"]:.6f}</td><td>{d["groups"][f"{m}-{seed}"]["best_mean_solver_seconds"]:.2f}</td></tr>' for m in methods for seed in seeds)
progress=''.join(f'<tr><td>{name.upper()}</td><td>{h["charged"]}/321</td><td>{h["complete_prefix"]}</td><td>{h["complete_results"]}</td><td>{h["rejections"]}</td><td>{h["new_job_receipts"]}</td></tr>' for name,h in d['health'].items())
failures=sum(len(h['new_failed_jobs']) for h in d['health'].values());jobs=sum(h['new_job_receipts'] for h in d['health'].values());newresponses=sum(h['new_provider_responses'] for h in d['health'].values())
html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>铜矿反演 r3：修复后阶段对比</title><style>body{{max-width:1200px;margin:35px auto;padding:0 24px;font:16px/1.7 system-ui;color:#192b3b}}img{{width:100%}}table{{border-collapse:collapse;width:100%;margin:15px 0}}td,th{{padding:8px 14px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff3d9;padding:16px}}h2{{margin-top:32px}}</style>
<h1>铜矿反演 r3：修复后训练阶段报告</h1><p class="note">北京时间 {stamp}。训练尚未完成，正式验证选型与 ID/OOD 测试尚未执行。这不是最终结果。</p>
<h2>同预算结论：{n} 个收费源码</h2><p>{conclusion}赢家求解较快不等于端到端发现更快；LDM 的不同种子差异较大，不能只选最快或损失最低的种子代表方法。</p>
<table><tr><th>方法</th><th>损失均值 ± 样本SD</th><th>赢家平均求解秒 ± 样本SD</th><th>累计求解小时均值</th></tr>{rows}</table>
<p>三个发现种子均为42、43、3407。每次求解统一120秒/2CPU/4GiB和相同地质约束；主指标为父体等权的独立确认测量 L/(1+L)。阴影和±为三种子样本标准差，不是置信区间。</p>
<img src="training-prefix-{n}.png" alt="同预算损失、累计求解作业时间以及三个种子损失和赢家求解时间">
<p>累计求解小时是该收费前缀中已完整评分源码的solver作业墙钟时长之和，再对三个发现种子取均值；并行作业时间相加，不是实际经过的时间，也不是CPU小时。接口、研究、API、启动、排队、暂停与恢复成本不在其中。全运行预算计数保存在聚合JSON中，但没有冒充对齐该前缀的总发现成本。接口拒绝照常占收费预算。</p>
<h2>三个种子</h2><table><tr><th>方法</th><th>发现种子</th><th>最佳损失</th><th>赢家求解秒/case</th></tr>{seedrows}</table>
<h2>九组进度与异常检查</h2><p>修复epoch以来新增{jobs}份作业收据，失败{failures}份；共享暂停为{d['paused']}。新provider响应文件{newresponses}份。若为0，表示当前尚在补完原批次评分，不能据此宣称新提交计时路径已在正式研究中经过验证。正式服务和客户端进程、最新收据、checkpoint及预算均已核对。</p>
<table><tr><th>组</th><th>已收费</th><th>连续已结算前缀</th><th>完整评分记录</th><th>累计收费拒绝</th><th>修复后作业收据</th></tr>{progress}</table><p>完整评分记录包含私有拒绝返回的惩罚结果，不能直接等同于合法求解数；连续已结算前缀包括本地收费拒绝。乱序完成和仅预留槽位均不用于跨方法比较。</p>
<h2>修复边界和预算保留</h2><p>每候选六接口并行；研究及客户端计时扣除有界宿主验证等待；基线SIGALRM清理采用嵌套finally捕获。217项Python回归、14项sidecar容器测试、1308作业峰值128满负载、192并行接口及102份开发私有闭环收据通过。原冻结评分与数据未改，接管前118858份文件摘要核对不变。</p><p>六条基线各129个已收费源码保持原字节；后续未收费参数才使用信号修复模板。已收费未完成源码仍按原代码补完，不免费重评。MCMC42第100和122收费源的历史信号失败保留，本次{n}源前缀包含其中{sum(pos <= n for pos in (100,122))}次，已按原收费位置保留；LDM43第34/35位收费拒绝与LDM3407的私有拒绝已计入当前前缀。新研究session用新sidecar，历史session保留原manifest/image。报告会继续区分新旧执行版本。</p>
<p>LDM42第11轮新增两个候选拒绝：一个将浮点数误作数组调用copy，另一个在两个接口中用满5秒硬限后被终止（约5.2 CPU秒）。共6份失败接口收据，属于两个收费源，位置162/163；本次前缀含{sum(pos <= n for pos in (162,163))}个。未发现宿主串行校验等待或基线信号竞态复发。研究端已自行继续提交；不由监控修改候选、不退还预算。监控汇总已纳入新拒绝原序，失败收据不是控制器失败。</p>
<p>22:36新增ACQ43第221收费源私有接口超时：worker实测5.65秒、容器累计CPU5.83秒，退出137、无OOM。源码摘要匹配signal-cleanup-r2模板，未见SIGALRM异常栈；不能将实际执行超限归为宿主校验排队，也不能据此确定具体计算热点。该拒绝计费、不免费重跑，当前比较前缀尚未包含它。九组持续推进。</p>
<p>10月26日00:21共享Docker控制面超时后暂停，期间128并发压测有接口超限；降低至96作业（88求解评分＋8接口）、启动并发1后，1308真实容器全部通过。01:40核对原作业、账本和收据后接管；额外修复启动子队列旧暂停标记遗漏。当前九组继续训练，降低并发及全部暂停/验收/恢复开销不计作算法加速。</p><p>LDM42第13轮另有7个研究阶段收费拒绝，按验证journal错误指针、源码摘要及完成时间定位在第196–202收费位。当前{n}源比较包含其中{sum(pos <= n for pos in range(196,203))}个。ACQ3407有一份00:21事故期间私有接口超限收据，监控保留且不免费重评；不能把历史故障当作96并发接管后新增故障。</p><p>所有正式训练完成后执行封存→验证选型→固定赢家后ID/OOD测试。正式测试矿体可视化在独立评估完成后发布，不向研究端泄露测试真值。</p>
<p>2026-10-26 07:36: 03:44 fixed 7200-second socket deadline interrupted live scoring. Async submit/status transport passed 27 regressions, 102 development closure receipts, and combined 1308-container qualification at peak64. Formal continuation at 06:03 retained 233177 accounting/receipt/result files. Current capacity is 56 solver/scoring + 8 interface, startup1. All nine groups continue; transport, qualification and recovery costs are not algorithm speedups. A new LDM3407 private interface exceeded its genuine 5-second execution limit (5.012s elapsed, 5.090 CPU seconds, exit137, no OOM). No signal traceback or transport failure is evidenced; its computation hotspot is unknown. The paid rejection remains charged, with no free reevaluation.</p><p><a href="training-prefix-{n}.pdf">图表PDF</a> · <a href="{a.snapshot.name}">聚合JSON与完整预算计数</a></p></html>'''
(a.output/'report.html').write_text(html);(a.output/a.snapshot.name).write_bytes(a.snapshot.read_bytes())
print(json.dumps(dict(prefix=n,time=stamp,summary=s,failed_jobs=failures,new_receipts=jobs)))
