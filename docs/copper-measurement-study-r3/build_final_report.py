from pathlib import Path
import json, html, sys

r = Path(sys.argv[1])
read = lambda name: json.loads((r/name).read_text())
test = read('formal-test-summary-20261027-0706.json')
completion = read('formal-completion.json')
cost = read('cost-ledger-summary-20261027-0806.json')
holdout = read('holdout-cost-summary-20261027-0836.json')
queue = read('queue-cost-summary-20261027-0836.json')
qualification = read('qualification-cost-inventory-20261027-0906.json')
parts=[]
def p(s): parts.append(s)
def table(headers, rows):
    p('<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>')
    for row in rows: p('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>')
    p('</tbody></table>')
def fig(name, caption): p(f'<figure><img src="{name}.png"><figcaption>{caption} · <a href="{name}.pdf">PDF</a></figcaption></figure>')
p('<h1>铜矿几何反演 r3：正式评估技术报告</h1><p class="lead">九组训练、私有验证选型和固定赢家 ID/OOD 测试已完成。LDM在三个发现种子的ID测试中均降低独立测量损失，但OOD优势不稳定，且求解约慢24%。</p>')
p('<p>范围：仅 copper-measurement-study-20261025-r3。旧r1/r2及开发验收不参与科学结果比较。报告日期：2026-10-27。数据、源码摘要、全部案例指标与成本附件均保留，以下未知成本不按零处理。</p>')
p('<aside style="border:2px solid #b45309;padding:18px;background:#fff7ed"><strong>2026-10-27补充：基线适配问题已在开发集定位并修复验收。</strong><p>原r3六基线提议invalid占88.7%–93.6%，评分调用中位数仅8–12（含初始化）；ACQ43的144次求解有114次未启动采集。对齐r3公开先验的开发复现表明，主要原因是高斯场提议超过四个连通体被拒绝，ACQ有效样本不足以达到warmup；另有独立的二阶距离重初始化符号翻转问题。原日志只有总invalid，不能将每一条历史拒绝事后逐一归因。</p><p>可行域投影加一阶距离通过32次开发验收，保留硬约束与冻结参数，ACQ全部实际启动。投影改变提议分布，修复版属于受约束随机优化适配器，不宣称严格后验MCMC。另完成432次30/90秒档的同实际时间开发验证比较，详见<a href="../copper-baseline-audit-20261027/report.html">适配审计与公平时间比较</a>。该开发集只有公开观测目标和几何真值，不是r3独立确认测量，也不构成新的独立泛化证明。</p><p>因此原报告证明的是LDM优于当时的适配实现，不能据此断言优于充分适配的强基线；24%只是原求解时间差。r3原结果、候选与收据全部保留，已揭示测试集未用于此次修复或调参。</p></aside>')
p('<h2>1. 结论与实验协议</h2><p>主指标是每次求解独立确认测量损失 L/(1+L)，先在父体内平均，再对父体等权。IoU是辅助几何指标。表中均值±样本标准差跨发现种子42/43/3407，不是置信区间。三个执行种子47/48/49与发现种子分开记录。</p><p>每组最多321个唯一收费源码，接口拒绝占预算且不补评。训练48案例、验证16案例；验证在三个冻结候选中选型；赢家固定后测试ID32案例×3执行种子、OOD16案例×3执行种子。全部组先封存、全部组完成选型，之后才进行测试。相同单候选上限120秒、2CPU、4GiB；非活动区、体积2%–65%与最多4连通体约束一致。相同上限不意味着相同实际耗时。</p>')
rows=[]
for m in ['ldm','mcmc','acq']:
    a=test['summary'][m]; vals=[a[s]['loss_total'] for s in ['test_id','test_ood']]
    rows.append([m.upper()]+[f"{v['mean']:.6f} ± {v['std']:.6f}" for v in vals]+[f"{a['test_id']['mean_iou']['mean']:.4f} / {a['test_ood']['mean_iou']['mean']:.4f}",f"{a['test_id']['mean_solver_wall_seconds']['mean']:.2f} / {a['test_ood']['mean_solver_wall_seconds']['mean']:.2f}"])
table(['方法','ID损失','OOD损失','ID/OOD IoU','ID/OOD求解秒'],rows)
p('<p>LDM42与3407在OOD仍领先，而LDM43的OOD损失0.491752劣于同种子基线。LDM平均OOD损失更低不能解释为每次运行都具有跨分布优势。全部测试合法率100%、失败求解0；这说明本批固定赢家执行成功，不抹除训练阶段的收费拒绝和基础设施故障。</p>')
fig('formal-test-loss','独立测试：均值±跨发现种子样本标准差；叉号为三个发现种子')
p('<h2>2. 训练预算、验证选型与泛化</h2><p>九组收费源并集均为321，对账通过。私有结果数可以少于321，差额由收费拒绝解释，不强补。LDM42为308私有完成结果+13本地研究拒绝；LDM43为319+2。其余差集详见预算对账JSON。所有已收费源码原字节保留。</p>')
rows=[]
for g,t in sorted(test['groups'].items()):
    v=completion['results']['validation'][g];winner=t['selected_sha256'];val=next(c['metrics']['loss_total'] for c in v['candidates'] if c['source_sha256']==winner)
    losses={x['split']:x['metrics']['loss_total'] for x in t['results']}
    rows.append([g,f'{val:.6f}',f"{losses['test_id']:.6f}",f"{losses['test_ood']:.6f}",winner[:16]])
table(['组','验证损失','ID损失','OOD损失','固定赢家SHA256前缀'],rows)
p('<p>训练原序共同完整前缀321：LDM0.374606±0.036835、MCMC0.411590±0.000895、ACQ0.411598±0.000993。训练最佳与验证选定赢家可能不同，不能把两者视为同一候选的轨迹。完整赢家摘要和验证候选结果见formal-completion.json。</p>')
fig('training-prefix-321','同收费原序完整前缀下的训练比较；旧实验曲线未拼接')
p('<h2>3. 冻结算法解析</h2><p>解释来自冻结赢家源码；未用测试结果继续改动算法。九份原字节源码位于sources/，完整摘要见winner-source-hashes.json。</p>')
table(['方法','实际策略与限制'],[
['LDM42','观测驱动排序和初始化、物理尺度球形提议、模拟退火及最佳合法场保留。CFG选择anneal/sphere；不把未启用的portfolio等分支算作实际算法。搜索约用94%剩余预算。'],
['LDM43','Voronoi初始化，按CFG阶段计划进行合法边界搜索，每阶段只在改进时更新最佳场，最终canonicalize。公开观测目标的优化未保证本种子的OOD收益。'],
['LDM3407','中心加权随机傅里叶特征、多长度尺度集成、分位阈值搜索和合法投影，随后局部球形边界修改。stage_a=0，因此特征权重扰动阶段未启用。'],
['MCMC','基线包高斯随机场提议、速度扩展和水平集更新，按平方损失差进行随机接受，保留最佳合法状态。这是适配物理单位及约束的MCMC-v2，不是原论文逐字复现。'],
['ACQ','与MCMC相同提议族；warmup后用在线Bayesian ridge预测损失和不确定性，以mu-beta*sigma挑选候选，再实际评分和接受。结果接近MCMC，不能仅凭算法名宣称加速。']])
p('<p>基线软搜索截止约为剩余预算75%，LDM通常约94%–95%，所以同120秒上限下实际时间仍不同。LDM优势不是神经网络直接输出真值：这些赢家本身是数值初始化、随机搜索和合法性修复程序。</p>')
p('<h2>4. 全案例、随机性与几何恢复</h2><p>all-test-cases.csv包含1296次固定赢家求解，涵盖父体、案例、执行种子、独立确认损失、观测拟合、IoU、相对体积误差、连通体数及时间。execution-seed-stability.json提供54个组×split×执行种子的父体等权损失。不得把发现种子标准差和执行种子变化混合。</p><p>以下统一选择各split中字典序最小case_id，发现种子固定42、执行种子固定47；规则不依赖指标。3D的原点、网格间距、坐标范围及视角一致，黑点为公开观测、灰点为非活动区域，标题列出体积和连通体数。坐标单位依数据集定义，不擅自标为米。三正交切片取各轴中间体素；绿为重叠、蓝为漏检、橙为误检、灰为非活动区。</p>')
stability=read('execution-seed-stability.json')
table(['组','分布','执行47损失','执行48损失','执行49损失'], [[g,s]+[f"{next(x['parent_equal_loss'] for x in stability if x['group']==g and x['split']==s and x['execution_seed']==seed):.6f}" for seed in [47,48,49]] for g in sorted(test['groups']) for s in ['test_id','test_ood']])
for split in ['test_id','test_ood']:
    fig(split+'-geometry',split+'固定案例：真值 / LDM / MCMC / ACQ，同坐标几何')
    fig(split+'-slices',split+'固定案例：三个正交中间切片，体素索引坐标')
p('<p>观测稀疏导致反演非唯一。更好的测量拟合不等于准确恢复真实几何，固定案例也不代表总体表现。这里是合成基准，尚不能外推为真实矿区勘探效果。</p>')
p('<h2>5. 资源与成本</h2><p>以下“小时”是各作业墙钟之和，不是整场实验历时，也不是CPU小时。并发等待之和不能直接加到运行时间上推算交付时间。CPU through-reap记录和接口/评分分项均在JSON附件中；研究时钟与API、接口可能存在嵌套，不将它们相加为无重叠总额。</p>')
rows=[]
for g,v in sorted(cost.items()):
    c=v['research'];t=v['private_training'];rows.append([g,f"{c.get('harness_wall_time_seconds',0)/3600:.2f}",f"{c.get('interface_wall_seconds',0)/3600:.2f}",f"{t.get('interface_wall_seconds',0)/3600:.2f}",f"{t.get('solver_wall_seconds',0)/3600:.2f}",f"{t.get('scorer_wall_seconds',0)/3600:.2f}",c.get('llm_requests',0)])
table(['组','研究时钟h','研究接口h','训练私有接口h','训练求解h','训练评分h','LLM请求计数'],rows)
rows=[]
for g,st in sorted(holdout.items()):
    for stage,v in st.items():rows.append([g,stage]+[f"{v[k]['wall_seconds']/3600:.3f}" for k in ['interface','solver','scorer']]+[f"{sum(x['cpu_seconds'] for x in v.values())/3600:.3f}"])
table(['组','阶段','接口h','求解h','评分h','CPU h'],rows)
p('<p>验证与测试共3726容器作业：270接口、1728求解、1728评分；零失败，资源字段完整。完成标记的5454收据另含1728案例级收据，不能当作额外容器执行重复统计。</p>')
table(['组','研究等待h','训练等待h','验证测试等待h'],[[g]+[f"{v[k]['summed_wait_seconds']/3600:.2f}" for k in ['research','training','holdout']] for g,v in sorted(queue.items())])
p('<p>等待来自jobs/scheduling记录；中断且未返回收据的作业可能没有此记录，所以属于已记录等待而非严格全量墙钟恢复成本。API只对每个响应最后一份usage求和，避免流式重复计数。</p>')
table(['组','响应数','缺usage','输入tokens','缓存输入tokens','输出tokens'],[[g]+[v['provider_usage'][k] for k in ['response_files','missing_usage','input_tokens','cached_input_tokens','output_tokens']] for g,v in sorted(cost.items()) if 'provider_usage' in v])
p('<p>请求计数与响应文件数不完全相同。缺失usage的3/6/8份响应分别属于LDM42/43/3407，未推定为免费；没有供应商账单，不能给出完整货币成本。</p>')
p('<h2>6. 修复、恢复与版本边界</h2><p>六基线最初各129个收费源码保留旧模板，后续未收费源码按冻结映射使用signal-cleanup修复。修复包括六接口并行、有界宿主验证等待从研究时钟扣除、SIGALRM嵌套finally清理。已收费失败不退款、不替换候选。历史资源调度由128/96等验收路径收敛至64总作业（56求解评分+8接口）、startup1、每进程32；“80%CPU”是目标，稳定性优先。</p>')
table(['时间（CST）','事故/恢复及成本披露'],[
['10-26 00:21起','Docker启动控制故障；36未完成作业按证据协调，未重启共享Docker。128控制压测有2接口超时，后续降并发。'],
['10-26 03:44—约06:03','固定7200秒socket在评分仍推进时超时。改为短submit/status及完成计数进展监控。96联合验收5接口超时失败，64联合1308作业零失败才恢复。约2小时19分钟为故障到恢复日历区间，不是所有工作完全停止的精确时间。'],
['10-27约00:57—04:10','LDM43研究turn计数触100。证明3个round5输入是metrics键序列化别名，保留全部原usage和budget-before，仅校正唯一计数至97；合法round20预留不退款。持久化exact输入、回归和摘要固定后单次接管，100/321上限不变。约3小时13分钟为诊断恢复区间。'],
['10-27 05:06—07:06检查','九组训练对账、封存后验证选型和固定赢家测试完成；研究不再运行。测试数据已揭示，不能重新称作未见评估或反馈研究。']])
p('<p>下表为OPS中已定位的主要small/load验收目录，包含失败运行和未完成早期load；独立通过不能替代联合通过。它们是工程成本，不属于科学训练收益，且不能宣称覆盖全部开发劳动、测试进程或每个私有开发闭环。</p>')
table(['验收目录','容器数','失败数','墙钟合计h','CPU h'],[[g,v['container_receipts'],v['failed'],f"{v['wall_seconds']/3600:.3f}",f"{v['cpu_seconds']/3600:.3f}"] for g,v in sorted(qualification.items())])
p('<p>恢复人工/代理调试时长、部分无收据中断及未定位开发作业的完整成本不可由现存汇总无歧义重建；明确列为未完全量化，而非零。完整时序和证据文件名见资格记录及report-delivery文档。总成本结论仅限可观测分项，不给出虚假的精确全包成本。</p>')
p('<h2>7. 可复核交付</h2><ul>')
for name in ['formal-completion.json','budget-reconciliation.json','formal-test-metrics.csv','formal-test-summary-20261027-0706.json','all-test-cases.csv','execution-seed-stability.json','winner-source-hashes.json','geometry-case-selection.json','cost-ledger-summary-20261027-0806.json','holdout-cost-summary-20261027-0836.json','queue-cost-summary-20261027-0836.json','qualification-cost-inventory-20261027-0906.json']:
    p(f'<li><a href="{name}">{name}</a></li>')
p('</ul><p>全部赢家源码：'+ ' · '.join(f'<a href="sources/{g}.py">{g}</a>' for g in sorted(test['groups']))+'</p>')
css='body{max-width:1180px;margin:40px auto;padding:0 28px;font:16px/1.8 system-ui;color:#19333d}h1{font-size:32px}h2{margin-top:42px;border-bottom:2px solid #b9d4d5}.lead{font-size:20px;background:#e5f4f2;padding:20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #c9d5d8;padding:9px;text-align:left;overflow-wrap:anywhere}th{background:#e5f0f2}img{width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:13px;color:#4a6269}a{color:#087e8b}@media print{body{margin:0;max-width:none}h2{break-before:page}figure,tr{break-inside:avoid}}'
(r/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>铜矿r3正式技术报告</title><style>'+css+'</style><body>'+''.join(parts)+'</body></html>')
print('report.html generated')
