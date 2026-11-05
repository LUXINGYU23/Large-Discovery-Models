"""Render the frozen development comparison, without accessing formal holdout."""
from pathlib import Path
import json,html
B=Path('/Aidata/shared/ldm/copper-baseline-audit-20261027')
out=B/'report'
a=json.loads((out/'actual-time-summary.json').read_text())
assert a['solves']==432 and a['failed']==0 and a['r3_control_hashes_unchanged']
p=[]
def add(s):p.append(s)
def table(headers,rows):
 add('<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>')
 for row in rows:add('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>')
 add('</tbody></table>')
add('<h1>MCMC / ACQ适配审计与同实际时间开发比较</h1><p>2026-10-27 · 独立于r3，保留原结果、原候选和收据。以下为开发验证比较，不是重新测试r3，也不是新的独立泛化证明。</p>')
add('<h2>主要结论与根因</h2><p>r3原适配器将超过四个连通体的提议直接拒绝；高斯场水平集提议频繁产生孤立小体。对齐r3公开先验的开发复现中，种子43约94.5%的提议因此失效，评分调用中位数仅7.5（含初始化）。ACQ的warmup只累计已评分提议，门槛8尚未满足或满足时已到截止时间，16次复现均未进入采集决策。它不是未部署服务或关闭的代码分支。</p><p>另外捕获了四份默认二阶fast marching翻转体素符号的数值反例。一阶格式在三种尺度下消除这些反例，但仅修这一点仍有约94.5%的连通体拒绝。因此不能将主要异常归咎于数值格式。</p>')
table(['开发训练版本（固定种子43参数）','方法','超连通体拒绝/提议','评分调用中位数','采集启动','公开原始损失'],[
 ['原适配＋诊断','MCMC','1861/1968',7.5,'—','1.822243'],['原适配＋诊断','ACQ','1859/1967',7.5,'0/16','1.822243'],
 ['仅一阶距离','MCMC','1856/1963',7.5,'—','1.822229'],['仅一阶距离','ACQ','1857/1964',7.5,'0/16','1.822229'],
 ['可行投影＋一阶距离','MCMC','0/1901',125,'—','1.412756'],['可行投影＋一阶距离','ACQ','0/1921',38,'16/16','1.574724']])
add('<h2>修复及验收边界</h2><p>在公开可行域内投影提议：超出连通体上限时按确定规则保留最大的连通体；体积越界使用已有受限连通构造；合法提议保持原连续场，不抹掉亚体素位移。之后使用一阶距离，并继续严格检查占据符号和全部硬约束。未调低warmup，未改温度、步长、六份冻结配置或评分定义。32次开发验收全部成功、无invalid，ACQ全部启动并实际在4候选间决策；14项数值检查和10项远程pytest通过。</p><p><strong>投影改变了提议分布。</strong>修复版是受约束随机优化适配器，不能声称仍满足严格后验MCMC的详细平衡；当前ACQ是共享提议族上的Bayesian ridge采集适配器，不代表所有采集方法。启动成功也不等于代理决策一定有收益。</p>')
add('<h2>冻结后的同实际时间比较</h2><p>4个开发验证父体、8案例，均与开发训练父体分离。3份LDM原冻结赢家和6份修复后基线，执行种子47/48/49，30秒和90秒两个目标，共432次成功求解。各任务2CPU、4GiB，统一worker入口绝对截止时间，计入导入及初始化，留1秒输出，额外5秒硬保护；抵消原算法不同的软截止比例，不sleep补齐。LDM42的小网格提前结束特例仅在计时副本关闭，全部计时副本另存摘要。下表时间均为实测worker时间，不用目标值替代。</p><p>均值±标准差跨三个冻结发现种子；父体先等权聚合，标准差不是置信区间。公开观测损失L/(1+L)与IoU分别报告。本开发验证集没有独立确认测量，不能与r3独立确认损失横向混为同一指标；也没有重新做等发现预算的算法搜索。</p>')
rows=[]
for r in a['summary']:
 rows.append([r['target_seconds'],r['method'].upper()]+[f"{r[k]['mean']:.6f} ± {r[k]['sd']:.6f}" for k in ['bounded_public_loss','iou']]+[f"{r['worker_seconds']['mean']:.3f}",f"{r['wall_seconds']['mean']:.3f}",f"{r['cpu_seconds']['mean']:.3f}"])
table(['目标秒','方法','公开损失↓','IoU↑','worker秒','容器墙钟秒','CPU秒'],rows)
add('<figure><img src="actual-time-quality.png"><figcaption>时间—质量关系，误差条为跨发现种子样本标准差。<a href="actual-time-quality.pdf">矢量PDF</a></figcaption></figure>')
add('<p><strong>结果解释：</strong>30秒档公开损失为LDM 0.122503、MCMC 0.557041、ACQ 0.560510；90秒档为0.092483、0.547681、0.556719。LDM在两个时间档的四个父体等权比较中均更低，说明这批开发数据上的优势不能仅由原来多用24%的时间解释。但这不是独立确认误差，不能把这里约78%–83%的相对降低搬到r3泛化结论中。</p><p>90秒档IoU分别为0.1822、0.1338、0.1342，几何恢复仍弱。LDM增加时间后公开损失改善，而平均IoU由0.2023降至0.1822：继续优化稀疏观测并不保证恢复真实形状。四父体中也有一个父体的LDM IoU略低于基线，不能只展示均值。</p>')
add('<h2>逐组执行与搜索质量</h2>')
table(['组','目标秒','公开损失','IoU','实测秒范围','调用中位数','invalid','ACQ启动/24','投影率','评分后占据未变率'],[[r['group'],r['target_seconds'],f"{r['bounded_public_loss']:.6f}",f"{r['iou']:.4f}",f"{r['worker_min']:.2f}–{r['worker_max']:.2f}",r.get('calls_median','—'),f"{r['invalid_ratio']:.2%}" if 'invalid_ratio'in r else '—',r.get('acquisition_active','—'),f"{r['projected_ratio']:.1%}" if 'projected_ratio'in r else '—',f"{r['unchanged_scored_ratio']:.1%}" if 'unchanged_scored_ratio'in r else '—'] for r in a['groups']])
add('<p>ACQ每次决策需构建多个提议并拟合代理，所以同时间下真实评分调用可能少于MCMC；只有最终质量改善才能证明这笔开销值得。可行投影率高说明原提议族仍与硬拓扑约束不匹配，投影修复了可执行性，不能据此宣称已找到最优提议设计。</p>')
add('<p>本次90秒档MCMC评分调用中位数100.5，ACQ为29.5；ACQ公开损失反而高约1.65%，未证明采集开销值得。基线仍有部分执行最佳值等于初始化：MCMC 18/72、ACQ 21/72。它们已实际评分不同占据形状，不能再归因于未启动或全部提议非法；仍可能存在提议效率、代理特征和冻结参数与任务不匹配的问题。本轮没有针对验证结果继续改设计，也没有把“运行问题修复”包装成“已充分调优强基线”。这些剩余机制的区分需要新的开发训练消融，不能由本比较直接断言。</p>')
add('<h2>数据隔离、撤销运行与证据</h2><p>原开发数据没有r3体积/连通体上限，面积正则也不同。识别后停止了自己的早期比较控制器，保留INVALID-POLICY记录，该批结果不进入本报告。最终开发副本通过冻结public_priors函数对齐公开先验；24案例真值均符合约束。12个开发父体与r3测试24父体零重叠，仅核对manifest身份，未读取测试反馈进行修复或选参。</p><p>r3的29份完成标记、账本及选型控制文件SHA256复核不变，r3候选未覆盖。开发验证只在方案冻结后运行，不依验证结果调参；若未来需要泛化结论，应另建未见父体与独立确认测量，当前已揭示r3测试集不得复用为未见测试。</p>')
add('<ul>')
for name in ['README.md','actual-time-cases.csv','actual-time-summary.json','actual-time-plan.json','comparison-qa.json','policy-acceptance.json','numeric-regression.json','pytest-projected.log','split-isolation.json','population-qa.json','r3-preservation-before.json']:
 add(f'<li><a href="{name}">{name}</a></li>')
add('</ul><p>九份冻结计时源码：'+ ' · '.join(f'<a href="comparison-sources/{g}/solver.py">{g}</a>' for g in sorted({r['group'] for r in a['groups']}))+'</p><p><a href="../copper-measurement-study-r3/report.html">返回原r3报告</a></p>')
css='body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.8 system-ui;color:#203840}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd9db;padding:8px;text-align:left}th{background:#eaf4f4}img{max-width:100%}figure{margin:24px 0}a{color:#087e8b}@media print{tr,figure{break-inside:avoid}}'
(out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>基线适配与同时间开发审计</title><style>'+css+'</style><body>'+''.join(p)+'</body></html>')
print('rendered',out/'report.html')
