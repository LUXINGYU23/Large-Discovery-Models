# -*- coding: utf-8 -*-
"""Publish seed-42 investor report from immutable experiment exports."""
import json,hashlib,html,os,re
import pymupdf
from pathlib import Path
from playwright.sync_api import sync_playwright
from algorithm_sections import CSS as algorithm_css, build_algorithm_sections
from comparison_sections import CSS as comparison_css, build_comparison_sections
O=Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops');B=O/'seed42-investor-report-20261031'
d=json.loads((B/'data/report.json').read_text());methods=['ldm','mcmc','acq'];names=['LDM','MCMC','ACQ']
def met(m,s):return next(r['metrics'] for r in d['summary'] if r['method']==m and r['split']==s)
def table(headers,rows,cls=''):
 return '<table class="'+cls+'"><thead><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'
def image(n,caption,cls='chart'):
 return f'<figure class="{cls}"><img src="assets/{n}.png"><figcaption>{caption}</figcaption></figure>'
css='''@font-face{font-family:ReportCJK;src:local("Noto Sans CJK SC")}*{box-sizing:border-box}body{margin:0;background:#e9eef3;color:#1d3045;font:15px/1.65 ReportCJK,"Noto Sans CJK SC",system-ui,sans-serif}a{color:#19689b}h1,h2,h3,p{margin-top:0}h1{font-size:35px;line-height:1.3;letter-spacing:-.8px}h2{font-size:26px;line-height:1.35;margin-bottom:17px}h3{font-size:17px;margin-bottom:7px}p{margin-bottom:12px}.nav{position:sticky;top:0;z-index:4;padding:12px 30px;background:#112c42;color:white;display:flex;gap:24px;justify-content:center}.nav a{color:white;text-decoration:none}.sheet{width:210mm;min-height:297mm;margin:22px auto;background:white;padding:16mm 17mm 15mm;position:relative;box-shadow:0 8px 30px #122b4220;break-after:page}.eyebrow{color:#197d88;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px}.lede{font-size:18px;line-height:1.65}.accent{color:#19689b}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:22px 0}.kpi{padding:16px 12px;border-radius:10px;background:#eef5fa;border-top:3px solid #2a7999}.kpi strong{display:block;font-size:27px;line-height:1.3;color:#185b82}.kpi small{font-size:11px}.chart{margin:14px 0}.chart img{width:100%;display:block}.chart figcaption{font-size:10px;color:#637588;line-height:1.6;margin-top:6px}.callout{padding:14px 16px;background:#eef6f7;border-left:3px solid #27828b;margin:15px 0;font-size:14px}.footer{position:absolute;bottom:9mm;left:17mm;right:17mm;border-top:1px solid #dbe4ec;padding-top:5px;color:#6d7c8e;font-size:9px;display:flex;justify-content:space-between}table{border-collapse:collapse;width:100%;font-size:11px;margin:10px 0 16px}th{background:#173e56;color:white;text-align:left;font-weight:600}th,td{padding:8px 7px;border-bottom:1px solid #dce5ed}td:first-child{font-weight:600}tbody tr:nth-child(2n){background:#f5f8fb}.small{font-size:11px;color:#607386}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}.flow div{padding:14px 10px;background:#eef4f7;border-radius:8px;font-size:12px}.flow b{display:block;color:#1d687d;margin-bottom:8px}.method{border:1px solid #dce6ef;border-left:4px solid #2563a6;padding:15px;margin:13px 0;border-radius:7px}.method.orange{border-left-color:#d38b26}.method.purple{border-left-color:#9c4d9c}.method p{font-size:13px;margin-bottom:0}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:22px}.tag{font-size:11px;display:inline-block;border:1px solid #bdd3de;padding:3px 8px;border-radius:20px;margin:3px}.casefigure{width:100%;margin:12px 0}.casefigure img{width:100%;display:block}.casecaption{font-size:10px;color:#607386}.dense th,.dense td{padding:6px;font-size:10px}.parentplot{width:90%;margin:8px auto}.parentplot img{width:100%}.parentplot figcaption{font-size:10px;color:#607386}.cover .eyebrow{margin-top:8mm}.rule{height:4px;width:55px;background:#1b8391;margin:22px 0}.sourcecode{font-family:monospace;font-size:9px;word-break:break-all}@media print{@page{size:A4;margin:0}body{background:white}.nav{display:none}.sheet{width:210mm;height:297mm;min-height:0;margin:0;box-shadow:none;overflow:hidden}a{color:inherit;text-decoration:none}}'''
css+=algorithm_css
css+=comparison_css
sheets=[]
def sheet(title,content,eyebrow,cover=False):
 i=len(sheets)+1;sheets.append(f'<section class="sheet {"cover" if cover else ""}" id="page{i}"><div class="eyebrow">{eyebrow}</div><h2>{title}</h2>{content}<div class="footer"><span>铜矿三维反演 · 发现种子42 · 更新2026.11.02</span><span>{i:02d} / PAGECOUNT</span></div></section>')
# 1. Executive effect report.
content='<h1>让自动算法研究<br>转化为更好的三维反演结果</h1><p class="lede">在本次铜矿合成基准中，LDM的独立测量损失低于修复后的MCMC与ACQ，ID与OOD测试均获得改善。</p><div class="kpis"><div class="kpi"><strong>13.1–14.7%</strong><small>独立测量损失相对降低<br>覆盖ID/OOD及两种对照</small></div><div class="kpi"><strong>36.7–59.7%</strong><small>几何IoU相对提升<br>对应绝对提升0.041–0.052</small></div><div class="kpi"><strong>+22.2秒</strong><small>每案例实际求解时间<br>约113.3秒 vs 91.1秒</small></div></div>'
content+=image('effects','实色为ID，浅色为OOD。损失与IoU按地质母体等权聚合；时间为实际求解墙钟秒。')
content+='<div class="callout"><b>核心结果：</b>种子42的LDM固定赢家在两类测试分布上均实现更低测量损失和更高几何重叠率，实际单案例求解时间增加约24.4%。</div><p class="small">范围：发现种子42；48个测试案例，每案例执行47/48/49，三种方法合计432次测试求解，全部成功。数据为已完成实验的冻结结果。</p>'
sheet('01 / 摘要与效果报告',content,'EXECUTIVE RESULTS',True)
# 2. Problem.
content='<p class="lede">将稀疏钻孔与露头观测，转化为可分析、可对照的三维矿体几何。</p><div class="rule"></div><p>铜矿反演需要在有限观测下估计矿体的空间范围、边界和连通关系。相同观测可能对应多种几何解释，因此算法既要拟合可见观测，也要满足公开的地质约束。</p><div class="flow"><div><b>01 观测输入</b>钻孔和露头观测点及其标签</div><div><b>02 空间约束</b>活动区、禁入区、体积与连通体约束</div><div><b>03 三维反演</b>在约束内搜索矿体形状与边界</div><div><b>04 独立评估</b>用确认观测与真值几何检验输出</div></div>'
content+='<h3>这份实验回答什么问题</h3><p>在相同的候选反馈额度与单次求解资源上限下，自动研究算法的LDM，能否比固定形式的随机搜索与采集引导搜索，得到更好的可迁移求解器？</p><div class="twocol"><div><h3>主指标：独立测量损失</h3><p>使用未交给求解器的确认观测检验预测。先将每次确认损失转换为效用，再按地质母体等权聚合；最终损失越低越好。</p></div><div><h3>几何指标：IoU</h3><p>比较预测矿体与真值矿体的体积重叠。IoU越高，空间范围越接近真值；这是辅助诊断指标。</p></div></div>'
content+='<div class="callout">主指标聚合：<b>Loss = 1 − 地质母体均值［执行与观测均值［1 / (1 + 确认损失)］］</b>。图表中的损失百分比表示该聚合指标的相对降低。</div><h3>展示材料中的三维图</h3><p>每组图始终同时展示真值、LDM、MCMC和ACQ，采用相同坐标范围和切片位置；灰色为非活动区，黑线为真值轮廓，彩点为公开观测。</p>'
sheet('02 / 问题介绍',content,'THE PROBLEM')
# 3. Methods.
content='<p class="lede">三种方法围绕同一反演问题展开，区别在于如何提出和选择下一步搜索方案。</p>'
content+='<div class="method"><h3>LDM：自动研究与改进求解算法</h3><p>研究阶段生成并修改完整Python求解程序，通过多分支候选生成、融合选择与迭代反馈改进算法。20轮研究，每轮选择16个候选，加初始化，共321个收费反馈槽位。种子42最终选出“多尺度球形翻转退火＋边界局部精修”；程序独立运行，每个案例内不再调用大模型。</p></div>'
content+='<div class="method orange"><h3>MCMC：投影随机优化基线</h3><p>外层按预设Sobol序列扫描超参数。每次求解通过随机场与形状扰动提出新状态，按目标函数与温度接受或拒绝，再将状态投影到公开的硬约束可行域，保留最优合法几何。</p></div>'
content+='<div class="method purple"><h3>ACQ：采集函数引导的随机优化</h3><p>外层同样采用预设Sobol超参序列。每次求解在投影随机搜索中引入在线贝叶斯岭回归代理模型。利用候选特征与采集分数，从候选提议中选择下一次真实评分方向。</p></div>'
content+='<h3>统一的训练与评估路径</h3><div class="flow"><div><b>训练搜索</b>48案例<br>最多321收费候选</div><div><b>候选冻结</b>各方法固定3份训练候选</div><div><b>验证选型</b>16案例<br>执行种子47</div><div><b>固定测试</b>ID32 + OOD16<br>执行47/48/49</div></div>'
content+=table(['实验项','设置'],[['发现种子','仅42'],['单次求解上限','120秒 / 2 CPU / 4 GiB'],['反馈计费','接口拒绝与失败占用额度，不补评'],['LDM结果来源','已冻结原实验赢家与既有收据'],['对照结果来源','修复后的MCMC / ACQ基线'],['选型规则','验证损失优先，依次比较合法率、耗时及源码标识']])
sheet('03 / 方法介绍',content,'THREE APPROACHES')
algorithm_md=build_algorithm_sections(B,d,table,sheet)
comparison_md=build_comparison_sections(B,d,table,sheet)
# 4. Training and validation.
content='<p>训练图显示批末累计收费槽位对应的最佳训练损失；字母对应前页源码节点；验证对三份冻结候选选型，星标为最终赢家。</p>'+image('training-validation','左图为外层搜索历史最佳训练损失；基线为预设Sobol调参，不是单次求解收敛曲线。<a href="trajectory.html">查看改动轨迹与源码</a> · <a href="curve-audit.html">基线原分数与内层轨迹</a>。')
rows2=[]
for m in methods:
 t=next(x for x in d['training'] if x['method']==m);g=next(x for x in d['validation'] if x['method']==m);v=next(x['metrics'] for x in g['validation_candidates'] if x['source_sha256']==g['selected_sha256']);rows2.append([m.upper(),321,t['evaluated_sources'],f"{t['training_best']['metrics']['loss_total']:.4f}",f"{v['loss_total']:.4f}",f"{v['mean_iou']:.4f}"])
content+=table(['方法','收费槽位','完整训练候选','训练最佳损失','选中验证损失','验证IoU'],rows2)
content+='<p>LDM种子42得到308份完整训练候选反馈，另外13份为收费接口拒绝；两种基线各321份完整候选反馈。测试使用验证选出的固定赢家。</p><h3>九份验证候选完整结果</h3>'
vr=[]
for g in d['validation']:
 for i,c in enumerate(g['validation_candidates']):vr.append([g['method'].upper(),c['source_sha256'][:12],f"{c['metrics']['loss_total']:.4f}",f"{c['metrics']['mean_iou']:.4f}",'选中' if c['source_sha256']==g['selected_sha256'] else '已评估'])
content+=table(['方法','源码摘要前缀','验证损失','验证IoU','状态'],vr,'dense')
sheet('04 / 实验结果：训练与验证',content,'TRAINING & SELECTION')
# 5. Test summary and paired parent distributions.
tr=[]
for s in ['test_id','test_ood']:
 for m in methods:
  a=met(m,s);tr.append(['ID' if s=='test_id' else 'OOD',m.upper(),f"{a['loss_total']:.4f}",f"{a['mean_iou']:.4f}",f"{a['mean_solver_wall_seconds']:.2f}"])
content=table(['测试集','方法','独立测量损失↓','IoU↑','实际秒/案例'],tr)
content+='<p>LDM相对MCMC的ID/OOD损失分别降低<b>13.12% / 13.68%</b>；相对ACQ分别降低<b>13.91% / 14.74%</b>。以下逐地质母体配对图覆盖全部测试母体。</p>'+image('parent-comparison','每个点为一个地质母体，包含其观测方式和三个执行种子的结果。点在虚线下方表示LDM损失更低。','parentplot')
content+='<p class="small">ID：16个地质母体；OOD：8个地质母体。相对MCMC，LDM分别在12/16与6/8个母体上获胜；相对ACQ分别为10/16与7/8。</p>'
sheet('04 / 实验结果：ID与OOD',content,'HELD-OUT RESULTS')
# 6. Execution and compute.
content=image('execution-seeds','发现种子始终为42；横轴仅切换实际求解的执行种子47/48/49。')
er=[]
for m in methods:
 er.append([m.upper()]+[f"{next(r['metrics']['loss_total'] for r in d['execution'] if r['method']==m and r['split']==s and r['execution_seed']==es):.4f}" for s in ['test_id','test_ood'] for es in [47,48,49]])
content+=table(['方法','ID/47','ID/48','ID/49','OOD/47','OOD/48','OOD/49'],er,'dense')
content+='<h3>实际计算量</h3><p>测试时LDM约113.3秒/案例，两种基线约91.1秒/案例，增加约22.2秒或24.4%。资源表为训练期间各任务累加的计数。</p>'
cr=[]
for g in d['validation']:
 c=g['training_private_resource_counters'];cr.append([g['method'].upper(),f"{c['solver_wall_seconds']/3600:.2f}",f"{c['solver_cpu_seconds_through_reap']/3600:.2f}",f"{c['interface_cpu_seconds_through_reap']/3600:.2f}",f"{c['scorer_cpu_seconds_through_reap']/3600:.2f}"])
content+=table(['方法','求解墙钟小时Σ','求解CPU小时Σ','接口CPU小时Σ','评分CPU小时Σ'],cr,'dense')
u=d['provider_usage'];content+=f'<p class="small">LDM研究harness累计6.20小时；已记录API输入{u["input_tokens"]:,} token（其中缓存输入{u["cached_input_tokens"]:,}），输出{u["output_tokens"]:,} token；3份响应未含usage。累计墙钟秒包含并发，harness、接口和求解可重叠，不相加为项目经过时间。</p><h3>资料索引</h3><p class="small">完整测试明细与指标：data/report.json；模型、收据与图像摘要：data/provenance.json；全部48案例及144套几何：browser.html。图示均来自冻结结果。</p>'
sheet('04 / 实验结果：执行与计算量',content,'EXECUTION & COMPUTE')
# 7-10. Full comparison examples, deterministic manifest order.
examples=[]
for split in ['test_id','test_ood']:
 seen=set()
 for c in sorted([c for c in d['case_metadata'] if c['split']==split],key=lambda c:c['order']):
  if c['parent_id'] not in seen:examples.append(c);seen.add(c['parent_id'])
  if len(seen)==2:break
examples=[examples[0],examples[2],examples[1],examples[3]]
for n,c in enumerate(examples,1):
 split=c['split'];cid=c['case_id'];key=f'{split}-{cid}-e47';rr=[r for r in d['test_rows'] if r['case_id']==cid and r['execution_seed']==47]
 content=f'<p class="small">案例 {cid} · {c["modality"]} · 执行种子47。展示按冻结清单顺序选择每个测试集前两个不同地质母体；全部48案例可在配套浏览器中切换。</p><figure class="casefigure"><img src="cases/{key}.png"></figure>'
 content+=table(['方法','确认观测原始损失↓','几何IoU↑','实际秒'],[[m.upper(),f"{next(r for r in rr if r['method']==m)['metrics']['confirmation_loss_total']:.4f}",f"{next(r for r in rr if r['method']==m)['metrics']['iou']:.4f}",f"{next(r for r in rr if r['method']==m)['solver_wall_seconds']:.2f}"] for m in methods],'dense')
 content+='<p class="casecaption">上行：真值与三种方法的三维表面；下三行：XY/XZ/YZ正交切片。采用同一坐标与切片平面，几何未做跨执行平均。黑色轮廓为真值、灰色为非活动区、彩点为公开观测。</p>'
 sheet(f'04 / 完整可视化：{"ID" if split=="test_id" else "OOD"}案例{1 if n<3 else 2}',content,'FULL CASE VISUALIZATION')
nav='<nav class="nav"><a href="#page1">摘要</a><a href="#page3">方法</a><a href="#page7">算法对照</a><a href="#page9">改进轨迹</a><a href="#page12">实验</a><a href="browser.html">完整案例</a><a href="trajectory.html">交互轨迹</a><a href="copper-seed42-investor-report.pdf">下载PDF</a></nav>'
(B/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>铜矿三维反演 | LDM 种子42结果报告</title><style>'+css+'</style>'+nav+''.join(sheets).replace('PAGECOUNT',str(len(sheets)))+'</html>')
# Offline all-case browser with common coordinates and linked cameras.
options=''.join(f'<option value="{c["split"]}-{c["case_id"]}">{"ID" if c["split"]=="test_id" else "OOD"} / {c["order"]+1:02d} / {c["modality"]} / {c["case_id"]}</option>' for c in sorted(d['case_metadata'],key=lambda c:(c['split'],c['order'])))
browser='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>种子42 | 全部案例与交互三维</title><script src="assets/plotly.min.js"></script><script src="data/report.js"></script><style>body{font:15px/1.7 system-ui,"Noto Sans CJK SC";margin:24px;background:#f2f6fa;color:#20364b}header{background:#173e56;color:white;padding:24px;border-radius:12px}header a{color:#bce3f0}select,button{padding:9px;margin:10px 8px 10px 0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #dde6ed;text-align:left}#panel{width:100%;background:white}.controls{position:sticky;top:0;background:#f2f6fa;z-index:3}.grid{display:grid;grid-template-columns:repeat(4,1fr)}.mesh{height:390px;min-width:0;background:white}.note{color:#597085;font-size:13px}.loader{padding:12px}h2{margin-top:30px}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}a{color:#16668b}</style><header><h1>发现种子42 · 完整案例可视化</h1><p>48案例 × 3执行种子 = 144套对齐比较。每套包含真值、LDM、修复后MCMC与ACQ。</p><a href="index.html">返回完整报告</a> · <a href="data/report.json">结果数据</a> · <a href="data/provenance.json">图像与模型来源摘要</a></header><div class="controls"><select id="case">OPTIONS</select><label>执行种子 <select id="exec"><option>47</option><option>48</option><option>49</option></select></label><button id="load3d">加载 / 更新交互3D</button><a id="original">打开完整静态图</a></div><div id="metrics"></div><h2>可旋转三维对比</h2><p class="note">四图采用同一物理坐标与相机。拖动任一图会同步其余视角；滚轮缩放，双击重置。红点/蓝点为公开观测标签。几何取原模型正值区域的体素边界，显示坐标保留至毫米。</p><div id="msg" class="loader">点击“加载交互3D”以显示当前案例的完整网格。</div><div class="grid"><div id="mesh0" class="mesh"></div><div id="mesh1" class="mesh"></div><div id="mesh2" class="mesh"></div><div id="mesh3" class="mesh"></div></div><h2>三维与三正交切片完整面板</h2><p class="note">每套均使用相同坐标范围和真值质心所在切片。黑线为真值，灰区为非活动区；不平均几何。</p><img id="panel" alt="真值及三种方法同坐标3D与XY、XZ、YZ完整切片"><script>
const c=document.querySelector('#case'),e=document.querySelector('#exec'),panel=document.querySelector('#panel'),msg=document.querySelector('#msg');let lastToken=0;let syncing=false;let loadedKey='';
function currentKey(){return c.value+'-e'+e.value}
function change(){let key=currentKey();panel.src='cases/'+key+'.png';document.querySelector('#original').href=panel.src;let cid=c.value.split('-').slice(1).join('-'),rows=REPORT.test_rows.filter(r=>r.case_id===cid&&r.execution_seed===Number(e.value));document.querySelector('#metrics').innerHTML='<table><tr><th>方法</th><th>确认观测原始损失</th><th>IoU</th><th>实际秒</th></tr>'+['ldm','mcmc','acq'].map(m=>{let r=rows.find(r=>r.method===m);return '<tr><td>'+m.toUpperCase()+'</td><td>'+r.metrics.confirmation_loss_total.toFixed(4)+'</td><td>'+r.metrics.iou.toFixed(4)+'</td><td>'+r.solver_wall_seconds.toFixed(2)+'</td></tr>'}).join('')+'</table>';msg.textContent='当前案例已更新；点击按钮加载对应3D。';for(let i=0;i<4;i++)Plotly.purge('mesh'+i);loadedKey='';}
async function showGeometry(){let key=currentKey(),token=++lastToken;msg.textContent='正在加载完整网格…';const s=document.createElement('script');s.src='geometry/'+key+'.js';s.onload=async()=>{if(token!==lastToken||key!==currentKey()){s.remove();return}const g=window.COPPER_GEOMETRY;for(let k=0;k<4;k++){const m=g.meshes[k],v=m.vertices,f=m.faces;let traces=[{type:'mesh3d',x:v.map(x=>x[0]),y:v.map(x=>x[1]),z:v.map(x=>x[2]),i:f.map(x=>x[0]),j:f.map(x=>x[1]),k:f.map(x=>x[2]),color:m.color,opacity:.85,flatshading:true,name:m.name,hoverinfo:'skip'}];g.observations.forEach(o=>traces.push({type:'scatter3d',mode:'markers',x:o.points.map(p=>p[0]),y:o.points.map(p=>p[1]),z:o.points.map(p=>p[2]),marker:{size:2,color:o.labels.map(l=>l===1?'#d95f4a':'#3896b4'),opacity:.45},hoverinfo:'skip',showlegend:false}));let aspect=g.upper.map((x,i)=>x-g.origin[i]);await Plotly.newPlot('mesh'+k,traces,{title:{text:m.name,font:{size:17}},margin:{l:0,r:0,t:42,b:0},showlegend:false,scene:{xaxis:{range:[g.origin[0],g.upper[0]],title:'X (m)'},yaxis:{range:[g.origin[1],g.upper[1]],title:'Y (m)'},zaxis:{range:[g.origin[2],g.upper[2]],title:'Z (m)'},aspectmode:'manual',aspectratio:{x:aspect[0]/Math.max(...aspect),y:aspect[1]/Math.max(...aspect),z:aspect[2]/Math.max(...aspect)},camera:{eye:{x:1.5,y:-1.5,z:1.1}}}},{responsive:true,displaylogo:false});let div=document.getElementById('mesh'+k);div.on('plotly_relayout',ev=>{if(syncing||!ev['scene.camera'])return;syncing=true;Promise.all([0,1,2,3].filter(i=>i!==k).map(i=>Plotly.relayout('mesh'+i,{'scene.camera':ev['scene.camera']}))).finally(()=>syncing=false);});}loadedKey=key;msg.textContent='已加载 '+key+'；四幅图可同步旋转。';s.remove();};s.onerror=()=>msg.textContent='几何文件加载失败，请保留整个报告文件夹。';document.body.appendChild(s);}
c.onchange=change;e.onchange=change;document.querySelector('#load3d').onclick=showGeometry;change();
</script></html>'''.replace('OPTIONS',options)
(B/'browser.html').write_text(browser)
# A concise Markdown companion with exact source links.
md='# 铜矿三维反演：发现种子42完整报告\n\n## 1. 摘要与效果报告\n\nLDM相对修复后的MCMC与ACQ，ID独立测量损失降低13.12% / 13.91%，OOD降低13.68% / 14.74%；单案例实际求解约113.3秒，对照约91.1秒。\n\n![总体效果](assets/effects.png)\n\n## 2. 问题介绍\n\n依据稀疏钻孔及露头观测，在公开地质约束内估计三维矿体。主指标按地质母体等权聚合独立确认观测效用，IoU衡量与真值矿体的体积重叠。\n\n## 3. 方法介绍\n\nLDM进行多分支候选生成、融合选择与迭代研究；MCMC采用投影随机优化；ACQ增加在线代理模型和采集选择。训练321收费槽位，单案例120秒/2CPU/4GiB；验证3候选×16案例×执行47；测试ID32/OOD16×执行47/48/49。\n\n## 4. 实验结果与完整可视化\n\n[完整排版报告](index.html) · [可打印PDF](copper-seed42-investor-report.pdf) · [全部144套案例及可旋转3D](browser.html) · [实验数据](data/report.json) · [来源摘要](data/provenance.json)\n\n![训练与验证](assets/training-validation.png)\n\n![逐母体配对](assets/parent-comparison.png)\n\n![执行种子](assets/execution-seeds.png)\n'
md=md.replace('## 3. 方法介绍\n\n','## 3. 方法介绍\n\n'+algorithm_md)
md=md.replace('## 4. 实验结果与完整可视化',comparison_md+'## 4. 实验结果与完整可视化')
(B/'README.md').write_text(md)
# Print the designed report with embedded CJK fonts; no external assets needed.
fontconf=O/'report-fonts/fonts.conf'
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'],env=dict(os.environ,FONTCONFIG_FILE=str(fontconf)));page=b.new_page(viewport={'width':1200,'height':1600});page.goto((B/'index.html').as_uri());page.locator('img').evaluate_all('async imgs=>{await Promise.all(imgs.map(i=>i.decode()));}');page.evaluate('document.fonts.ready');page.pdf(path=str(B/'copper-seed42-investor-report.pdf'),print_background=True,prefer_css_page_size=True);b.close()
# Preserve portable offline links instead of Chromium's build-host absolute paths.
with pymupdf.open(B/'copper-seed42-investor-report.pdf') as pdf:
 for page in pdf:
  for link in page.get_links():
   if link['kind']==pymupdf.LINK_LAUNCH:
    relative=Path(link['file']).relative_to(B).as_posix()
    assert (B/relative).is_file(),relative
    link['file']=relative;page.update_link(link)
 pdf.saveIncr()
print('PUBLISHED',flush=True)
