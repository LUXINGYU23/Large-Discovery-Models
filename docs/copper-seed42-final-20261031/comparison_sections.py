# -*- coding: utf-8 -*-
"""Frozen baseline algorithms and source-grounded LDM improvement trajectory."""
import html
import json

CSS = """
.trace-table{font-size:10px;line-height:1.5}.trace-table th,.trace-table td{padding:6px}
.trace-table td:nth-child(2){width:46%}.trace-table td:nth-child(3){white-space:nowrap}
.trace-figure{margin:10px 0}.trace-figure img{width:100%;display:block}
.trace-figure figcaption{font-size:10px;line-height:1.6;color:#607386;margin-top:4px}
"""


def build_comparison_sections(B, d, table, sheet):
    t = json.loads((B/'data/method-trajectory.json').read_text())
    content = '<p class="lede">三种方法优化同一公开观测目标；区别在于<b>研究阶段改变什么</b>，以及<b>求解阶段怎样改变三维几何</b>。</p>'
    content += table(['比较维度','LDM最终赢家','修复后MCMC','修复后ACQ'],[
        ['外层研究','生成／改写完整Python算法','固定算法，Sobol扫描参数','固定算法，Sobol扫描参数'],
        ['选中程序的起点','公开正负观测最近邻初始化','投影后的随机场初始化','投影后的随机场初始化'],
        ['搜索状态','以符号表示矿体占据体','保留亚体素界面的连续距离场','与MCMC相同'],
        ['几何提议','多尺度球内符号翻转','高斯随机场速度＋界面传播','同类随机场提议，每轮最多4个'],
        ['提议选择','逐个公开评分；精修比较两邻域','每轮1个提议直接评分','代理预测均值／不确定性，选1个评分'],
        ['接受准则','损失差＋随时间降低的温度','平方损失差／固定温度','与MCMC相同'],
        ['公开硬约束','非法球形提议修复后再评','投影后保留连续场与符号','与MCMC相同'],
        ['边界专用精修','有：3³与7³立方翻转','无独立精修阶段','无独立精修阶段'],
        ['本次实际测试秒','约113.3秒／案例','约91.1秒／案例','约91.1秒／案例'],
    ],'dense')
    content += """
<div class="method"><h3>LDM改变算法本身</h3><p>从只返回初始化，发展出球形搜索、多尺度选择、低温退火、快速合法评分和边界精修。最终交付可独立运行的求解程序。</p></div>
<div class="method orange"><h3>MCMC提供随机场几何搜索对照</h3><p>在固定程序内调整温度、步长、相关尺度与初始化方式。每次通过随机速度场移动界面，公开评分后接受或拒绝。</p></div>
<div class="method purple"><h3>ACQ在同一随机场搜索上增加提议筛选</h3><p>用已评分历史拟合在线代理，先筛选更有希望的提议再调用真实目标。这里的采集选择发生在单案例内部；外层仍是预设Sobol调参。</p></div>
"""
    sheet('03 / 三种算法逐项比较',content,'ALGORITHM COMPARISON')

    mc=t['baselines']['mcmc']['constants']['CONFIG'];ac=t['baselines']['acq']['constants']['CONFIG']
    content='<h3>修复后MCMC：随机场提议与固定温度接受</h3><p>随机速度场经快速行进法扩展到界面附近，再按物理步长移动距离场；投影到公开硬约束可行域，保留亚体素变化。真实评分后更新当前状态，并单独保存最低公开损失的几何。</p>'
    content+='<div class="formula">φ′ = φ − U(0, step_cells) × 最小体素边长 × 扩展速度 / 最大绝对速度<br>接受概率 = min(1, exp((当前公开损失² − 新公开损失²) / temperature))</div>'
    content+='<h3>ACQ：代理辅助的提议挑选</h3><p>先积累4个已评分提议作为暖启动；之后每轮最多生成4个合法提议。在线贝叶斯岭回归拟合 log(1＋公开损失)，输入包括随机场参数、体积分数、质心、空间展布、表面与场值统计及其平方项。代理只决定下一次评哪个提议，实际接受与最佳状态仍依据真实公开评分。</p>'
    content+='<div class="formula">选择提议 j* = argmin［预测均值 μⱼ − β × 预测标准差 σⱼ］<br>选中1个提议 → 真实评分 → 更新代理 → 按与MCMC相同的规则接受／拒绝</div>'
    content+='<h3>本次验证选中的实际参数</h3>'+table(['参数','MCMC赢家','ACQ赢家'],[
        ['temperature',f"{mc['temperature']:.6f}",f"{ac['temperature']:.6f}"],
        ['step_cells',f"{mc['step_cells']:.6f}",f"{ac['step_cells']:.6f}"],
        ['range_scale',f"{mc['range_scale']:.6f}",f"{ac['range_scale']:.6f}"],
        ['初始化','random_field','random_field'],
        ['实际每轮提议数','1','暖启动后最多4'],
        ['暖启动／β','采集分支不启用',f"{ac['warmup']} ／ {ac['beta']:.6f}"],
        ['工作截止','solve入口剩余额度的75%','solve入口剩余额度的75%'],
    ],'dense')
    content+='<p class="small">两份基线赢家的函数体经AST核对相同，CONFIG中的method与β决定采集分支差异。保留硬约束投影后的优化程序身份；不将其解释为严格后验采样。</p>'
    content+='<p class="small">原始冻结源码：<a href="source/mcmc42-selected-solver.py">MCMC '+t['baselines']['mcmc']['source_sha256'][:12]+'</a> · <a href="source/acq42-selected-solver.py">ACQ '+t['baselines']['acq']['source_sha256'][:12]+'</a>。完整身份与有效参数见 <a href="data/method-trajectory.json">方法与轨迹数据</a>。</p>'
    sheet('03 / MCMC与ACQ：提议、接受与参数',content,'BASELINE MECHANICS')

    content='<p>参考autoresearch的“淡色候选点＋突出纪录点＋历史最佳阶梯＋改动注释”画法。铜矿实验保留自身批次语义：横坐标仅表示<b>该轮结束时的累计收费槽位</b>。</p>'
    content+='<figure class="trace-figure"><img src="assets/ldm42-annotated-trajectory.png" alt="LDM候选、历史最佳与关键代码改进节点"><figcaption>308份完整候选全部入图；13次收费接口拒绝没有损失值，不补点。A–L为初始化及11次批末新纪录；蓝色星标是验证选中的最终程序，不是训练最低点。</figcaption></figure>'
    content+='<figure class="trace-figure"><img src="assets/ldm42-trajectory-zoom.png" alt="首批之后训练最佳损失的放大轨迹"><figcaption>下图仅放大纵轴，展示后续微小提升的真实幅度。字母与下一页源码溯源表一一对应。</figcaption></figure>'
    content+='<p class="small"><a href="trajectory.html">打开交互轨迹、完整节点说明及源码差异</a> · 图形参考：<a href="'+t['reference']['url']+'">karpathy/autoresearch · analysis.ipynb</a>（固定提交228791fb499a）。纪录之间是相邻最佳源码对照，不冒充直接父子关系或单变量消融。</p>'
    sheet('04 / LDM改进轨迹：从初始化到精修',content,'ANNOTATED DISCOVERY TRAJECTORY')

    for part,subset in enumerate([t['nodes'][:6],t['nodes'][6:]]):
        rows=[]
        for n in subset:
            delta='起点' if n['improvement_absolute'] is None else '−'+f"{n['improvement_absolute']:.6f}"
            rows.append([f"<b>{n['id']} · R{n['round']}</b><br>槽位{n['batch_end_slot']}",
                '<b>'+n['title']+'</b><br>'+n['code_change']+f'<br><a href="{n["source_path"]}">{n["source_sha256"][:10]}</a>',
                f'{n["loss"]:.6f}<br><span class="small">{delta}</span>',f'{n["mean_solver_seconds"]:.1f}'])
        content='<p class="small">损失为该程序48个训练案例的既有确认评分聚合；“减少”对照上一条纪录程序，秒数为该程序的实际训练求解均值。</p>'
        content+=table(['节点／轮次','实际启用的设计变化与源码','训练损失↓<br>较上一纪录','实际秒'],rows,'trace-table')
        if part==0:
            content+='<h3>第一阶段：先建立可持续改进的几何搜索</h3><p>先把观测初始化变成可迭代的球形贪心搜索，再减少重复评分开销，并逐步扩大尺度集合。球内直接翻转符号会改变实际占据体；约束修复使搜索围绕合法几何继续。</p><div class="callout">A→B是从“只返回初始化”到“执行搜索”的变化，实际时间也从约1秒增至91秒。C→F阶段约83秒，重点改动为评分路径、尺度集合与体积修复。</div>'
        else:
            s=t['training_summary']
            content+=f'<div class="callout">初始化程序 {s["baseline_loss"]:.6f} → 训练最佳 {s["final_training_loss"]:.6f}，下降{s["relative_reduction_percent"]:.2f}%。这是完整程序的观测轨迹，包含搜索结构、参数、随机流及实际算时共同变化；R10平均实际秒从约83增至114。</div>'
            content+='<p class="small">R20验证选中程序的训练损失为0.343746、验证损失0.364064；它没有刷新训练纪录，正式测试继续使用该固定赢家。L节点及最终赢家的best_pair记录问题保留第6页披露。</p>'
        sheet('04 / 节点溯源：'+('早期搜索 A–F' if part==0 else '退火与精修 G–L'),content,'SOURCE-VERIFIED MILESTONES')
    write_trajectory_browser(B,t)
    return """
### 三种算法比较与关键改进轨迹

LDM改写完整求解程序；MCMC在固定随机场优化器中调参；ACQ在相同优化器内增加在线贝叶斯岭回归与LCB提议筛选。三种方法的实际初始化、状态、提议、接受、约束与精修逐项对照见完整报告。

![带改进标注的训练轨迹](assets/ldm42-annotated-trajectory.png)

初始化加11次批末新纪录，共A–L十二个溯源节点；308份完整候选全部入图，13次收费接口拒绝不补分数。首批球形搜索、三尺度扩展、低温退火、边界精修及双提议比较均有源码与分数记录。

[交互轨迹与全部节点源码差异](trajectory.html) · [结构化轨迹](data/method-trajectory.json) · [画法参考：autoresearch](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/analysis.ipynb)。

"""


def write_trajectory_browser(B,t):
    (B/'data/method-trajectory.js').write_text('window.TRAJECTORY='+json.dumps(t,ensure_ascii=False,separators=(',',':'))+';')
    cards=[]
    for n in t['nodes']:
        diff=f' · <a href="{n["diff_path"]}">与上一纪录的源码差异</a>' if n['diff_path'] else ''
        cards.append(f'<article id="node{n["id"]}"><h2>{n["id"]} · R{n["round"]} · {n["title"]}</h2>'
            f'<p><b>实际改动：</b>{html.escape(n["code_change"])}</p><p><b>设计作用：</b>{html.escape(n["mechanism"])}</p>'
            f'<p>训练损失 {n["loss"]:.9f}；平均求解 {n["mean_solver_seconds"]:.3f}秒；批末收费{n["batch_end_slot"]}槽。</p>'
            f'<p><a href="{n["source_path"]}">冻结源码</a>{diff} · <code>{n["source_sha256"]}</code></p></article>')
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LDM42改进轨迹与算法对照</title>
<script src="assets/plotly.min.js"></script><script src="data/method-trajectory.js"></script>
<style>body{max-width:1200px;margin:25px auto;padding:20px;background:#f2f6fa;color:#20364b;font:15px/1.8 system-ui}header{background:#173e56;color:white;padding:25px;border-radius:12px}header a{color:#bee5ef}a{color:#176e91}article{background:white;border-left:4px solid #278d62;border-radius:8px;padding:18px 24px;margin:20px 0}h2{font-size:20px}code{overflow-wrap:anywhere;font-size:11px}#plot{height:650px;margin:20px 0}.note{font-size:13px;color:#607386}img{width:100%}label{margin-right:20px}</style>
<header><h1>发现种子42：算法改进轨迹</h1><p>308份完整候选 · 321收费槽位 · 初始化＋11次批末纪录改进</p>
<a href="index.html#page9">返回报告</a> · <a href="data/method-trajectory.json">完整轨迹JSON</a> · <a href="algorithm-notes.html">已有源码复核记录</a></header>
<p class="note">横坐标为批末累计收费槽位；同轮点共享横坐标，不表示并行批次内部顺序。候选悬浮提示含源码、损失和实际秒；A–L的改动可在下方逐项核查。13次收费接口拒绝不虚构分数。</p>
<label><input id="raw" type="checkbox" checked>显示全部候选</label><label><input id="labels" type="checkbox" checked>显示节点标签</label><div id="plot"></div>
<img src="assets/ldm42-annotated-trajectory.png" alt="带关键算法改动标注的静态轨迹">
<p class="note">图形语法参考 <a href="REFERENCE">karpathy/autoresearch · analysis.ipynb</a>。改动描述按实际活动分支核对，未直接照抄候选标题；相邻纪录不声称直接父子关系，分数差不是单个组件的独立效应。R10实际工作时间同时增加；L及最终赢家保留既有best_pair记录问题。</p>
CARDS
<script>
const t=TRAJECTORY,ns=t.nodes,cs=t.candidates;
const traces=[
{type:'scatter',mode:'markers',name:'完整候选',x:cs.map(r=>r.batch_end_slot),y:cs.map(r=>r.metrics.loss_total),text:cs.map(r=>'R'+r.round+' · '+r.source_sha256.slice(0,12)+'<br>'+r.metrics.mean_solver_wall_seconds.toFixed(2)+'秒'),marker:{color:'#b6c0c8',size:7,opacity:.55},hovertemplate:'槽位 %{x}<br>训练损失 %{y:.6f}<br>%{text}<extra></extra>'},
{type:'scatter',mode:'lines',name:'批末历史最佳',x:ns.map(n=>n.batch_end_slot).concat([321]),y:ns.map(n=>n.loss).concat([ns[ns.length-1].loss]),line:{color:'#238c59',shape:'hv',width:3}},
{type:'scatter',mode:'markers+text',name:'源码节点 A-L',x:ns.map(n=>n.batch_end_slot),y:ns.map(n=>n.loss),text:ns.map(n=>n.id),textposition:'bottom center',customdata:ns.map(n=>n.title),marker:{color:'#2ecc71',size:11,line:{color:'#175735',width:1}},hovertemplate:'%{text} · %{customdata}<br>训练损失 %{y:.6f}<extra></extra>'},
{type:'scatter',mode:'markers',name:'验证选中程序',x:[t.selected.batch_end_slot],y:[t.selected.metrics.loss_total],marker:{symbol:'star',size:18,color:'#2563a6'},hovertemplate:'验证赢家<br>训练损失 %{y:.6f}<extra></extra>'}
];
Plotly.newPlot('plot',traces,{margin:{l:75,r:25,t:45,b:70},title:'从候选点到历史最佳：可缩放、悬浮查看源码身份',xaxis:{title:'累计收费槽位（批末）'},yaxis:{title:'训练损失 ↓'},legend:{orientation:'h',y:1.12},paper_bgcolor:'#f2f6fa',plot_bgcolor:'white'},{responsive:true,displaylogo:false});
document.getElementById('raw').onchange=e=>Plotly.restyle('plot',{visible:e.target.checked},[0]);
document.getElementById('labels').onchange=e=>Plotly.restyle('plot',{mode:e.target.checked?'markers+text':'markers'},[2]);
</script></html>""".replace('CARDS',''.join(cards)).replace('REFERENCE',t['reference']['url'])
    (B/'trajectory.html').write_text(page)
