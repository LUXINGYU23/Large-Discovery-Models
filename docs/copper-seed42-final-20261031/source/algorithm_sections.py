# -*- coding: utf-8 -*-
"""Source-grounded algorithm pages for the existing seed-42 report."""
import json

CSS = """
.phasebar{display:flex;height:42px;margin:13px 0 6px;border-radius:7px;overflow:hidden;font-size:12px;color:white;text-align:center}
.phasebar span{display:flex;align-items:center;justify-content:center}
.phasebar .search{width:58%;background:#216b8c}.phasebar .polish{width:36%;background:#278e92}.phasebar .reserve{width:6%;background:#a6b4c0;font-size:9px}
.algorithm-figure{margin:15px 0}.algorithm-figure img{display:block;width:100%}.algorithm-figure figcaption{font-size:10px;color:#607386;margin-top:6px}
.pseudocode{white-space:pre-wrap;background:#f2f6f9;border-left:3px solid #216b8c;padding:14px 16px;font:11px/1.85 ReportCJK,"Noto Sans CJK SC",sans-serif;margin:12px 0}
.formula{background:#f2f6f9;border-radius:6px;padding:12px 15px;font-size:12px;margin:10px 0 15px}.compact p{font-size:12px}
.audit-note{font-size:11px;line-height:1.7;background:#f4f6f8;border:1px solid #dce5ed;padding:12px 15px;margin-top:13px}
.identity{font:9px/1.8 monospace;word-break:break-all;color:#607386}
"""

STEPS = [
    ('观测驱动初始化', ['正负观测最近邻距离', '构造初始占据体', '限制在公开可行域内']),
    ('多尺度球形翻转', ['随机可行中心与物理半径', '球内矿体 / 非矿体互换', '局部增、减或改变连通关系']),
    ('约束检查与修复', ['活动区与禁入区', '体积范围、连通体数量', '不合法先修复再评分']),
    ('退火接受 / 拒绝', ['更优提议接受', '较差提议按温度概率接受', '温度随时间线性降低']),
    ('边界双尺度精修', ['从矿体边界随机选点', '比较3×3×3与7×7×7翻转', '按公开损失寻找局部改进']),
    ('输出与独立评分', ['合法性校验后返回几何', '隐藏确认观测独立打分', '报告指标取既有评分收据']),
]


def diagram():
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="510" viewBox="0 0 1100 510">',
         '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#7193a5"/></marker></defs>',
         '<rect width="1100" height="510" rx="12" fill="#f4f8fb"/>',
         '<g font-family="Noto Sans CJK SC,sans-serif">']
    for i, (title, lines) in enumerate(STEPS):
        x, y = 26 + i % 3 * 362, 30 + i // 3 * 262
        s += [f'<rect x="{x}" y="{y}" width="325" height="185" rx="12" fill="white" stroke="#cbdae4" stroke-width="2"/>',
              f'<text x="{x+20}" y="{y+36}" fill="#216b8c" font-size="22" font-weight="bold">{i+1} {title}</text>']
        s += [f'<text x="{x+20}" y="{y+78+j*35}" fill="#425b70" font-size="19">{line}</text>'
              for j, line in enumerate(lines)]
    s += ['</g><g stroke="#7193a5" fill="none" stroke-width="3" marker-end="url(#arrow)">',
          '<path d="M353 123H379"/><path d="M715 123H742"/><path d="M912 218V242H188V282"/>',
          '<path d="M353 385H379"/><path d="M715 385H742"/></g>',
          '<text x="550" y="266" text-anchor="middle" font-family="Noto Sans CJK SC,sans-serif" font-size="17" fill="#607386">步骤2 → 3 → 4反复进行，达到退火截止时间后进入精修</text></svg>']
    return ''.join(s)


def build_algorithm_sections(B, d, table, sheet):
    a = json.loads((B / 'data/algorithm-audit.json').read_text())
    (B / 'assets/ldm42-algorithm.svg').write_text(diagram())
    content = """
<p class="lede">LDM交付的是一份可直接执行的三维几何搜索程序：<b>多尺度球形翻转退火＋边界局部精修</b>。</p>
<p>这个名称是对冻结源码实际执行路径的概括。LDM的产出是搜索算子、接受规则、可行域处理与时间分配的程序组合；最终求解器使用NumPy / SciPy计算，每个案例内不再调用大模型。</p>
<figure class="algorithm-figure"><img src="assets/ldm42-algorithm.svg" alt="冻结LDM42求解器六步流程"><figcaption>流程依据最终赢家源码逐函数整理；精修的内部记录问题见第6页源码复核。</figcaption></figure>
<h3>一次120秒额度怎样分配</h3>
<div class="phasebar"><span class="search">初始化＋退火：至69.6秒</span><span class="polish">边界精修：至112.8秒</span><span class="reserve">余量</span></div>
<p class="small">名义截止点：120×0.58与120×0.94；初始化计入前段。检查发生在循环边界，实际求解收据约113.3秒，不能把这些比例当成逐阶段实测耗时。</p>
<div class="twocol compact"><div><h3>外层：研究算法</h3><p>依据训练反馈修改Python程序，产生不同搜索结构和参数组合；验证阶段从3份冻结候选中选择最终程序。</p></div><div><h3>内层：反演一个案例</h3><p>最终程序读入观测与公开约束，反复改变三维占据体并评分，在固定时间内返回一个合法几何。</p></div></div>
"""
    sheet('03 / LDM具体设计了什么算法', content, 'THE DISCOVERED SOLVER')

    content = '<p>算法直接优化矿体占据体 M = [f &gt; 0]。评分前将几何转成规范距离场，使用公开观测损失；隐藏真值和确认观测只用于外部评估。</p><h3>最终赢家的有效参数</h3>'
    content += table(['环节', '源码中的实际设置', '作用'], [
        ['初始化', '公开正负观测最近邻；通常 d<sub>+</sub> &lt; d<sub>−</sub>', '从观测构造初始矿体'],
        ['球形翻转', '尺度集合0.05 / 0.075 / 0.105 / 0.13 / 0.17', '覆盖多种空间尺度'],
        ['物理半径', 'r = max(最小体素边长, D × 尺度 × U[0.35,1))', 'D为计算域物理对角线'],
        ['退火温度', 'T<sub>0</sub> = 0.00003，按退火阶段时间线性降温', '允许少量非改进移动'],
        ['约束修复', '优先保留大连通体，按观测距离排序增减体素', '满足活动区、体积和连通性约束'],
        ['边界精修', '随机边界点；立方邻域半径1与3', '比较局部3³ / 7³体素翻转'],
        ['随机数', 'default_rng(执行种子 + 41)', '内部偏移41与发现种子42不同'],
    ], 'dense')
    content += """
<div class="formula"><b>退火接受规则：</b>公开损失降低则接受；否则以 exp((当前损失 − 新损失) / T(t)) 的概率接受。新状态不合法时，退火阶段先修复再评分；精修阶段跳过不合法提议。</div>
<h3>可复核的流程伪代码</h3>
<pre class="pseudocode">输入：公开观测、网格与约束；时间额度120秒；执行种子
1  用共享观测初始化器构造合法矿体，计算公开损失
2  至总计69.6秒：
     随机选可行中心、尺度和抖动半径，翻转球内体素符号
     评分；若不合法，修复活动区 / 连通体 / 体积后重评
     按改进或退火概率接受，更新搜索状态与退火最优记录
3  用退火阶段最优状态进入冻结的 best_pair 边界精修
     至总计112.8秒，比较边界点周围两种立方翻转
4  对精修输出做合法性校验；必要时回退至退火状态或初始化
5  返回几何，由独立评分器产生报告所用指标</pre>
<p class="small">本赢家启用 sphere / anneal / best_pair；源码中的RBF、循环重启、算子组合搜索、薄片和拓扑专用分支未启用，因此没有将它们列为本次实际算法。best_pair的记录细节见下一页。</p>
"""
    sheet('03 / 最终赢家：算子、参数与伪代码', content, 'OPERATORS & IMPLEMENTATION')

    tg = next(t['training_best'] for t in d['training'] if t['method'] == 'ldm')
    vg = next(g for g in d['validation'] if g['method'] == 'ldm')
    vm = {c['source_sha256']: c['metrics'] for c in vg['validation_candidates']}
    ids = [a['training_best']['source_sha256'], a['selected']['source_sha256']]
    tm = [tg['metrics'], a['selected']['training_metrics']]
    content = '<p class="lede">报告中的“最好算法”，指<b>按既定验证规则选中的种子42最终赢家</b>。训练分数最高的候选也单独保留，两者没有混用。</p>'
    content += table(['比较项', '训练最佳候选', '验证选中／正式测试赢家'], [
        ['源码摘要前缀', ids[0][:12], ids[1][:12]],
        ['训练损失↓'] + [f"{m['loss_total']:.6f}" for m in tm],
        ['验证损失↓'] + [f"{vm[i]['loss_total']:.6f}" for i in ids],
        ['验证IoU↑'] + [f"{vm[i]['mean_iou']:.6f}" for i in ids],
        ['退火截止比例', '0.56（67.2秒）', '0.58（69.6秒）'],
        ['随机种子偏移', '31', '41'],
        ['用途', '训练搜索最佳记录', '本报告全部LDM测试与案例图'],
    ], 'dense')
    content += """
<p>两份程序的函数体经AST核对相同，仅上述退火截止比例和随机偏移不同。最终赢家训练损失略高，但验证损失更低，因此按验证规则选中；验证实际执行种子为47，保留已接受的现有选型。</p>
<h3>这份程序已经交付的效果</h3>
<p>固定赢家在ID / OOD上的损失为<b>0.3478 / 0.3516</b>；相对修复后的MCMC与ACQ降低<b>13.1%–14.7%</b>。它以约<b>22.2秒</b>的额外实际求解时间，取得本次完整程序比较中的几何与观测拟合改善。各组件的单独贡献未作消融测量，不能将全部提升归于某一个算子。</p>
<div class="audit-note"><b>源码复核记录：</b>冻结赢家第321行在 best_pair 精修中将最后一个提议 prop 存入 best，却将选中提议 chosen 的分数记入 best_l。当半径1的提议胜过半径3时，可能出现模型与内部记录分数错配；隔离的单轮单元探针已复现（示例记录0.4、返回模型的桩评分0.8）。这不是正式案例的实测差值。<br>
原程序和当前赢家保持冻结；本次没有补评、修复后重跑或重选。报告数值继续取独立评分器对实际返回模型的既有收据。<a href="algorithm-notes.html">查看源码、参数和复核证据</a>。</div>
"""
    content += '<p class="identity">最终赢家 SHA256：' + ids[1] + '<br>训练最佳 SHA256：' + ids[0] + '</p>'
    sheet('03 / 哪个候选是“最好算法”', content, 'SELECTED WINNER & SOURCE EVIDENCE')

    links = ''.join('<li><a href="' + a[k]['report_path'] + '">' + label +
                    '</a> · SHA256 <code>' + a[k]['source_sha256'] + '</code></li>'
                    for k, label in [('selected', '验证选中的冻结求解器'),
                                     ('training_best', '训练最佳冻结求解器')])
    notes = """
<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LDM42算法与源码复核</title>
<style>body{max-width:1000px;margin:35px auto;padding:20px;font:16px/1.8 system-ui;color:#20364b}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f0f5f8;padding:20px}a{color:#176e91}</style>
<h1>LDM42：算法、候选身份与源码复核</h1>
<p><a href="index.html#page4">返回报告算法章节</a> · <a href="data/algorithm-audit.json">结构化审计证据</a> · <a href="source/audit_selected_algorithm.py">只读源码复核与隔离探针脚本</a></p>
<h2>冻结源码</h2><ul>LINKS</ul>
<p>复核只读正式实验目录，复制源码原字节供审阅；report.json摘要前后一致，没有调用solve()、Docker、评估服务或大模型API。</p>
<h2>实际执行路径</h2>
<p>initialize → sphere / anneal → best_pair → canonicalize。RBF、cycle、portfolio、slab、topology、ellipsoid分支未启用。共享初始化器从公开正负观测计算最近邻距离；两个类别都有时以d+ &lt; d−构造占据体，再满足公开约束。退火损失为公开观测NLL与接触点Huber项，按组与模态权重聚合；不使用隐藏确认观测或真值。</p>
<h2>最佳记录的实现问题</h2>
<pre>if chosen is not None:
    field, cur = chosen, chosen_val
    if cur &lt; best_l:
        best, best_l = prop.copy(), cur  # 冻结源码第321行</pre>
<p>循环比较半径1、3两种提议。chosen可能是半径1的模型，prop始终指最后一次构造的半径3模型。因此此行的模型与分数可能不对应。隔离探针直接提取原函数，使用合成数组、固定时钟和桩评分，只执行一轮：半径1分数0.4，半径3分数0.8，函数返回半径3模型而内部记录0.4。该探针证明控制流问题，不测量真实案例受影响的频次或性能。</p>
<p>本报告保留原赢家与正式收据。报告中的损失、IoU及时间来自外部评分对实际输出的评估，不采用该内部best_l作为测试结果。此处没有修复或重新计算科学结果；原实验效应仍按冻结程序整体解释。</p>
<h2>选型口径</h2>
<p>训练最佳776ee0300e97…与验证赢家2d2c33ed6a58…的函数体一致，仅anneal_frac与seed_offset不同。验证仅使用执行47；不补评、不重选。“最好”指既定验证选型规则下的当前赢家。</p></html>
""".replace('LINKS', links)
    (B / 'algorithm-notes.html').write_text(notes)
    return """
最终选中算法可概括为“多尺度球形翻转退火＋边界局部精修”：公开观测最近邻初始化，物理尺度球内符号翻转，线性降温接受规则，约束修复，再做边界3³ / 7³邻域搜索。名义退火截止69.6秒，工作截止112.8秒，实际约113.3秒。

![算法流程](assets/ldm42-algorithm.svg)

训练最佳为776ee0300e97…（训练损失0.339720，验证0.371223）；正式测试采用验证赢家2d2c33ed6a58…（训练0.343746，验证0.364064）。两者仅退火时间比例和随机偏移不同。

[算法详情与源码复核](algorithm-notes.html) · [原始赢家源码](source/ldm42-selected-solver.py) · [结构化审计](data/algorithm-audit.json)。复核发现精修best记录可能误存最后提议，已披露；不补评、不重选，结果取独立评分收据。

"""
