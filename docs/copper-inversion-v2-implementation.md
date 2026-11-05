# Copper v2 实施台账

2026-10-24。状态：进行中，未完成任务切换和正式实验。

## 已实施

- `tasks/copper_inversion/v2/contract.py`：严格公开字段、三维物理网格、mask、开放边界、分组点观测、公开先验；拒绝额外真值字段。
- `geometry.py`：占据场规范化、禁入区拒绝、公开体积/分量硬约束、空集合距离、物理面积/体积/碎片诊断。
- `objective.py`：接触与0/1标签分离；按组均值、模态权重、定位尺度及标签噪声计算目标。
- `data.py`：公开输入序列化及只读数组加载。
- `metrics.py`：独立IoU、按物理面面积加权的表面F1/距离/HD95；不把域截断面当界面。
- `baselines.py`：共享数据初始化和Metropolis开发基线，使用相同公共目标。这不是原论文MCMC的合格复现，正式基线适配仍待完成。
- LDM方案已限定为AST与Qwen embedding等权融合；尚未启动新LDM运行。

## 远程证据

H20 `/Aidata/shared/ldm/copper-v2-development/`，使用已有科学绘图环境运行：

```
/Aidata/shared/ldm/copper-technical-report-20261024/plot-env/bin/python -m unittest tasks.copper_inversion.tests.test_v2_science -v
```

9项通过：禁入区拒绝、幅值不变性、空集合/开放边界、接触项分离、物理单位缩放、组内重复采样、只读与隐藏字段拒绝、空集合指标、共享基线目标。原始输出：[science-tests.txt](../tasks/copper_inversion/resources/evidence/v2-development/science-tests.txt)。

这些是科学核心开发检查，不证明容器隔离、隐藏数据隔离、正式基线复现或完整LDM闭环已通过。

## 尚待完成（目标保持完整）

1. 审计旧mask来源；新pilot地形和可行域独立生成，不沿用未确认语义的mask。
2. 独立母体生成、地质类型和观测协议、谱系去重与质量检查；仅生成pilot用于调试。
3. 接入既有容器执行/只读接口/独立评分、可信真值挂载；补充攻击和变形测试。
4. suite聚合、按母体宏平均、失败计入分母、预算账本与恢复幂等；旧活动路径切换并移除。
5. 融合表征唯一配置、远程Qwen缓存及DeepSeek官方Flash low接入。
6. 原论文MCMC及强确定性基线适配，同120秒资源、同训练调参额度。
7. pilot校准、最终契约冻结后生成正式独立train/val/test并隔离封存。
8. 预注册预算下完整发现/验证/测试、3发现seed与3执行seed、正式报告。

正式隐藏集必须置于研究agent无法访问的账户或服务中；当前开发目录没有正式隐藏数据。不会把本机/远程研究进程可读的目录权限称为真正隔离。

## 第二次实施增量：容器链路与首个开发母体

- 复用 `core.execution.Execution`，加入显式v2科学入口和仅scorer可挂载的真值目录。评分作业禁止挂载候选源码，真值digest计入执行身份；v2代码也计入实现摘要。
- 新增v2 runner/scorer；输入只读检查沿用共享runner的摘要检查。模型仍由宿主冻结、独立容器重新读取和评分。当前core/v2入口是切换期间的显式路径，尚未移除旧活动任务；无自动版本猜测或数据兼容逻辑。
- 远程11项检查通过，包括真值不可见、输入不可写、合法候选独立评分、非法体判零和物理坐标舍入边界。证据：[combined-tests.txt](../tasks/copper_inversion/resources/evidence/v2-development/combined-tests.txt)。
- 独立生成环境：GemPy 2026.0.3、gempy-engine 2026.0.3.post1、NumPy 2.4.6，位置 `/Aidata/shared/ldm/copper-v2-development/gempy-env`。未改变solver镜像或旧实验环境。
- 第一种开发配方为GemPy褶皱层状体；结构约束与地形采用不同随机流。28×32×25各向异性网格；目标占可行岩体18.39%。输出岩性必须通过离散编号检查，避免把GemPy过渡值直接误当分类结果。
- 开发观测为3条斜孔、108个观测（含6个接触），形成钻孔/钻孔＋地表两个包。母体身份和真值保留在私有manifest，不传入公开problem；当前仍位于维护账户可读开发目录，不能称正式隔离。
- `experiments/v2/pilot.py`通过现有容器运行确定性初始化、Metropolis开发基线，各跑两个包；10秒开发预算、2CPU、4GiB，4/4成功。初始IoU分别约0.1011/0.0474，短链未改善。公共损失约0.4993/0.2647，**这是目标拟合与几何准确度可能脱节的证据，不是算法优势结果**。
- 结果：[pilot-first-results.json](../tasks/copper_inversion/resources/evidence/v2-development/pilot-first-results.json)。执行目录 `/Aidata/shared/ldm/copper-v2-development/pilot-runs/second`。

### 尚未满足的科学门槛

目前观测由与评分同分辨率的开发真值重建距离产生，有离散化过于匹配的风险；代码和manifest明确禁止把它当正式套件。正式版本需要独立分辨率的前向采样、噪声/缺失分层、其余三种ID地质族及独立OOD配方。未创建正式测试集，未运行LDM发现，也未开展120秒正式基线比较。

### 旧mask审计线索

上游 `Knowledge-driven-stochastic-modeling/examples/README.md` 将 `ind_active` 定义为地形以下/空气，且示例可视化将非活动体素设为NaN。仍需核验UBC轴向与物理剖面；新GemPy开发数据采用独立生成的地形，不使用旧mask作为真值。

## 第三次实施增量：隐式观测、地质类型与套件协议

- `experiments/v2/observations.py`已替换粗网格SDF采样：通过GemPy私有结构模型直接查询孔轨迹和连续地表，沿孔分类变化处进行16次二分定位接触。评分网格仅保存几何真值，不再生成测量标签。实现定位噪声、标签错误的显式开发选项；噪声分层的大样本验收仍待完成。
- 生成器保存可重建的结构控制点、方向、地形及包版本；新增闭合透镜、双倾斜薄体、GemPy断层组配方。首样检查：闭合透镜1个分量、双薄体2个分量，均不触域边界；褶皱及断层层状体允许侧面延伸。仍需多seed人口级质量门槛、可视化审查和OOD独立配方，不能由四个首样推断完整数据质量。
- 修复GemPy岩性编号偏移：断层元素会使目标层编号从2变成3。目标编号保留在私有结构元数据，未暴露给solver。
- `check_generation.py`远程验证通过：重建22,400体素与保存真值完全一致；103个干净钻孔二值测量、6个接触以及120个地表测量直接匹配隐式模型。接触附近用小物理位移验证跨越真实分类边界。证据：[generation-qualification.log](../tasks/copper_inversion/resources/evidence/v2-development/generation-qualification.log)。这仅检查一个褶皱母体。
- `v2/suite.py`新增严格manifest和共享Execution适配：校验公开数据/真值哈希，每个母体恰好两个观测包，按母体宏平均IoU。solver失败/非法输出作为零IoU保留；独立评分器故障暂停而非处罚候选。
- 外层统一 `loss_total = 1 - mean_iou`，供现有GP/selector最小化；每个case的公开数据目标仍是 `objective_v2`，不得混称同一个loss。结果反馈不包含私有manifest或逐case真值路径。
- 缓存身份包含suite、源码、实现、镜像、预算。测试模拟子case完成后缺失suite最终receipt：恢复得到相同结果，没有新建或改写已完成容器记录。该测试不替代完整Campaign预算/中断回归，后者仍待接入。
- `v2/task_spec.py`固定使用fusion 1280维、AST/Qwen块等权；没有AST-only/Qwen-only选择入口，复用既有编码和LossSelector。
- 远程核心、执行及套件检查合计14项通过，证据：[combined-tests-third.txt](../tasks/copper_inversion/resources/evidence/v2-development/combined-tests-third.txt)。

尚未完成：完整pilot母体集与谱系去重；正式隐藏账户/服务隔离；只读候选fixture扩展；主workflow切换与旧路径移除；融合LDM实际闭环；原论文基线适配；正式协议冻结及统计实验。目标继续保持进行中。

## 第四次实施增量：只读准入、公开研究资源与融合工作流

- `v2/fixtures.py`提供6个固定公开接口输入，接触数0/1/3/4/7/2、变化shape、非零origin、各向异性spacing、千倍单位尺度及封闭边界。`v2/interface_checks.py`复用隔离Execution，控制器冻结模型并检查合法性；`CheckedDomain`在共享源码准入后执行该门槛。
- 远程候选准入测试通过：初始化基线通过全部6例；全正非法候选被拒绝；重复检查复用已有容器receipt。
- `v2/research.py`仅导出公开训练problem、固定测试与公开科学模块，不导出truth、私有manifest、母体分组或独立几何评分模块。远程白名单导出测试通过。
- v2研究说明明确：一个源码处理所有case；公共数据目标和外层 `1-mean_iou` 分离；禁止把缺测当负类或硬编码矿体。复用既有Pi工具、响应校验和研究树，仅替换任务资源。
- `v2/workflow.py`已接入共享Campaign、LDMEngine、预算账本、研究树、融合编码及LossSelector。正式默认81个套件评估槽位（初始＋20×4），仅fusion，没有其他编码CLI入口；官方DeepSeek Flash/low固定。正式阶段仍由未完成的隔离和协议门槛阻断，不能误启动正式实验。
- `build_suite.py`可冻结开发观测包为有哈希的suite，拒绝重复母体/案例。它不是跨split谱系审计或正式发布工具。

### 首次实际融合开发闭环（运行中，结果待核验）

已冻结远程代码到 `/Aidata/shared/ldm/copper-v2-development/source-pilot-r1`，启动独立运行 `/Aidata/shared/ldm/copper-v2-development/runs/fusion-pilot-r1`。启动记录 `fusion-pilot-r1-launch.json`，PID 3872222；进程启动33秒时确认仍存活。后续必须以进程身份、运行记录和结果为准，不根据此台账判断仍在运行。

配置：1个开发母体/2观测包、1轮研究、每研究分支1候选、选择2个候选，加初始算法共3个suite评估；每case为10秒开发预算。缓存Qwen3-Embedding-0.6B位于 `/Aidata/shared/codebo/models/Qwen3-Embedding-0.6B-97b0c614`，使用GPU0。复用现有LDM的四研究分支，不额外开表征消融。密钥从受保护远程存储通过stdin传输到控制器环境，未写入本地或仓库。

这轮仅用于证明真实API和框架链路，不是正式120秒效果比较。未完成的阶段与科学门槛保持不变。

首轮启动后的已核实进展：官方API只读工具预检成功，响应模型为deepseek-flash、thinking=low；初始候选完成6个接口作业和两个case的solver/scorer作业，整个suite形成一条外层评估记录。研究阶段仍在进行，尚无首轮最终结果。证据：[fusion-preflight.json](../tasks/copper_inversion/resources/evidence/v2-development/fusion-preflight.json)、[interface-tests.txt](../tasks/copper_inversion/resources/evidence/v2-development/interface-tests.txt)、[research-tests.txt](../tasks/copper_inversion/resources/evidence/v2-development/research-tests.txt)。

## 第五次实施增量：跨划分审计、均衡pilot与发现额度修正

- `split_audit.py`检查跨split的母体ID、祖先谱系、真值哈希和标准化域内旋转/镜像粗几何指纹碰撞；同时要求显式谱系注册表。它不能证明任意连续变换或所有近重复均已排除，近邻检索/地质审核仍待补充。
- suite读取新增同一母体两观测包的真值、物理网格和mask一致性检查，防止把不同几何错误并成一个统计样本。
- 远程5项split/suite回归通过。新增共享 `ScientificSlots`：接口拒绝的有效源码与被选中进行suite评估的源码共同消耗 `algorithm_feedback_slots`，相同源码恢复不重复扣费。正式默认上限81；接口修复后源码改变会使用新槽位。远程6项预算/准入/suite回归通过。原有运行快照未修改，因此首轮开发运行不能当作新计费契约的验收证据。

### ID pilot 数据

首版 `datasets/pilot-id-r1`已生成，但检查发现验证集全噪声、测试集全干净，造成噪声分布混杂。因此保留为开发历史，不用于正式比较。

修正版 `datasets/pilot-id-r2`使用全新预分配母体种子：train 8 / validation 4 / test-ID 4，共16母体、32观测包；每个split均为50%干净、50%噪声。四类地质在每个split均出现；质量剔除与最多3次预分配替换在方法运行前声明。所有16个母体生成成功，跨split审计通过。测试标签仅是pilot分组名，这些数据全部是可检查的开发集，不是真正封存的正式测试数据。

证据：[split audit](../tasks/copper_inversion/resources/evidence/v2-development/pilot-id-r2-split-audit.json)、[三维地质QA](../tasks/copper_inversion/resources/evidence/v2-development/pilot-id-r2-geology.png)。已人工查看全部16个三维预览：闭合透镜、多体、开放层状体与断层位移均可辨认，未把开放域面补成闭合面。此检查不替代地质专家审查。

三维审核发现生成器仍有固定坐标轴偏好：断层法向固定x，双薄体按x分列。开发配方已更新为随机走向、随机整体方位；新增断层和双体首样在远程生成成功。该改动尚未生成整套新pilot，不回写r2数据或更改其原始摘要。

### 实际LDM开发运行

PID 3872222在运行17分钟时仍存在；研究分支已产生候选源码快照，但当时suite完成记录仍只有初始候选。继续以进程/子进程和结果检查判断进展，不把快照视为准入或正式评分成功。原始快照 `source-pilot-r1` 保持冻结。

仍缺：OOD配方/4个pilot OOD母体，完整Campaign额度回归、真实首轮最终验收、正式隔离与数据冻结、经典基线适配及后续正式实验。

## 第六次实施增量：修复候选准入等待，完成融合诊断回放

首轮真实运行r1未完成：控制器将有效提交扣留到四个研究分支全部到齐；部分分支超时终止后，恢复中的有效分支继续等待已失败的同伴，造成无法推进。已改为每个分支独立校验、持久化准入结果；仍在整批研究结束后才发布候选池交给LDM，继续检查跨分支源码重复。远程研究适配器9项回归通过，含部分完成后的恢复及重复提交。证据：[admission-regression.log](../tasks/copper_inversion/resources/evidence/v2-development/admission-regression.log)。

r1已按启动PID身份核验后显式停止；冻结源码没有修改，没有把它伪装成修复后的成功运行。停止记录：[fusion-pilot-r1-stopped.json](../tasks/copper_inversion/resources/evidence/v2-development/fusion-pilot-r1-stopped.json)。

将r1实际生成的四份源码在新的开发运行中回放：经过6例只读接口检查、AST/Qwen等权融合编码、既有选择器及独立suite评分。初始算法加两个选中候选全部合法，结果如下。

| 算法 | 母体宏平均IoU | 外层1−IoU | 每case平均求解墙钟秒 |
|---|---:|---:|---:|
| 固定公开初始化 | 0.069821 | 0.930179 | 0.522 |
| 回放候选6a3bf3ae | 0.411029 | 0.588971 | 0.786 |
| 回放候选f65abb69 | 0.415216 | 0.584784 | 0.558 |

证据：[replay-admission-r2.json](../tasks/copper_inversion/resources/evidence/v2-development/replay-admission-r2.json)。该结果仅使用1个开发母体的2个观测包、10秒预算；不是独立泛化比较，也不证明优于MCMC。源码回放不能代替真实API研究回合验收。

修复版真实API运行已用全新身份启动：`runs/fusion-pilot-r2`，源码`source-fusion-pilot-r2`，PID 4067547（启动时记录）。仍固定官方DeepSeek Flash/low与融合表征，1轮、4个研究分支、2个选中候选。是否完成必须读取实际收据，不以启动记录判断。

随机走向修正后的完整ID开发集开始生成，独立新种子2026102430，输出`datasets/pilot-id-r3`、冻结生成源码`source-dataset-r3`。此前r2数据保留，不修改其摘要。正式数据、OOD与泛化结论仍未完成。

## 第七次实施增量：补齐开发OOD与基础设施故障计费

- `pilot-id-r3`已完成16母体/32包：8训练、4验证、4 ID测试。`pilot-ood-r1`另生成4个独立母体/8包，其中2个断层闭合透镜组合、2个观测覆盖偏移；每组一干净、一噪声。都属于开发数据，未创建正式测试。
- 结构OOD使用GemPy断层关系作用于闭合透镜；覆盖OOD仅改变孔口和露头采样区域，窗口由独立随机流生成，不依据真值位置选择。没有声称跨生成器泛化。
- 联合20母体谱系/哈希/粗几何碰撞审核通过：[联合审计](../tasks/copper_inversion/resources/evidence/v2-development/pilot-ood-r1-split-audit.json)。人工查看了全部[ID三维预览](../tasks/copper_inversion/resources/evidence/v2-development/pilot-id-r3-geology.png)与[OOD三维预览](../tasks/copper_inversion/resources/evidence/v2-development/pilot-ood-r1-geology.png)：闭合体没有接触外域边界，双体可分辨，断层透镜可见形变。开放层状体保持域面开放。仍不能替代专业地质审核或真实数据验证。
- 新增近重复排序：对144对跨split母体做24³域归一化，并在48种轴置换/镜像下最大化Dice；最高0.8268。此数值不作为独立性证明或事后剔除阈值；任意连续旋转、平移和变形对齐仍未覆盖。输出保留全部配对供审核，不因算法结果删除样本。
- 接口宿主I/O错误改为`EvaluationPaused`，不计科学拒绝槽位；候选缺失/不可读输出、非法模型仍拒绝并计费。远程4项接口/预算检查通过：[interface-infra-tests.log](../tasks/copper_inversion/resources/evidence/v2-development/interface-infra-tests.log)。真实r2冻结源码不回写，下一运行使用新身份。

剩余门槛：真实r2首轮完成、在完整pilot上做公平小预算端到端比较、原方法基线与强确定性基线适配、几何/误差/先验校准、正式隔离与新数据冻结，以及最终训练/选型/测试实验。当前开发改进不能发布为LDM对蒙特卡洛的泛化优势。

## 第八次实施增量：真实融合闭环完成，基线开始适配

`fusion-pilot-r2`真实API运行已终止且产出3条完整suite评估收据；4个研究分支均有独立持久化准入记录。融合编码选择2个新程序，最佳母体宏IoU为0.085518（初始0.069821），另外一个0.085438。全部合法；每case实际求解墙钟均约0.53～0.59秒，硬上限10秒。它比诊断回放的0.415差，两次不能择优合并。仍仅1个开发母体，不能支持泛化优势。

证据：[真实运行结果](../tasks/copper_inversion/resources/evidence/v2-development/fusion-pilot-r2-result.json)、[预算账本](../tasks/copper_inversion/resources/evidence/v2-development/fusion-pilot-r2-budget.json)。账本记录3个科学反馈槽位、30个接口作业、6个solver及6个scorer作业；65次LLM请求，研究墙钟累计约454秒。未把研究成本藏进求解耗时。

基线适配新增 `stochastic_solver.py`：复用包内可复现GSTools高斯场、特征函数和在线Bayesian Ridge，保留论文平方损失接受式；改用v2公开目标、物理spacing、禁入区约束及合法最优状态。物理步长使用最大绝对速度归一化；空/满状态新增显式birth提案；不存整链避免无关内存开销。这些均是公开的适配差异，不声称原配置精确复现或后验正确性。

新增 `deterministic_solver.py`：距离场/接触RBF初始化、公开目标选择平滑及阈值、确定性边界局部改进。RBF遇到点数不足或秩亏时按固定规则回退距离场，记录回退。各方法都经过同一固定接口和SuiteEvaluator，未查询隐藏指标指导内部求解。

第一次MCMC/ACQ准入因5秒接口总预算超时而失败：包导入约1.7秒未被求解器内部计时覆盖。修正版从模块加载时计时并预留余量，在新运行身份重新验收；原失败收据保留，不删除或补记为通过。这些尚未调参、未完成完整pilot比较。

修正版单母体验收已完成：初始化、MCMC-v2、ACQ-v2、距离局部优化、接触RBF五个程序全部通过6个固定接口，并完成各2个观测包的独立评分，合法率100%。平均IoU分别0.069821、0.069834、0.069834、0.064165、0.065311。MCMC每case记录9次公开目标调用，ACQ为6次且实际执行1次采集选择；10秒预算内实际求解约8.25～8.46秒。该微型开发例不能说明方法排名；尤其公开损失下降未必改善隐藏几何。[结果证据](../tasks/copper_inversion/resources/evidence/v2-development/baseline-adaptation-r2-results.json)。

完整训练pilot已分别冻结启动：`runs/fusion-suite-pilot-r1`（启动PID4167870）、`runs/baseline-suite-pilot-r1`。共同使用`pilot-id-r3/suites/train`的8母体/16包、seed47、每case10秒、相同solver镜像与资源。LDM有5个科学反馈槽位，1轮8候选选4；基线为5个事先固定程序各评一次。此阶段仅验证端到端数据与资源合同，并不是“每个基线已获得与LDM相同调参额度”的正式比较。正式各方法81槽位调参、三发现重复仍待实施。

## 第九次实施增量：注册入口切换、故障恢复与数值校准

- 注册入口`ldm_task/procedure.py`改为v2工作流，提供基于suite摘要的`describe_ldm_task`。旧19个运行配置归档至`docs/archive/copper-v1/registration-configs.tar.gz`后移除，当前配置只有`fusion_pilot.yaml`。旧case-id、mock、AST-only/Qwen-only开关均由新入口拒绝。旧科研报告和已冻结运行保留。
- `experiment.json`改为v2指标与pilot合同，依赖检查始终要求Qwen融合依赖和suite manifest。资格证据重置为`registered`，不沿用v1的contract_verified。远程共享注册验证通过，公共runner配置dry-run通过，3项入口/研究导出测试通过。旧核心工作流与相关历史测试的清理仍待完成，当前注册入口已不再调用它们。
- Suite模型冻结遇到宿主I/O故障时暂停，不写入候选失败收据；独立评分器缺失/不可读冻结模型也作为基础设施错误。新增真实容器测试先注入磁盘错误，再恢复，证明只计1个科学反馈槽位且完整评分不遗漏失败case。远程5项套件/执行/预算检查通过。
- 解析球体校准：spacing从4→2→1时，体积相对误差14.44%→5.42%→0.84%，接触距离RMSE 1.080→0.214→0.124物理单位。各向异性spacing(1,1.5,2)的接触RMSE为0.180。千倍单位变换后公开目标差异≤1.2e−16。
- 校准同时证实轴向体素面积的系统偏差：球体在较细网格的面积约为解析表面积的1.51倍，不能称收敛到欧氏面积。新增斜平板解析截面归一化，检查角度依赖。这个结果是继续校准弱正则的依据，不能掩盖或据此立即宣布最终科学合同冻结。

证据：[注册检查](../tasks/copper_inversion/resources/evidence/v2-development/v2-registration.json)、[共享runner dry-run](../tasks/copper_inversion/resources/evidence/v2-development/v2-shared-runner-dry-run.json)、[入口测试](../tasks/copper_inversion/resources/evidence/v2-development/v2-entry-tests.log)、[故障恢复测试](../tasks/copper_inversion/resources/evidence/v2-development/suite-infra-recovery.log)、[几何校准](../tasks/copper_inversion/resources/evidence/v2-development/geometry-calibration-r2.json)。

完整pilot进程在本次检查时均存活（LDM PID4167870，基线PID4189545）；基线已有初始化与MCMC两个完整16包结果，平均IoU分别0.204182与0.204197，合法率100%。其余结果未完成，不能据此评价LDM优劣。运行源码保持冻结，以上修复用于下一运行身份。

## 第十次实施增量：冻结选型与一次性测试协议

新增`v2/selection.py`与受信控制器CLI `experiments/v2/frozen_eval.py`，复用既有源码封存、只读接口和SuiteEvaluator：

- 验证前一次冻结1～3个源码、全部验证/测试manifest摘要、科学实现、不可变镜像摘要、预算和排序规则；无追加候选或重置测试接口。
- `frozen → selected → tested`顺序执行，禁止先测试后选型。排序为宏IoU、合法率、求解耗时、源码哈希；测试ID/OOD都只使用选出的同一源码。
- 主机文件锁串行化协议变更；科学身份改变拒绝续跑。重复调用已完成阶段只读取经过摘要核验的结果。中断中的子case复用既有套件收据。
- 仅接受开发manifest。它没有冒充独立权限服务；正式数据发布、训练候选来源审计、三发现重复和真正隐藏账户/服务仍待完成。

远程状态机2项测试通过，覆盖三候选选一、测试前置条件、结果重放、源文件变更及不可变镜像要求。另用固定初始化源码完成真实开发验收：验证4母体/8包、ID4母体/8包、OOD4母体/8包；三阶段均100%合法。总计66个接口/solver/scorer作业。再次调用验证和测试，收据数、mtime和内容哈希全部不变，新增作业0。

证据：[状态机测试](../tasks/copper_inversion/resources/evidence/v2-development/selection-tests-r2.log)、[真实重放核验](../tasks/copper_inversion/resources/evidence/v2-development/selection-integration-r1-replay.json)、[固定初始化开发测试结果](../tasks/copper_inversion/resources/evidence/v2-development/selection-integration-r1-test-result.json)。该测试没有比较LDM，开发测试数据已经可见，不能转为正式未见数据。

### 完整训练pilot的LDM结果已完成

`fusion-suite-pilot-r1`进程已退出并写出5条完整套件结果；8个母体/16观测包上，初始宏IoU0.204182，四个选中候选分别为0.267945、0.216412、0.215158、0.151987，均100%合法。最佳程序`f6a9935c48ad…`每case平均求解墙钟0.538秒，预算上限10秒。账本为5个科学反馈槽位、54个接口、80个solver、80个scorer作业；75次LLM请求，研究墙钟累计491.35秒。不能将最快求解时间等同于算法发现总成本。

证据：[完整训练结果](../tasks/copper_inversion/resources/evidence/v2-development/fusion-suite-pilot-r1-result.json)、[完整训练预算](../tasks/copper_inversion/resources/evidence/v2-development/fusion-suite-pilot-r1-budget.json)。这是训练集单次发现的开发结果，基线全部完成、验证选型、独立正式测试和公平调参重复仍未完成；不据此宣称泛化优势。

## 第十一次实施增量：候选来源审计与统一开发选型

完整基线训练pilot已完成，5个程序在8母体/16观测包全部合法：初始化0.204182、MCMC-v2 0.204197、ACQ-v2 0.203536、距离局部优化0.186796、接触RBF0.190370（母体宏IoU）。固定基线未获得与LDM相同的调参次数，因此当前0.267945对0.204197的差距不构成正式优势结论。[基线训练结果](../tasks/copper_inversion/resources/evidence/v2-development/baseline-suite-pilot-r1-results.json)。

新增`training_receipts.py`审计训练来源：检查候选源码哈希、完整计划case/seed集合、逐case收据与汇总一致性，并从全部case重新计算聚合指标。冻结协议现在强制提供训练来源；未经完整训练suite评估的新源码不能进入验证。冻结时同时检查训练与所有保留split之间母体及真值哈希不重合，后续阶段核验原训练证据摘要。此为受信控制器内一致性审计，不是抵御拥有宿主权限者的密码学证明。

远程3项协议测试通过，覆盖无训练来源的新源码和缺失训练case被拒绝。实际两个已完成训练run各5个候选全部通过审计。[测试](../tasks/copper_inversion/resources/evidence/v2-development/selection-provenance-tests.log)、[真实训练收据审计](../tasks/copper_inversion/resources/evidence/v2-development/pilot-training-audit.json)。

已启动`runs/pilot-selection-r2`，冻结源码`source-pilot-selection-r2`，启动PID214494。LDM按训练规则取前三名；每个固定基线保持独立方法、各自一个源码，不把不同基线混成一个可按case择优的组合。全部6种方法先冻结，再依次完成全部验证和全部测试；继续每case10秒开发预算。正式隐藏服务、同额调参和三发现重复仍未完成。

## 第十二次实施增量：经典方法独立调参额度

新增`experiments/v2/tune_stochastic.py`，使用SciPy scrambled Sobol生成预注册参数表；默认配置先评估，然后搜索温度、物理步长、相关尺度、初始化；ACQ额外搜索batch/warmup/beta。它是参数搜索，不冒称LDM或代码BO。沿用`ScientificSlots`、只读准入与`SuiteEvaluator`，拒绝源码和完整套件反馈共用最多81槽位，验证/测试manifest不可用于调参。

计划锁定种子、参数表、源码模板、科学实现、镜像、数据和预算；恢复只接受相同计划，已完成候选收据复用。远程5项测试通过，含81个配置唯一性/种子复现/范围约束、超过81拒绝，以及第二候选中断后恢复：前一收据不变、第二候选不重复扣科学槽位、完成后再恢复不调用求解器。[测试证据](../tasks/copper_inversion/resources/evidence/v2-development/stochastic-tuning-tests-r2.log)。这是调参状态机测试，完整容器恢复由独立suite测试覆盖。

已启动两个开发调参run：`mcmc_v2-tuning-pilot-r1`（PID261664）、`acq_v2-tuning-pilot-r1`（PID261665），共同冻结源码`source-tuning-pilot-r1`。各5个训练槽位、8母体/16包、seed47、10秒/2CPU/4GiB，与当前LDM开发pilot反馈上限相同。新增随机场初始化只依赖公开网格和随机种子；先前固定基线源码及成绩不回写。这仍不是正式81槽位、120秒、三发现种子的比较。

检查时三个进程均存活。原选型运行已完成LDM验证：固定前三名的验证IoU分别0.397591、0.317555、0.317933，选中训练第一名`f6a9935c48ad…`；MCMC默认验证IoU0.314343。测试尚未全部完成，不将这些验证分数当独立测试结果。

## 第十三次实施增量：独立评估账户的文件权限边界

核验H20现有账户后，建立专用系统账户`ldm-copper-eval`（uid991，无交互登录shell），私有工作目录`/Aidata/shared/ldm/copper-v2-evaluator`。父目录由root持有、评估组可遍历；datasets/runs/inbox为评估账户0700；code目录由root持有，评估账户不能改写。没有授予Docker组或sudo权限，没有创建网络服务，也没有发布正式数据。

实际访问探针通过：评估账户可读私有canary，普通`zsgpu`进程直接读取被拒绝。[访问审计](../tasks/copper_inversion/resources/evidence/v2-development/evaluator-account-audit.json)。`zsgpu`本身是有sudo权限的受信维护账户，仍能通过管理员权限绕过文件权限；此边界不能抵御管理员，研究guest不得获得宿主凭证或sudo能力。

这仅完成正式隔离所需的一层基础设施；受限评估请求、受信容器调度、数据发布权限及从研究guest发起的端到端越权探针仍待实现和验收。当前三个开发运行继续使用原冻结工作区，不迁移途中状态或重标为正式隔离运行。

## 第十四次实施增量：独立账户的受控训练评分服务

新增`v2/service.py`：本地Unix socket仅接受固定`info`和`evaluate(source)`结构。数据集、运行目录、镜像、预算及允许的宿主UID在受信配置中固定；不接受路径、case选择、shell命令或测试集请求。返回仅含聚合指标和源码摘要；内部错误留在私有日志，不透传私有路径。每个新有效源码在接口反馈前占用同一科学反馈槽位，避免耗尽额度后继续免费探测合法性。

H20部署开发实例：root持有源码`/Aidata/shared/ldm/copper-v2-evaluator/code/service-r1`，私有数据为独立复制的开发母体，socket为`/Aidata/shared/ldm/copper-v2-rpc-r1/evaluator.sock`。允许调用UID1001；服务UID991，PID489722（外层runuser PID489721）。为受信评估服务授予Docker组权限以复用既有Execution，**Docker组等价于宿主管理能力**；候选进程没有该组、宿主凭证或Docker socket。该可信边界与研究guest权限必须分开审计。

首次启动因旧Python解释器位于研究账户私有home而被拒绝，进程已终止且未执行候选。随后建立root持有的独立CPython3.11.16环境，科学依赖固定numpy1.26.4/scipy1.14.1/scikit-fmm2025.6.23/pyyaml6.0.2，再从相同冻结源码启动。没有放宽研究账户home权限。

远程2项协议单测及真实socket探针通过：合法源码独立评分、候选看不到`/truth`/宿主私有canary/Docker socket、相同源码重放结果一致且只占1槽位、非法全正输出拒绝、额外新源码被2槽位上限拒绝。路径覆盖请求拒绝。额外检查证明未授权root peer UID被应用层拒绝、nobody被socket文件权限拒绝（管理员仍可改变身份，不能据此声称抵御root）。

证据：[协议测试](../tasks/copper_inversion/resources/evidence/v2-development/evaluator-service-tests.log)、[真实评分探针](../tasks/copper_inversion/resources/evidence/v2-development/private-service-probe-r1.json)、[peer权限检查](../tasks/copper_inversion/resources/evidence/v2-development/private-service-peer-audit.json)。仍缺研究guest越权探针、LDM客户端对接、正式数据发布与选型服务、正式协议资格；当前服务仅接受开发训练数据，未解除正式运行门槛。

## 第十五次实施增量：评分客户端与协议绑定

新增 `v2/service_client.py`，客户端固定服务UID与科学契约摘要，校验返回源码身份、指标范围、目标恒等式和失败分母；只将通过检查的结果写入EvaluationResult收据。缓存重放核对候选、契约与逻辑评估身份。服务断线、契约不一致和基础设施错误暂停评估，不产生零分或伪成功；预算耗尽显式报错。服务新协议强制携带contract_digest；旧部署service-r1保持冻结，未覆盖运行源码。

远程17项服务/客户端测试通过，包括真实Unix socket peer检查、断线、错误指标、缓存契约篡改及契约拒绝不执行科学作业。[测试证据](../tasks/copper_inversion/resources/evidence/v2-development/service-client-tests-r1.log)。这尚非完整Campaign接入验收；公开训练目录替换私有manifest、接口拒绝结果集成及新服务实例的真实LDM闭环仍待完成。

两条经典调参pilot已核验completion.json：各完成5个候选/5个反馈槽位。MCMC训练最佳母体IoU0.204799，ACQ为0.204897，全部16包合法；融合LDM同额开发训练最佳0.267945。这里只能比较开发训练成绩，尚不能称独立泛化优势。开发选型PID214494检查时仍存活，未重启或修改运行身份。

## 第十六次实施增量：公开训练目录单向发布

新增 `experiments/v2/publish_training.py` 与 `v2/public_catalog.py`。发布操作仅允许train，将通过schema检查的公开problem重新序列化到新目录；清单只含无语义case标识、目录和公开文件摘要，不包含真值路径、母体谱系、隐藏参数或执行种子。已有发布目录拒绝覆盖，文件设为只读。研究侧加载器无需导入私有suite模块，拒绝额外字段、重复case、路径逃逸、文件符号链接及内容摘要变化。

远程8项测试通过，含真实开发数据发布、迁移后加载和拒绝覆盖；发布产物仅为catalog与两包problem文件，共5文件。[证据](../tasks/copper_inversion/resources/evidence/v2-development/public-catalog-tests-r1.log)。已发布开发目录 `/Aidata/shared/ldm/copper-v2-development/public-training-integration-r1`。该目录是开发数据，非正式独立数据。

尚未切换活动workflow：仍需把公开catalog摘要与服务契约绑定，替换PublicTrainingExport的私有manifest输入，并一并更新注册配置/依赖检查和服务拒绝计费。当前独立模块验收不能替代完整研究guest权限与服务闭环验收。

## 第十七次实施增量：公开输入与私有评分绑定、入口切换

活动v2 workflow不再加载私有suite manifest：改用public-catalog与ServiceEvaluator；PublicTrainingExport仅读取公开目录。注册入口和依赖检查同步切换，旧manifest参数移除。服务握手核对固定UID、契约摘要、公开观测的语义摘要、solver时限及反馈上限；语义摘要不依赖NPZ压缩时间戳或目录位置。接口拒绝与已评估候选继续共享控制器ScientificSlots；私有服务有独立上限和完整solver/scorer账本，报告需汇总两侧真实成本。

远程29项测试通过，覆盖公开/私有输入摘要一致、发布导出、固定融合入口及服务契约各字段错配。[测试](../tasks/copper_inversion/resources/evidence/v2-development/public-service-binding-r1.log)。

独立部署service-r2（外层PID693053），冻结源码位于私有账户code/service-r2，复用已审计开发母体，5槽位、10秒。旧service-r1未修改。实际客户端提交与LDM相同的固定初始化源码，完成6个接口及2包独立评分，IoU0.069821、合法率100%；重放返回相同收据且服务仅使用1槽位。[真实客户端证据](../tasks/copper_inversion/resources/evidence/v2-development/service-client-integration-r2.json)。

示例配置改指向这一公开目录和服务身份，仍为开发集成用途。新workflow尚未完成包含真实LLM提案、融合编码、BO选择的全回路验收；服务端接口拒绝与本地准入发生不一致时仍暂停，需补充明确故障分类。正式数据、研究guest攻击测试、正式选型服务与科学校准门槛仍未解除。

## 第十八次实施增量：启动私有服务融合闭环与观测尺度校准

启动器已移除私有manifest参数，改为公开catalog及固定服务身份。远程冻结源码source-fusion-service-pilot-r1，启动PID710173；官方DeepSeek V4.1 Flash/low，Qwen缓存，融合表征，1轮8候选选4＋初始化共5反馈槽位。API key仅通过SSH管道传入进程环境，未保存至本地或源码。服务r2复用同一初始化源码收据，不产生新的科学反馈。检查时进程存活，初始化已成为首个有效Observation，主研究回合完成5次LLM请求；整个搜索尚未完成。[启动身份](../tasks/copper_inversion/resources/evidence/v2-development/fusion-service-pilot-r1-launch.json)。

新数据配方升级为implicit-query-v4：area_weight默认0，保留面积诊断；测量定位噪声（有噪声时0.25×最小spacing，无噪声时0）与0.5×体素对角线的离散尺度分别记录，有效尺度为二者平方和开方。该离散尺度是开发期保守分辨率尺度，并非传感器误差或skfmm严格误差上界。旧数据及正在运行的冻结源码不修改。

远程10项单位/输入测试通过；解析球体四种spacing下，最大接触重建误差/新尺度分别0.572、0.286、0.292、0.317。该结果支持继续开发校准，不能外推为断层、薄体或地形边界全部通过，正式契约仍未冻结。[测试](../tasks/copper_inversion/resources/evidence/v2-development/measurement-policy-tests-r1.log)、[校准](../tasks/copper_inversion/resources/evidence/v2-development/geometry-calibration-r3.json)。

旧开发选型6种方法均已产生ID/OOD测试收据，全部合法。融合LDM的母体IoU为0.318871/0.282751，默认MCMC为0.222316/0.210484；这是10秒、单发现、少量开发母体且默认基线未同额调参的结果，不能替代正式公平比较。开发集已见，不再当作新测试数据。

## 第十九次实施增量：私有接口拒绝与设施故障分离

服务端接口拒绝现在返回固定套件分母下的全失败指标：IoU0、合法率0、failed_solves等于计划solve_count；无需实际运行套件便可确认该源码不合法。此处solve_count是计划评分分母，不伪称实际启动了这些solver；实际作业数仍以私有账本为准。客户端将这类完整已知结果保留为零分Observation并标记service_outcome，避免反复暂停造成免费换候选。相同源码重放只消耗一个槽位。

非完整服务响应、源文件或收据读取错误、损坏JSON继续触发EvaluationPaused，不补造失败成绩。指标新增合法率与失败分母一致性检查、非空套件检查。远程25项服务、客户端和预算测试通过，覆盖服务拒绝不启动套件、完整分母、幂等扣费与损坏收据不重提交。[证据](../tasks/copper_inversion/resources/evidence/v2-development/service-failure-tests-r2.log)。新改动尚未部署到冻结service-r2或fusion-service-pilot-r1。

本轮核实PID710173仍存活，约3分钟，已完成初始化与主研究回合；候选分支尚未提交完成，未因账本暂未刷新而重启。闭环结果仍待真实完成收据，正式目标保持未完成。

## 第二十次实施增量：真实guest隔离探针与研究出站策略

核查运行中的Pi sidecar实际Docker挂载：仅分支artifacts、Gondolin缓存和只读resources；非privileged、UID1001、没有私有评估目录、Docker socket或宿主SSH目录。[挂载证据](../tasks/copper_inversion/resources/evidence/v2-development/fusion-service-sidecar-mounts-r1.json)。这不等于guest攻击验收，因此新增probe_guest.mjs，直接复用锁定镜像中的GondolinController启动独立测试guest，不调用LLM、不修改运行中的研究workspace。

真实guest探针通过：read工具拒绝workspace外路径；bash尝试覆盖只读canary、新建只读文件及经workspace符号链接改写均失败；私有真值/canary/socket与宿主SSH路径不可读；guest不继承注入sidecar的假API key和SSH agent环境；无法连接私有Unix socket。[探针证据](../tasks/copper_inversion/resources/evidence/v2-development/guest-isolation-probe-r2.json)。探针不声明穷尽虚拟机逃逸、任意网络协议或管理员威胁。

代码审计同时发现研究guest使用了框架默认开放网络策略。铜矿任务现显式设置denied_hosts=('*',)，模型调用仍在受信provider proxy，不依赖guest出站。配置3项远程测试通过，固定官方端点并断言deny-all；真实guest的HTTPS探测失败。[配置测试](../tasks/copper_inversion/resources/evidence/v2-development/research-egress-tests-r1.log)。单个失败请求不单独构成全面网络隔离证明，正式资格仍需更多协议与真实LLM闭环检查。

当前fusion-service-pilot-r1保持旧冻结网络配置，仍为开发诊断；分支provider记录持续更新、进程存活，尚未形成完整结果。新网络策略及前一轮服务拒绝修复须在下一独立运行身份下验收，不能把当前运行重标为新策略通过。

## 第二十一次实施增量：私有训练封存衔接选型

服务新增seal_training固定请求：与evaluate在同一串行服务内执行，审计私有完整训练收据后按既有IoU/合法率/时间/SHA规则选最多3个候选，并持久封存。封存后训练请求拒绝，重放返回原候选与顺序；回复不包含私有路径或逐case信息。该请求不执行验证/测试，也不接受数据路径。

新增受信账户CLI freeze_service_selection，核对服务身份、训练封存和最新私有收据后，调用既有selection.freeze；不要求研究控制进程接触训练真值或验证/测试manifest。验证/测试仍沿用原有固定状态机，正式模式门槛不变。该接口尚未部署至运行中的service-r2。

首次远程测试发现JSON排序会改变重放的候选顺序，已增加显式source_sha256s列表并在选型入口检查其唯一性与集合一致性。9项服务封存、衔接与原选型测试现全部通过。[测试](../tasks/copper_inversion/resources/evidence/v2-development/service-selection-tests-r2.log)。仍需私有账户真实封存→验证→测试验收，不能仅凭mock路径测试宣布已完成。

### 私有服务融合闭环已完成

fusion-service-pilot-r1进程已退出，status=completed，result.json记录5条Observation。初始化IoU0.06982055，最佳0.07914852；生成8个候选并选4个，研究79次LLM请求。控制器与私有服务科学槽位均为5；私有侧30个接口、10次solver和10次独立scorer，求解墙钟累计41.24秒。控制器另有54个公开接口作业，不能遗漏或与私有账本混算成重复训练反馈。[完整审计](../tasks/copper_inversion/resources/evidence/v2-development/fusion-service-pilot-r1-audit.json)。

这证明公开catalog→真实提案→融合选择→私有评分→共享Campaign记录的开发闭环已运行，不证明恢复质量理想：最佳IoU仍仅0.079。仅1母体/2包，当前运行使用封存时的旧网络策略与服务协议；最新deny-all及训练封存功能仍需新身份下验收，正式资格未完成。

## 第二十二次实施增量：最新策略的完整开发套件运行

新建source-fusion-service-suite-r1与私有code/service-r3两个冻结快照，包含deny-all研究guest网络策略、私有接口失败计分和训练封存。私有datasets/development-suite-r1持有原开发train/validation/ID/OOD数据；train公开发布为public-training-suite-r1。数据仍为旧开发配方和已见pilot，不冒充新正式测试或v4校准数据。

service-r3外层PID835378，公开契约摘要f186fffa05b4ddcad1982897538552b35a2e9458188315fc0082039188aee8e1，公开观测摘要及10秒/5槽位握手通过。融合运行PID837201，8母体/16观测包、1轮8提案选4＋初始化、官方DeepSeek V4.1 Flash/low、既有Qwen缓存。密钥通过受保护SSH管道传入环境。检查时进程存活并执行初始化套件，未重启任何旧身份。

当前全部v2远程回归74项通过（12.36秒）；它们覆盖现有实现与开发容器测试，不等于所有正式科学/权限门槛已完成。[回归](../tasks/copper_inversion/resources/evidence/v2-development/v2-regression-r1.log)、[契约](../tasks/copper_inversion/resources/evidence/v2-development/service-r3-public-contract.json)、[运行身份](../tasks/copper_inversion/resources/evidence/v2-development/fusion-service-suite-r1-launch.json)。下一项实证是此运行完成后在私有账户真实封存并沿用固定验证/测试协议。

## 第二十三次实施增量：经典调参复用私有评分边界

tune_stochastic已移除私有manifest参数和直接SuiteEvaluator，改用公开catalog、契约绑定和同一个ServiceEvaluator。Sobol参数表、只读准入、ScientificSlots、源码封存与幂等调参收据继续复用；服务身份进入计划摘要，不能改服务后续跑。更新中断恢复测试，调参与客户端共24项远程测试通过。[测试](../tasks/copper_inversion/resources/evidence/v2-development/tuning-service-tests-r1.log)。

使用与LDM service-r3相同冻结服务源码、镜像、私有train数据和公开观测摘要，建立两个独立5槽位/10秒服务：service-mcmc-r1（外层PID869366）与service-acq-r1（869367）。公开调参源码冻结为source-tuning-service-suite-r1，客户端PID869697/869701；均8母体/16包、seed47，独立参数搜索5槽位。LDM、MCMC、ACQ不共享可消耗额度，实际求解/评分费用各自保留。

检查时三个客户端进程均存活，新基线首候选正在执行；当前无完成结果，不提前报告优劣。[MCMC身份](../tasks/copper_inversion/resources/evidence/v2-development/mcmc-service-tuning-r1-launch.json)、[ACQ身份](../tasks/copper_inversion/resources/evidence/v2-development/acq-service-tuning-r1-launch.json)。仍为开发同预算检查，不是正式81槽位/120秒/三发现重复实验。

## 第二十四次实施增量：三组完成与私有封存→验证→测试实证

三组训练均已完成5槽位、80次求解及独立评分，全部合法，无失败求解。核查完成状态及账本，未因进程退出误判故障或重启。训练最佳IoU分别为融合LDM0.216662、MCMC0.204799、ACQ0.204897。

三个私有训练服务均完成持久seal_training；重复封存顺序不变，封存后新训练评估被拒绝。新增compare_service_selection开发编排入口，复用freeze_from_service和既有selection状态机：全部方法冻结后验证、全部验证结束后测试，完成后重放核对收据。未改变活动冻结科学实现。

真实运行three-method-selection-r1在ldm-copper-eval账户中完成；三方法状态均tested，各验证3候选×8包、测试唯一候选×16包，seed47，每次10秒/2CPU/4GiB。450份作业及逐case收据重放后修改时间不变。LDM的ID/OOD测试IoU0.254444/0.309812，MCMC0.222316/0.210484，ACQ0.222316/0.216101；测试全部合法、无失败。LDM在ID仅胜2/4母体、OOD胜3/4母体，不能只报告均值优势。

[完整比较报告](copper-v2-service-comparison.md)包含同预算定义、最终源码身份、曲线、求解及发现耗时、母体差异、原始证据。私有流程开发验收已完成；数据仍为已见旧开发配方，单种子，不能当作新的正式独立泛化证据。科学校准、新数据、正式预算及多次重复门槛仍未解除。

## 第二十五次实施增量：main同步、独立合成发布与九组正式启动

合并main 3580273b，保留原工作。预算更新为20×16＋初始化=321；4侧枝各8提案。接入共享MainHistoryForks 1+3角色提示，修复铜矿角色丢失及固定研究seed。服务、调参器与任务注册同步321上限，formal入口校验固定预算和42/43/3407；私有服务与选型另外核验不可仅凭stage声明的发布资格。套件固定4路case并发，执行账本写入加锁。

18项解析几何/spacing校准通过；私有账户生成56个新母体及112包，完成近重复、接触误差、air与旧开发指纹排除检查；另生成独立确认观测，正式manifest绑定确认文件摘要，独立scorer计算辅助误差，求解器不接触这些观测。正式发布仅代表受当前检查约束的合成benchmark，不代表真实矿区/专家验证或任意亚体素几何均已校准。详见[正式实验记录](copper-v2-formal-study.md)。

远程v2回归76通过2跳过，真实确认观测容器测试1通过；新Pi镜像8100350c5774编译和guest隔离探针通过。部署阶段解释器、包装脚本、依赖环境和共享tmp权限故障已修复并留档，未因故障更换原定随机流。

九组正式运行位于/Aidata/shared/ldm/copper-formal-study-20261024-r1，监督器1641770。检查时全部存活，LDM三个发现种子已各完成初始化完整48包评估并进入第一轮研究；基线正在首个配置。固定批处理已接通全部训练结束封存→全部验证→全部测试→重放核对。发生失败将保留原运行并停止后续阶段，不重开免费额度。当前没有正式稳定性结果；不得把“启动”记为“完成”。

### 增量26：保留科学身份的并发调度

新增 `experiments/v2/parallel_runtime.py`，连接共享批评估协议、私有线程服务、固定参数表并行和跨进程flock资源队列。增加批顺序、预算幂等、同源互斥、封存等待和队列释放测试。远程28项通过。

正式九组原目录已迁移至独立调度纪元（80全局/16进程），冻结源码没有被覆盖，8252份旧收据与身份摘要核对一致。`reschedule_formal.py` 记录迁移前状态；`continue_parallel.py` 接管监督与终态流程。实际达到80个求解容器并发。

`replay_parallel.py` 修复未完成研究轮次恢复时消息JSON字段序变化造成的摘要不一致：仅当完整解析内容和元数据等价时使用持久化原消息，仍检查原摘要。42/3407两组已接续，远程额外恢复测试1项通过。操作文件单独记录摘要，不能混同科学源码身份，也不能将已发生中断和开销从报告移除。
