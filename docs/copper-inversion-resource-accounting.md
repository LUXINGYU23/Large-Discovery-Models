# 完整工作负载的内存峰值与 CPU 记录

## 平台与测量范围

主服务器内核为 5.15，实际 cgroup 没有 `memory.peak`；已授权 H20 服务器
`120.133.37.228:10522` 使用 6.8，容器实测支持该计数器。真实 campaign 预检现在
要求此能力，不能用轮询最大值代替。代码、环境、缓存和活动结果位于
`/Aidata/shared/ldm/`，没有升级任何服务器内核。

内核 `memory.peak` 保留 cgroup 及后代自创建以来的最大内存使用量，短暂分配和
随后释放不会丢失该峰值。[Linux cgroup v2 文档](https://www.kernel.org/doc/html/v6.11/admin-guide/cgroup-v2.html)

测量边界为工作进程及全部后代结束并回收；计数包含监督器到这一时点的开销。
CPU 字段明确命名为 `cpu_seconds_through_reap`。读数之后的遥测序列化不包含在
这个边界中，不将其描述为监督器退出后的总 CPU。宿主 0.2 秒采样继续作为独立诊断。

## 信任与恢复

- 可信 Python PID 1 保留 SETUID/SETGID/KILL，用于创建和清理 worker。
- worker 使用 UID/GID 65534、清空附加组、没有有效 capabilities，保持无网络、只读
  代码/输入挂载、no-new-privileges、CPU/内存/PID 限制。
- 每个 job 在宿主私有 manifest 中生成独立随机认证密钥，仅传给监督器。worker 的
  环境不含该密钥，也不能读取 PID 1 的环境或向它发信号。
- worker 结束后监督器清理全部后代，读取内核计数，并输出 HMAC 认证记录。
  控制器只接受认证成功的记录；候选打印的伪造 JSON 不构成证据。
- 容器内 GNU timeout 保持工作负载时限，宿主只允许额外 3 秒有界收尾。控制器断连时
  容器仍自行结束并保留 Docker 日志，恢复后可验证同一记录并按原 usage key 计账。
- OOM 由 Docker 状态及可信 `memory.events` 共同判断。已知强制终止可能缺少记录，
  此时诊断为未知；其他缺失认证记录属于基础设施暂停，不伪装成有效评分。

认证密钥是每个 job 的内部记录认证材料，不是模型 API key；原始 manifest 不暴露给
研究 guest 或候选，也不作为公开反馈。公开历史只包含验证后的资源数据。

## 远程证据

[H20 机器证据](../tasks/copper_inversion/resources/evidence/runtime/supervisor.json)：
`COPPER_REQUIRE_PEAK=1` 下 228 项通过，要求峰值存在，不跳过这个断言。

- 候选 fork 的子进程瞬时分配 96 MiB 后退出，内核峰值约 124 MiB，正确保留。
- 控制器在 dispatch 后立刻断连，宿主采样数为 0；恢复仍取回约 122 MiB 峰值与
  2.07 秒 CPU，执行结果正确标记为超时。
- 验证 worker 身份/capabilities、密钥不可读、监督器不可被 worker 发信号、伪造
  记录拒绝，以及原有只读接口、独立评分、故障恢复和共享引擎测试。
- H20 依赖检查通过；主服务器缺失 `memory.peak` 时真实配置预检明确失败。

同一 evaluator 镜像 digest 保持
`sha256:9c0c855e43120f0000e3e5dcf79002763c65e55839e4908306cac29d1ff71062`。
Pi sidecar `sha256:caffc0f9407f0ac895f8019af3d80f6d2feaa94df38218e800a01af7226a0df4`
及铜矿 guest `ldm/copper-inversion-research:1d71efa7d6f5` 已同步到 H20，为下一阶段真实
多分支验收准备；本测试记录不声称已完成那次模型运行。
