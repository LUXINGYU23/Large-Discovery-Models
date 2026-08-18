# Iron Mind 版本化资源

本目录只保存公开复现所需、适合进入 Git 的小型固定资源：

- `upstream_contract.json`：上游 revision、suite、数据文件哈希、行数与 schema 哈希；
- `reaction_schemas.json`：无外部数据 mock 所使用的固定 schema；
- `mock_oracle.csv`：仅用于软件闭环测试的确定性 fixture；
- `qualification_evidence.json`：仓库任务注册流程要求的紧凑状态说明。
- `verification_record.json`：数据契约、独立 evaluator 探针与真实 LDM smoke 的发布验证摘要。

正式 Olympus 数据、运行输出、模型文件和凭据必须保存在仓库外。正式运行会根据
`upstream_contract.json` 重新构造 schema，并拒绝 revision、文件内容、行数或 schema
不一致的数据快照。
