# Iron Mind 版本化资源

这里保存小型、可公开提交的任务输入和验证信息：

- `upstream_contract.json`：上游版本、suite、数据文件哈希和 schema 信息；
- `reaction_schemas.json` 与 `mock_oracle.csv`：无外部依赖的 mock fixture；
- `qualification_evidence.json` 与 `verification_record.json`：任务注册和发布验证摘要。

正式 Olympus 数据、模型凭据和运行结果由用户放在仓库外目录。数据准备脚本和依赖检查
根据 `upstream_contract.json` 验证所使用的冻结快照。
