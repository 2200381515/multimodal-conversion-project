# Multimodal Conversion Project（多模态转化预测）

本仓库用于实现“多模态转化预测”的工程化复现流程，当前已覆盖：
- Week 1：数据管线（队列构建 / QC / 标签 / 防泄漏检查）
- Week 2：Baseline 与统一评估（Scale-only / SC-only / FC-only / Concat baseline；YAML 驱动可复现）

> 免责声明：本项目目前仅用于科研与工程验证，不构成任何临床诊断、治疗或医疗建议。

---

## 项目结构

```text
multimodal-conversion-project/
├─ baselines/                 # Week2：baseline训练与评估（统一入口 run_week2.py）
├─ configs/                   # 运行配置（YAML）
├─ pipeline/                  # Week1：数据管线
├─ startup/                   # 启动期工具与报告
├─ results/                   # 运行输出（建议不提交Git）
├─ data/                      # 数据（建议不提交Git）
└─ training/                  # 后续深度学习训练入口（可扩展）
