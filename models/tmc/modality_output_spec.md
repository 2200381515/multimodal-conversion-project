# Week3-Day1: 三模态统一输出协议（TMC 接口）

目标：把量表（scale）、结构连接（SC）、功能连接（FC）三路 encoder 的输出统一成一个可被 TMC/DS 融合消费的协议。
Day1 只定义协议与 evidence/opinion 的计算；Day2 会基于此实现 DS 融合与缺失模态策略。

---

## 1. 统一的“模态输出”结构（ModalityOutput）

每个模态 m ∈ {scale, sc, fc} 都应输出一个 dict（或 dataclass），至少包含：

- `tokens`: Tensor, shape [B, T, D] 或 None
- `pooled`: Tensor, shape [B, D]（给 EvidenceHead 用）
- `logits`: Tensor, shape [B, K]
- `evidence`: Tensor, shape [B, K], evidence >= 0
- `alpha`: Tensor, shape [B, K], alpha = evidence + prior
- `prob`: Tensor, shape [B, K], Dirichlet mean
- `belief`: Tensor, shape [B, K]
- `uncertainty`: Tensor, shape [B, 1], u = K / sum(alpha)
- `strength`: Tensor, shape [B, 1], S = sum(alpha)

以及两项“可信融合必需”的元信息：

- `modality_mask`: Tensor[bool] 或 Tensor[int], shape [B]
  - 1 表示该样本该模态存在且可用；0 表示缺失/读失败/被 QC 剔除
- `quality_score`: Tensor[float], shape [B] 或 [B,1]
  - 建议归一化到 [0, 1]。可用于后续融合权重或作为鲁棒性分析分组依据。

> Day2 的缺失模态策略建议：
> - 当 mask=0 时，直接把该模态设为“高不确定/低证据”（例如 evidence=0 -> alpha=1 -> u≈1）
> - 或者直接跳过该模态的 DS 融合（但要保证 fused 输出稳定）

---

## 2. TMC 后端融合所需接口

Day2/Day3 需要如下输入/输出：

输入：
- 每个模态的 `alpha`（或直接输入 belief+uncertainty）
- `modality_mask`, `quality_score`（用于缺失与低质处理）

输出：
- `fused_belief`: [B, K]
- `fused_uncertainty`: [B, 1]
- `fused_alpha`（可选：用于算 fused loss）

---

## 3. 数值稳定要求（必须遵守）

- evidence 必须 >= 0（ReLU 或 Softplus）
- alpha = evidence + prior，prior > 0（默认 1）
- 在任何转换后都应检查 `NaN/Inf`
- strength 需要 clamp_min(eps) 避免除 0

---

## 4. 约定的 key 名称

请固定使用以下 key 名，方便训练/评估脚本统一读取：

`tokens`, `pooled`, `logits`, `evidence`, `alpha`, `prob`,
`belief`, `uncertainty`, `strength`,
`modality_mask`, `quality_score`
