是的，目前只支持 Dense 模型

  限制原因

  从代码中可以看出，有以下几个关键限制：

  1. MLP 层是标准的 Dense FFN（qwen3.py:90-116）

  class Qwen3MLP(nn.Module):
      def __init__(self, hidden_size, intermediate_size, hidden_act):
          self.gate_up_proj = MergedColumnParallelLinear(...)  # 固定的线性层
          self.down_proj = RowParallelLinear(...)              # 固定的线性层
          self.act_fn = SiluAndMul()

      def forward(self, x):
          gate_up = self.gate_up_proj(x)  # 所有 token 走同一路径
          x = self.act_fn(gate_up)
          x = self.down_proj(x)
          return x

  MoE 需要的是：
  # MoE 的 forward 逻辑（伪代码）
  def forward(self, x):
      router_logits = self.gate(x)           # 路由器计算专家分数
      routing_weights = softmax(router_logits)
      selected_experts = topk(routing_weights)  # 选择 top-k 专家

      # 每个 token 路由到不同专家
      for expert_id in selected_experts:
          expert_output += experts[expert_id](x) * weight

  2. 缺少 MoE 核心组件

  ┌───────────────────────────────┬────────────┬──────────┬──────────┐
  │ 组件                          │ Dense 模型 │ MoE 模型 │ 当前支持 │
  ├───────────────────────────────┼────────────┼──────────┼──────────┤
  │ 路由器 (Router/Gate)          │ 无         │ 需要     │ ❌       │
  ├───────────────────────────────┼────────────┼──────────┼──────────┤
  │ 多个专家 (Experts)            │ 1 个 FFN   │ N 个 FFN │ ❌       │
  ├───────────────────────────────┼────────────┼──────────┼──────────┤
  │ Top-K 选择                    │ 无         │ 需要     │ ❌       │
  ├───────────────────────────────┼────────────┼──────────┼──────────┤
  │ 专家并行 (Expert Parallelism) │ 无         │ 可选     │ ❌       │
  └───────────────────────────────┴────────────┴──────────┴──────────┘

  3. 调度器不支持 MoE 的动态计算

  当前的 Scheduler 和 ModelRunner 假设：
  - 每个 token 的计算量相同
  - KV Cache 大小可预测

  MoE 模型中：
  - 不同 token 可能激活不同数量的专家
  - 计算量动态变化
  - 需要专门的负载均衡策略

  4. 只实现了 Qwen3（Dense）模型

  nanovllm/models/ 目录下只有 qwen3.py，没有 MoE 模型（如 Qwen2-MoE、Mixtral、DeepSeek-V2）的实现。

  ---
  如果要支持 MoE，需要做什么？

  1. 新增 MoE 层实现
    - Router/Gate 网络
    - 多专家容器
    - Top-K 路由逻辑
  2. 新增高效算子
    - Grouped GEMM（批量处理不同专家的计算）
    - Token 重排序（按专家分组）
  3. 修改调度器
    - 考虑专家负载均衡
    - 处理专家容量限制
  4. 新增模型实现
    - 如 mixtral.py、qwen2_moe.py 等

  ---
  总结

  ┌────────────────┬─────────────────────────────────────────────────────────┐
  │ 问题           │ 答案                                                    │
  ├────────────────┼─────────────────────────────────────────────────────────┤
  │ 只支持 Dense？ │ 是                                                      │
  ├────────────────┼─────────────────────────────────────────────────────────┤
  │ 主要限制       │ MLP 层是固定 FFN，无路由器、无多专家                    │
  ├────────────────┼─────────────────────────────────────────────────────────┤
  │ 根本原因       │ 项目目标是简洁可读（~1200 行），MoE 会显著增加复杂度    │
  ├────────────────┼─────────────────────────────────────────────────────────┤
  │ 能否扩展？     │ 可以，但需要新增 MoE 层、模型实现、可能还需要调度器改动 │
  └────────────────┴─────────────────────────────────────────────────────────┘



### 模型下载
使用 huggingface-cli（推荐）

```bash
export HF_HOME=/path/to/.cache/huggingface

huggingface-cli download \
    --token <YOUR_HF_TOKEN> \
    --local-dir /path/to/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --local-dir-use-symlinks False \
    Qwen/Qwen3-30B-A3B-Instruct-2507
```
