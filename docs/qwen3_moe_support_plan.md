# Qwen3 MoE 支持计划

## 1. 背景

当前 Nano-vLLM 仅支持 Dense 模型（Qwen3），本文档规划如何扩展以支持 Qwen3 MoE 模型的推理。

### 1.1 Qwen3 MoE 架构概述

Qwen3 MoE 与 Dense 版本的核心区别在于 **MLP 层被替换为稀疏 MoE 层**：

```
Dense Qwen3:
    DecoderLayer = Attention + MLP

Qwen3 MoE:
    DecoderLayer = Attention + SparseMoeBlock (部分层)
                 = Attention + MLP            (部分层，由 decoder_sparse_step 控制)
```

### 1.2 MoE 关键组件

| 组件 | 作用 |
|------|------|
| Router (Gate) | 计算每个 token 应该路由到哪些专家 |
| Experts | 多个并行的 FFN，每个专家结构与原 MLP 相同 |
| Top-K 选择 | 每个 token 只激活 k 个专家（稀疏激活） |
| 加权求和 | 将激活专家的输出按路由权重加权求和 |

### 1.3 关键配置参数

```python
num_experts: int           # 专家总数（如 64）
num_experts_per_tok: int   # 每个 token 激活的专家数（如 8）
moe_intermediate_size: int # MoE 层的中间维度
decoder_sparse_step: int   # 每隔多少层使用 MoE（如 1 表示每层都用）
mlp_only_layers: list      # 强制使用 Dense MLP 的层索引
```

---

## 2. 需要新增的模块

### 2.1 模型层模块 (`nanovllm/layers/`)

#### 2.1.1 `moe.py` - MoE 核心实现

```python
# 需要实现的类：

class MoeRouter(nn.Module):
    """
    Top-K 路由器
    
    输入: hidden_states [num_tokens, hidden_size]
    输出: 
        - router_weights [num_tokens, top_k] 路由权重
        - selected_experts [num_tokens, top_k] 选中的专家索引
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, norm_topk_prob: bool = True):
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
    
    def forward(self, hidden_states: torch.Tensor):
        # 1. 计算路由 logits: [num_tokens, num_experts]
        router_logits = F.linear(hidden_states, self.weight)
        
        # 2. Softmax 归一化
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # 3. Top-K 选择
        router_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 4. 可选：归一化 top-k 权重
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        
        return router_weights.to(hidden_states.dtype), selected_experts


class MoeExperts(nn.Module):
    """
    专家集合 - 所有专家权重合并为 3D 张量
    
    权重形状:
        - gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
        - down_proj: [num_experts, hidden_size, intermediate_size]
    """
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.act_fn = SiluAndMul()
    
    def forward(self, hidden_states, selected_experts, router_weights):
        # 实现专家计算（见下方详细设计）
        pass


class SparseMoeBlock(nn.Module):
    """
    稀疏 MoE 块 - 组合 Router 和 Experts
    """
    def __init__(self, config):
        self.router = MoeRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
        )
        self.experts = MoeExperts(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
    
    def forward(self, hidden_states):
        # 1. 路由
        router_weights, selected_experts = self.router(hidden_states)
        
        # 2. 专家计算
        output = self.experts(hidden_states, selected_experts, router_weights)
        
        return output
```

#### 2.1.2 专家计算的两种实现方案

**方案 A：朴素实现（循环遍历）**

```python
def forward(self, hidden_states, selected_experts, router_weights):
    num_tokens = hidden_states.size(0)
    final_output = torch.zeros_like(hidden_states)
    
    # 为每个专家找到分配给它的 tokens
    for expert_idx in range(self.num_experts):
        # 找到路由到该专家的 (token_idx, top_k_idx)
        mask = (selected_experts == expert_idx)
        if not mask.any():
            continue
        
        token_indices, topk_indices = torch.where(mask)
        expert_input = hidden_states[token_indices]
        
        # 专家计算: gate-up-down
        gate_up = F.linear(expert_input, self.gate_up_proj[expert_idx])
        expert_output = self.act_fn(gate_up)
        expert_output = F.linear(expert_output, self.down_proj[expert_idx])
        
        # 加权累加
        weights = router_weights[token_indices, topk_indices].unsqueeze(-1)
        final_output.index_add_(0, token_indices, expert_output * weights)
    
    return final_output
```

**方案 B：高效实现（Grouped GEMM / Triton Kernel）**

```python
# 使用 Triton 实现高效的 token 重排序和批量专家计算
# 参考 vLLM 的 fused_moe kernel 或 Megablocks 库

@triton.jit
def fused_moe_kernel(...):
    """
    融合的 MoE 计算 kernel:
    1. Token 按专家分组重排序
    2. 批量执行专家计算
    3. 结果按原顺序重排回去
    """
    pass
```

**建议：** 第一阶段使用方案 A 实现功能，后续优化时引入方案 B。

---

### 2.2 模型实现 (`nanovllm/models/`)

#### 2.2.1 `qwen3_moe.py` - Qwen3 MoE 模型

```python
from transformers import Qwen3MoeConfig
from nanovllm.layers.moe import SparseMoeBlock

class Qwen3MoeMLP(nn.Module):
    """与 Dense 版本相同的 MLP，用于非 MoE 层"""
    # 复用现有 Qwen3MLP 实现


class Qwen3MoeAttention(nn.Module):
    """与 Dense 版本相同"""
    # 复用现有 Qwen3Attention 实现


class Qwen3MoeDecoderLayer(nn.Module):
    """
    MoE 版本的 Decoder 层
    根据 layer_idx 决定使用 MoE 还是 Dense MLP
    """
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(...)
        
        # 关键：条件选择 MoE 或 Dense MLP
        use_moe = self._should_use_moe(config, layer_idx)
        if use_moe:
            self.mlp = SparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config)
        
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    def _should_use_moe(self, config, layer_idx):
        if layer_idx in getattr(config, 'mlp_only_layers', []):
            return False
        sparse_step = getattr(config, 'decoder_sparse_step', 1)
        return (layer_idx + 1) % sparse_step == 0
    
    def forward(self, positions, hidden_states, residual):
        # 与 Dense 版本相同的残差连接逻辑
        ...


class Qwen3MoeModel(nn.Module):
    """模型主体"""
    def __init__(self, config: Qwen3MoeConfig):
        self.embed_tokens = VocabParallelEmbedding(...)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, i) 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(...)


class Qwen3MoeForCausalLM(nn.Module):
    """顶层入口"""
    packed_modules_mapping = {
        # 与 Dense 版本类似，但需要添加 MoE 专家权重映射
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # MoE 专家权重映射（如果需要）
    }
    
    def __init__(self, config: Qwen3MoeConfig):
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(...)
```

---

### 2.3 权重加载 (`nanovllm/utils/loader.py`)

需要扩展权重加载器以支持 MoE 专家权重：

```python
def load_moe_experts(experts: MoeExperts, state_dict: dict, prefix: str):
    """
    加载 MoE 专家权重
    
    HuggingFace 权重格式:
        experts.{expert_id}.gate_proj.weight
        experts.{expert_id}.up_proj.weight
        experts.{expert_id}.down_proj.weight
    
    Nano-vLLM 格式:
        gate_up_proj: [num_experts, 2*inter, hidden]
        down_proj: [num_experts, hidden, inter]
    """
    num_experts = experts.num_experts
    
    for expert_id in range(num_experts):
        gate_weight = state_dict[f"{prefix}.experts.{expert_id}.gate_proj.weight"]
        up_weight = state_dict[f"{prefix}.experts.{expert_id}.up_proj.weight"]
        down_weight = state_dict[f"{prefix}.experts.{expert_id}.down_proj.weight"]
        
        # 合并 gate 和 up
        experts.gate_up_proj.data[expert_id] = torch.cat([gate_weight, up_weight], dim=0)
        experts.down_proj.data[expert_id] = down_weight
```

---

### 2.4 引擎适配 (`nanovllm/engine/model_runner.py`)

#### 2.4.1 模型类型自动选择

```python
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM

def get_model_class(config):
    """根据配置自动选择模型类"""
    if hasattr(config, 'num_experts') and config.num_experts > 0:
        return Qwen3MoeForCausalLM
    return Qwen3ForCausalLM

class ModelRunner:
    def __init__(self, config: Config, ...):
        ...
        model_class = get_model_class(hf_config)
        self.model = model_class(hf_config)
        ...
```

#### 2.4.2 KV Cache 分配（无需修改）

MoE 只影响 FFN 层，不影响 Attention 层，因此 KV Cache 分配逻辑无需修改。

---

## 3. 文件变更清单

### 3.1 新增文件

| 文件路径 | 说明 |
|----------|------|
| `nanovllm/layers/moe.py` | MoE 核心层：Router、Experts、SparseMoeBlock |
| `nanovllm/models/qwen3_moe.py` | Qwen3 MoE 模型实现 |

### 3.2 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `nanovllm/utils/loader.py` | 添加 MoE 专家权重加载逻辑 |
| `nanovllm/engine/model_runner.py` | 添加模型类型自动选择 |
| `nanovllm/__init__.py` | 导出 MoE 相关类（可选） |

---

## 4. 实现计划

### Phase 1: 基础功能（优先级高）

1. **实现 `moe.py`**
   - MoeRouter 类
   - MoeExperts 类（朴素循环实现）
   - SparseMoeBlock 类

2. **实现 `qwen3_moe.py`**
   - 复用 Dense 版本的 Attention、Embedding、LMHead
   - 实现条件 MoE/Dense 选择逻辑

3. **适配权重加载**
   - 支持 HuggingFace Qwen3-MoE 权重格式

4. **适配 ModelRunner**
   - 模型类型自动检测

**预计代码量：** ~300 行

### Phase 2: 性能优化（优先级中）

1. **Triton Kernel 优化**
   - 实现 fused_moe_kernel
   - Token 重排序优化

2. **专家并行（Expert Parallelism）**
   - 跨 GPU 分布专家

**预计代码量：** ~200 行

### Phase 3: 高级特性（优先级低）

1. **负载均衡监控**
   - 统计各专家被激活的频率

2. **动态批处理优化**
   - 针对 MoE 的调度策略优化

---

## 5. 测试计划

### 5.1 单元测试

```python
def test_moe_router():
    """测试路由器输出形状和 top-k 选择"""
    router = MoeRouter(hidden_size=512, num_experts=8, top_k=2)
    hidden = torch.randn(100, 512)
    weights, indices = router(hidden)
    assert weights.shape == (100, 2)
    assert indices.shape == (100, 2)
    assert (indices >= 0).all() and (indices < 8).all()

def test_moe_experts():
    """测试专家计算的正确性"""
    # 对比朴素实现和优化实现的输出

def test_sparse_moe_block():
    """测试完整 MoE 块的前向传播"""
```

### 5.2 集成测试

```python
def test_qwen3_moe_generation():
    """测试完整的生成流程"""
    llm = LLM("Qwen/Qwen3-30B-A3B", ...)  # MoE 模型
    outputs = llm.generate(["Hello"], SamplingParams(...))
    assert len(outputs) > 0
```

### 5.3 性能基准

```python
def benchmark_moe_vs_dense():
    """对比 MoE 和 Dense 模型的推理速度"""
    # 相同参数量下，MoE 应该更快（稀疏激活）
```

---

## 6. 参考资料

1. [HuggingFace Qwen3-MoE 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
2. [vLLM MoE 实现](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe)
3. [Megablocks: Efficient Sparse Training](https://github.com/databricks/megablocks)
4. [Switch Transformer 论文](https://arxiv.org/abs/2101.03961)

---

## 7. 风险与注意事项

1. **显存占用**
   - MoE 模型专家权重较大，需确保显存充足
   - 建议先在小规模 MoE 模型上测试（如 Qwen3-30B-A3B）

2. **数值精度**
   - 路由器 softmax 建议用 float32 计算，避免数值不稳定

3. **CUDA Graph 兼容性**
   - 朴素循环实现可能无法使用 CUDA Graph
   - Triton Kernel 版本应可兼容

4. **张量并行**
   - Phase 1 先实现单 GPU 版本
   - 专家并行需要额外设计通信策略



Phase 1 基础功能已完成。以下是实现总结：

  实现总结

  新增文件

  1. nanovllm/layers/moe.py (~150 行)

  ┌──────────────────┬─────────────────────────────────────────────────┐
  │ 类               │ 功能                                            │
  ├──────────────────┼─────────────────────────────────────────────────┤
  │ `MoeRouter`      │ Top-K 路由器，计算每个 token 应路由到的专家     │
  ├──────────────────┼─────────────────────────────────────────────────┤
  │ `MoeExperts`     │ 专家集合，3D 张量存储所有专家权重，循环遍历计算 │
  ├──────────────────┼─────────────────────────────────────────────────┤
  │ `SparseMoeBlock` │ 组合 Router + Experts，替代标准 MLP             │
  └──────────────────┴─────────────────────────────────────────────────┘

  2. nanovllm/models/qwen3_moe.py (~220 行)

  ┌────────────────────────┬────────────────────────────────────────┐
  │ 类                     │ 功能                                   │
  ├────────────────────────┼────────────────────────────────────────┤
  │ `Qwen3MoeAttention`    │ 与 Dense 版本相同的注意力层            │
  ├────────────────────────┼────────────────────────────────────────┤
  │ `Qwen3MoeMLP`          │ Dense MLP，用于非 MoE 层               │
  ├────────────────────────┼────────────────────────────────────────┤
  │ `Qwen3MoeDecoderLayer` │ 根据 `layer_idx` 选择 MoE 或 Dense MLP │
  ├────────────────────────┼────────────────────────────────────────┤
  │ `Qwen3MoeModel`        │ 模型主体                               │
  ├────────────────────────┼────────────────────────────────────────┤
  │ `Qwen3MoeForCausalLM`  │ 顶层入口                               │
  └────────────────────────┴────────────────────────────────────────┘

  修改文件

  3. nanovllm/utils/loader.py

  - 新增 load_moe_expert_weight() 函数
  - 支持加载 MoE 专家权重（gate_proj, up_proj, down_proj）
  - 支持加载 Router 权重

  4. nanovllm/engine/model_runner.py

  - 新增 get_model_class() 函数
  - 根据 hf_config.num_experts 自动选择 Dense 或 MoE 模型

  使用方式

  使用方式与 Dense 模型完全相同，会自动检测模型类型：

  from nanovllm import LLM, SamplingParams

  # MoE 模型会自动识别
  llm = LLM("/path/to/Qwen3-30B-A3B", enforce_eager=True)
  outputs = llm.generate(["Hello"], SamplingParams(temperature=0.6, max_tokens=256))

  注意事项

  1. CUDA Graph：朴素循环实现可能与 CUDA Graph 不兼容，建议使用 enforce_eager=True
  2. 性能：当前为朴素实现，后续 Phase 2 可通过 Triton Kernel 优化
  3. 张量并行：当前 MoE 层未实现张量并行，仅支持单 GPU
