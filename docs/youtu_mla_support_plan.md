# Youtu-LLM (MLA) 适配计划

## 1. 背景

### 1.1 MLA 简介

MLA (Multi-head Latent Attention) 是 DeepSeek-V2 提出的高效注意力机制，通过低秩压缩显著减少 KV Cache 的显存占用。Youtu-LLM 采用了 MLA 架构，需要在 Nano-vLLM 中添加支持。

### 1.2 MLA vs MHA 对比

| 特性 | 标准 MHA | MLA |
|------|----------|-----|
| KV Cache 存储 | 完整的 K, V 向量 | 压缩后的潜在向量 (latent) |
| 每层 Cache 维度 | `2 × num_kv_heads × head_dim` | `kv_lora_rank + qk_rope_head_dim` |
| RoPE 应用 | 全部 head_dim | 仅 qk_rope_head_dim 部分 |
| 投影方式 | 单次投影 | 两阶段投影 (down → up) |

### 1.3 Youtu-LLM 模型配置

```json
{
  "architectures": ["YoutuForCausalLM"],
  "model_type": "youtu",
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 16,
  "num_hidden_layers": 32,
  "intermediate_size": 6144,
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "qk_head_dim": 192,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "rope_interleave": true,
  "rope_parameters": {
    "rope_theta": 1600000,
    "rope_type": "default"
  }
}
```

### 1.4 MLA 关键参数解读

| 参数 | 值 | 说明 |
|------|-----|------|
| `kv_lora_rank` | 512 | KV 压缩后的潜在维度 |
| `q_lora_rank` | 1536 | Q 压缩后的潜在维度 |
| `qk_nope_head_dim` | 128 | 不使用 RoPE 的 QK 头维度 |
| `qk_rope_head_dim` | 64 | 使用 RoPE 的 QK 头维度 |
| `qk_head_dim` | 192 | 总 QK 头维度 (128 + 64) |
| `v_head_dim` | 128 | Value 的头维度 |

### 1.5 KV Cache 压缩效益

```
标准 MHA (假设相同头数和维度):
  每层 KV Cache = 2 × 16 × 128 = 4096

MLA (Youtu-LLM):
  每层 KV Cache = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576

压缩比 ≈ 7.1x
```

---

## 2. MLA 架构详解

### 2.1 标准 MHA 流程 (当前 Nano-vLLM)

```
hidden_states [N, hidden_size]
       │
       ▼
   QKV Proj ──────────────────► Q [N, num_heads, head_dim]
       │                        K [N, num_kv_heads, head_dim]
       │                        V [N, num_kv_heads, head_dim]
       │                              │
       │                              ▼
       │                         RoPE (全部)
       │                              │
       │                              ▼
       │                    ┌─────────────────┐
       │                    │    KV Cache     │
       │                    │ [num_blocks,    │
       │                    │  block_size,    │
       │                    │  num_kv_heads,  │
       │                    │  head_dim]      │
       │                    └─────────────────┘
       │                              │
       ▼                              ▼
                          Attention (Q @ K^T @ V)
                                      │
                                      ▼
                                   O Proj
                                      │
                                      ▼
                               output [N, hidden_size]
```

### 2.2 MLA 流程 (Youtu-LLM)

```
hidden_states [N, hidden_size]
       │
       ├───────────────────────────────────────────┐
       │                                           │
       ▼                                           ▼
  Q Down Proj                                KV Down Proj
  [hidden → q_lora_rank]                     [hidden → kv_lora_rank + qk_rope_head_dim]
       │                                           │
       ▼                                           │
   Q LayerNorm                                     ├──► k_pe [N, qk_rope_head_dim] ──► RoPE
       │                                           │
       ▼                                           │
   Q Up Proj                                       ▼
  [q_lora_rank → num_heads × qk_head_dim]    KV LayerNorm (对 kv_lora_rank 部分)
       │                                           │
       │                                           │
       ▼                                           ▼
  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
  │ q_nope [N, num_heads, qk_nope_dim]  │    │        压缩后的 KV Latent           │
  │ q_pe   [N, num_heads, qk_rope_dim]  │    │  [N, kv_lora_rank + qk_rope_head_dim]│
  └─────────────────────────────────────┘    └─────────────────────────────────────┘
       │                                           │
       ▼                                           │
   RoPE (仅 q_pe)                                  │
       │                                           │
       ▼                                           ▼
   q = concat(q_nope, q_pe)                 ┌─────────────────────────────────┐
       │                                    │         KV Cache                │
       │                                    │  [num_blocks, block_size,       │
       │                                    │   kv_lora_rank + qk_rope_dim]   │
       │                                    │                                 │
       │                                    │  << 仅存储压缩后的 latent >>     │
       │                                    └─────────────────────────────────┘
       │                                                   │
       │                                                   │
       │               ┌───────────────────────────────────┘
       │               │
       │               ▼
       │          KV Up Proj (Decode 阶段按需解压)
       │          [kv_lora_rank → num_heads × (qk_nope_dim + v_head_dim)]
       │               │
       │               ▼
       │          ┌─────────────────────────────────────┐
       │          │ k_nope [N, num_heads, qk_nope_dim]  │
       │          │ v      [N, num_heads, v_head_dim]   │
       │          └─────────────────────────────────────┘
       │               │
       │               ├──► k_pe (从 cache 中取出) ──► RoPE
       │               │
       │               ▼
       │          k = concat(k_nope, k_pe)
       │               │
       ▼               ▼
       └──────► Attention (Q @ K^T @ V) ◄───────┘
                       │
                       ▼
                    O Proj
                       │
                       ▼
                output [N, hidden_size]
```

### 2.3 关键区别总结

| 方面 | MHA | MLA |
|------|-----|-----|
| Q 投影 | 单次: `hidden → num_heads × head_dim` | 两阶段: `hidden → q_lora_rank → num_heads × qk_head_dim` |
| KV 投影 | 单次: `hidden → num_kv_heads × head_dim × 2` | 两阶段: `hidden → (kv_lora_rank + qk_rope_dim) → num_heads × (qk_nope_dim + v_head_dim)` |
| KV Cache 内容 | 完整 K, V | 压缩后的 latent + rope 部分的 k_pe |
| RoPE 应用 | Q, K 的全部维度 | 仅 Q, K 的 rope 部分 (qk_rope_head_dim) |
| Decode 时 | 直接从 Cache 取 K, V | 需要从 Cache 解压出 K, V |

---

## 3. 需要修改的模块

### 3.1 文件变更总览

| 文件路径 | 变更类型 | 说明 |
|----------|----------|------|
| `nanovllm/models/youtu.py` | **新建** | Youtu-LLM 模型实现 |
| `nanovllm/layers/attention_mla.py` | **新建** | MLA 注意力层 + KV Cache 存取 |
| `nanovllm/layers/rotary_embedding.py` | **修改** | 支持部分维度 RoPE |
| `nanovllm/layers/linear.py` | **修改** | 添加 MLA 专用投影层 |
| `nanovllm/engine/model_runner.py` | **修改** | KV Cache 分配、模型类选择 |
| `nanovllm/utils/loader.py` | **修改** | MLA 权重映射 |
| `nanovllm/config.py` | **修改** | MLA 配置检测 |

---

## 4. 详细实现方案

### 4.1 新建 `nanovllm/layers/attention_mla.py`

#### 4.1.1 MLA KV Cache 存取 Kernel

```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_mla_kvcache_kernel(
    latent_ptr,           # 压缩后的 KV latent [N, latent_dim]
    latent_stride,
    latent_cache_ptr,     # MLA KV Cache [num_blocks * block_size, latent_dim]
    slot_mapping_ptr,
    D: tl.constexpr,      # D = kv_lora_rank + qk_rope_head_dim
):
    """
    MLA 的 KV Cache 存储 kernel
    与标准 MHA 不同，MLA 只存储一份压缩后的 latent，而不是 K 和 V 两份
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    
    latent_offsets = idx * latent_stride + tl.arange(0, D)
    latent = tl.load(latent_ptr + latent_offsets)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(latent_cache_ptr + cache_offsets, latent)


def store_mla_kvcache(
    kv_latent: torch.Tensor,      # [N, kv_lora_rank + qk_rope_head_dim]
    kv_cache: torch.Tensor,       # [num_blocks * block_size, latent_dim]
    slot_mapping: torch.Tensor,   # [N]
):
    """存储 MLA 的压缩 KV latent 到 cache"""
    N, latent_dim = kv_latent.shape
    assert kv_latent.stride(-1) == 1
    assert slot_mapping.numel() == N
    
    store_mla_kvcache_kernel[(N,)](
        kv_latent, kv_latent.stride(0),
        kv_cache, slot_mapping,
        latent_dim
    )
```

#### 4.1.2 MLAAttention 类

```python
class MLAAttention(nn.Module):
    """
    MLA 注意力层
    
    与标准 Attention 的区别：
    1. KV Cache 只存储压缩后的 latent，维度为 kv_lora_rank + qk_rope_head_dim
    2. Decode 阶段需要从 cache 中解压出 K, V
    3. 需要持有 kv_b_proj 权重用于解压
    """
    
    def __init__(
        self,
        num_heads: int,
        qk_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        scale: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scale = scale
        
        # KV Cache: 只存储压缩后的 latent
        # shape: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        self.kv_cache = torch.tensor([])
        
        # kv_b_proj 用于 decode 时解压
        # 需要在模型初始化时从外部设置
        self.kv_b_proj_weight = None  # [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    
    def forward(
        self,
        q: torch.Tensor,              # [N, num_heads, qk_head_dim]
        k_pe: torch.Tensor,           # [N, 1, qk_rope_head_dim] (已应用 RoPE)
        kv_latent: torch.Tensor,      # [N, kv_lora_rank]
        k_nope: torch.Tensor,         # [N, num_heads, qk_nope_head_dim]
        v: torch.Tensor,              # [N, num_heads, v_head_dim]
    ):
        context = get_context()
        kv_cache = self.kv_cache
        
        # 拼接 kv_latent 和 k_pe 用于存储
        # [N, kv_lora_rank + qk_rope_head_dim]
        kv_latent_with_rope = torch.cat([kv_latent, k_pe.squeeze(1)], dim=-1)
        
        # 存储到 KV Cache
        if kv_cache.numel():
            store_mla_kvcache(kv_latent_with_rope, kv_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill: 直接使用当前计算的 K, V
            # 拼接 k_nope 和 k_pe 得到完整的 K
            k_pe_expanded = k_pe.expand(-1, self.num_heads, -1)  # [N, num_heads, qk_rope_head_dim]
            k = torch.cat([k_nope, k_pe_expanded], dim=-1)       # [N, num_heads, qk_head_dim]
            
            # Attention 计算
            # 注意：Q 的 head_dim 是 qk_head_dim，V 的 head_dim 是 v_head_dim
            # 需要使用支持不同 head_dim 的 attention 或拆分计算
            o = self._compute_attention_prefill(q, k, v, context)
        else:
            # Decode: 需要从 cache 中解压 K, V
            o = self._compute_attention_decode(q, context)
        
        return o
    
    def _compute_attention_prefill(self, q, k, v, context):
        """
        Prefill 阶段的 attention 计算
        
        注意：标准 flash_attn 要求 Q 和 K 的 head_dim 相同，且等于 V
        但 MLA 中 qk_head_dim != v_head_dim
        
        解决方案：
        1. 使用支持不同维度的 attention 实现
        2. 或者将 V padding 到 qk_head_dim
        """
        # 方案 1: 使用 flash_attn_varlen_func (要求 head_dim 匹配)
        # 需要处理维度不匹配问题
        
        # 临时方案: 如果 qk_head_dim == v_head_dim + qk_rope_head_dim
        # 可以考虑用 padding 或重新设计
        
        # 这里假设使用支持 MLA 的 attention kernel
        # 实际实现可能需要自定义 Triton kernel
        
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=context.block_tables if context.block_tables is not None else None,
        )
        return o
    
    def _compute_attention_decode(self, q, context):
        """
        Decode 阶段的 attention 计算
        需要从 cache 中解压 K, V
        """
        # 从 cache 中获取所有历史 kv_latent
        # 解压得到 K, V
        # 这部分较复杂，需要处理 block table 索引
        
        # 简化方案：使用 flash_attn_with_kvcache
        # 但需要先解压 cache 到完整的 K, V 格式
        
        # 完整方案需要自定义 kernel 直接在 latent 上计算
        pass
```

### 4.2 新建 `nanovllm/models/youtu.py`

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention_mla import MLAAttention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear, 
    RowParallelLinear, 
    ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope_mla
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class YoutuMLA(nn.Module):
    """
    Youtu-LLM 的 Multi-head Latent Attention 实现
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_head_dim: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_interleave: bool = True,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads // tp_size
        self.total_num_heads = num_heads
        
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        
        self.scaling = qk_head_dim ** -0.5
        
        # === Q 投影 (两阶段) ===
        # 第一阶段: hidden → q_lora_rank
        self.q_a_proj = ReplicatedLinear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        # 第二阶段: q_lora_rank → num_heads * qk_head_dim
        self.q_b_proj = ColumnParallelLinear(q_lora_rank, num_heads * qk_head_dim, bias=False)
        
        # === KV 投影 (两阶段) ===
        # 第一阶段: hidden → kv_lora_rank + qk_rope_head_dim
        # 注意: qk_rope_head_dim 部分直接输出，不经过第二阶段
        self.kv_a_proj = ReplicatedLinear(
            hidden_size, 
            kv_lora_rank + qk_rope_head_dim, 
            bias=False
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        # 第二阶段: kv_lora_rank → num_heads * (qk_nope_head_dim + v_head_dim)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank, 
            num_heads * (qk_nope_head_dim + v_head_dim), 
            bias=False
        )
        
        # === 输出投影 ===
        self.o_proj = RowParallelLinear(num_heads * v_head_dim, hidden_size, bias=False)
        
        # === RoPE ===
        self.rotary_emb = get_rope_mla(
            qk_rope_head_dim,
            max_position=max_position,
            base=rope_theta,
            interleave=rope_interleave,
        )
        
        # === Attention ===
        self.attn = MLAAttention(
            num_heads=self.num_heads,
            qk_head_dim=qk_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            scale=self.scaling,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        N = hidden_states.size(0)
        
        # === Q 投影 ===
        q = self.q_a_proj(hidden_states)              # [N, q_lora_rank]
        q = self.q_a_layernorm(q)                     # [N, q_lora_rank]
        q = self.q_b_proj(q)                          # [N, num_heads * qk_head_dim]
        q = q.view(N, self.num_heads, self.qk_head_dim)
        
        # 拆分 q_nope 和 q_pe
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # === KV 投影 ===
        kv = self.kv_a_proj(hidden_states)            # [N, kv_lora_rank + qk_rope_head_dim]
        kv_latent, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        kv_latent = self.kv_a_layernorm(kv_latent)    # [N, kv_lora_rank]
        kv = self.kv_b_proj(kv_latent)                # [N, num_heads * (qk_nope_head_dim + v_head_dim)]
        kv = kv.view(N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        
        # 拆分 k_nope 和 v
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # k_pe 扩展为 [N, 1, qk_rope_head_dim] 用于 RoPE
        k_pe = k_pe.unsqueeze(1)
        
        # === RoPE (仅对 rope 部分) ===
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        
        # 拼接得到完整的 Q
        q = torch.cat([q_nope, q_pe], dim=-1)         # [N, num_heads, qk_head_dim]
        
        # === Attention ===
        o = self.attn(q, k_pe, kv_latent, k_nope, v)  # [N, num_heads, v_head_dim]
        
        # === 输出投影 ===
        output = self.o_proj(o.flatten(1, -1))        # [N, hidden_size]
        return output


class YoutuMLP(nn.Module):
    """Youtu-LLM 的 MLP 层"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        from nanovllm.layers.linear import MergedColumnParallelLinear
        
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class YoutuDecoderLayer(nn.Module):
    """Youtu-LLM Decoder 层"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        
        self.self_attn = YoutuMLA(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_head_dim=config.qk_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_parameters.get("rope_theta", 1600000),
            rope_interleave=getattr(config, "rope_interleave", True),
        )
        
        self.mlp = YoutuMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class YoutuModel(nn.Module):
    """Youtu-LLM 模型主体"""
    
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            YoutuDecoderLayer(config, i) 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class YoutuForCausalLM(nn.Module):
    """Youtu-LLM 顶层入口"""
    
    packed_modules_mapping = {
        # Q 投影
        "q_a_proj": ("q_a_proj", None),
        "q_b_proj": ("q_b_proj", None),
        # KV 投影
        "kv_a_proj_with_mqa": ("kv_a_proj", None),
        "kv_b_proj": ("kv_b_proj", None),
        # MLP
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config):
        super().__init__()
        self.model = YoutuModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model(input_ids, positions)
    
    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)
```

### 4.3 修改 `nanovllm/layers/rotary_embedding.py`

添加 MLA 专用的 RoPE 实现，支持 interleave 模式和部分维度：

```python
class RotaryEmbeddingMLA(nn.Module):
    """
    MLA 专用 RoPE
    
    与标准 RoPE 的区别：
    1. 仅应用于 qk_rope_head_dim 维度
    2. 支持 interleave 模式
    3. K 可能只有 1 个头 (所有头共享 k_pe)
    """
    
    def __init__(
        self,
        head_size: int,            # qk_rope_head_dim
        max_position_embeddings: int,
        base: float,
        interleave: bool = True,   # 是否使用 interleave 模式
    ):
        super().__init__()
        self.head_size = head_size
        self.interleave = interleave
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        if interleave:
            # interleave 模式: [cos0, cos0, cos1, cos1, ...]
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        else:
            # 标准模式: [cos0, cos1, ..., cos0, cos1, ...]
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        
        cache = torch.stack([cos, sin], dim=-1)  # [max_pos, head_size, 2]
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    def _apply_rotary_interleave(self, x, cos, sin):
        """interleave 模式的 RoPE"""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = cos[..., 0::2]
        sin = sin[..., 0::2]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.stack([y1, y2], dim=-1).flatten(-2)
    
    def _apply_rotary_half(self, x, cos, sin):
        """标准 half-rotate 模式"""
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x1.size(-1)]
        y2 = x2 * cos[..., x1.size(-1):] + x1 * sin[..., x1.size(-1):]
        return torch.cat([y1, y2], dim=-1)
    
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,       # [N, num_heads, qk_rope_head_dim]
        key: torch.Tensor,         # [N, 1, qk_rope_head_dim] 或 [N, num_heads, qk_rope_head_dim]
    ):
        cos_sin = self.cos_sin_cache[positions]  # [N, head_size, 2]
        cos = cos_sin[..., 0].unsqueeze(1)       # [N, 1, head_size]
        sin = cos_sin[..., 1].unsqueeze(1)       # [N, 1, head_size]
        
        if self.interleave:
            query = self._apply_rotary_interleave(query.float(), cos, sin).to(query.dtype)
            key = self._apply_rotary_interleave(key.float(), cos, sin).to(key.dtype)
        else:
            query = self._apply_rotary_half(query.float(), cos, sin).to(query.dtype)
            key = self._apply_rotary_half(key.float(), cos, sin).to(key.dtype)
        
        return query, key


@lru_cache(1)
def get_rope_mla(
    head_size: int,
    max_position: int,
    base: float,
    interleave: bool = True,
):
    return RotaryEmbeddingMLA(head_size, max_position, base, interleave)
```

### 4.4 修改 `nanovllm/engine/model_runner.py`

#### 4.4.1 模型类选择

```python
def get_model_class(hf_config):
    """根据配置自动选择模型类"""
    model_type = getattr(hf_config, 'model_type', '')
    
    # MLA 模型 (Youtu-LLM)
    if model_type == 'youtu' or hasattr(hf_config, 'kv_lora_rank'):
        from nanovllm.models.youtu import YoutuForCausalLM
        return YoutuForCausalLM
    
    # MoE 模型
    num_experts = getattr(hf_config, 'num_experts', 0)
    if num_experts > 0:
        return Qwen3MoeForCausalLM
    
    # Dense 模型
    return Qwen3ForCausalLM
```

#### 4.4.2 KV Cache 分配

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    # 检测是否是 MLA 模型
    is_mla = hasattr(hf_config, 'kv_lora_rank')
    
    if is_mla:
        # MLA: KV Cache 维度 = kv_lora_rank + qk_rope_head_dim
        kv_cache_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
        # MLA 只需要 1 份 cache (不是 K 和 V 两份)
        block_bytes = hf_config.num_hidden_layers * self.block_size * kv_cache_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # shape: [num_layers, num_blocks, block_size, kv_cache_dim]
        self.kv_cache = torch.empty(
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            kv_cache_dim
        )
        
        # 分配给每层的 attention 模块
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'kv_cache') and hasattr(module, 'kv_lora_rank'):
                module.kv_cache = self.kv_cache[layer_id].view(-1, kv_cache_dim)
                layer_id += 1
    else:
        # 标准 MHA: 需要 K 和 V 两份 cache
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        self.kv_cache = torch.empty(
            2, 
            hf_config.num_hidden_layers, 
            config.num_kvcache_blocks, 
            self.block_size, 
            num_kv_heads, 
            head_dim
        )
        
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

### 4.5 修改 `nanovllm/utils/loader.py`

添加 MLA 权重加载支持：

```python
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                
                # === MLA 特殊权重处理 ===
                
                # kv_a_proj_with_mqa -> kv_a_proj
                if "kv_a_proj_with_mqa" in weight_name:
                    param_name = weight_name.replace("kv_a_proj_with_mqa", "kv_a_proj")
                    try:
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        continue
                    except AttributeError:
                        pass
                
                # ... (保留原有的 MoE 等处理逻辑)
                
                # 处理 packed_modules
                packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v = packed_modules_mapping[k]
                        if isinstance(v, tuple):
                            param_name = weight_name.replace(k, v[0])
                            shard_id = v[1]
                        else:
                            param_name = weight_name.replace(k, v)
                            shard_id = None
                        
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            if shard_id is not None:
                                weight_loader(param, loaded_weight, shard_id)
                            else:
                                weight_loader(param, loaded_weight)
                            packed = True
                            break
                        except AttributeError:
                            pass
                
                if packed:
                    continue
                
                # 默认加载
                try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                except AttributeError:
                    pass
```

---

## 5. 关键技术挑战

### 5.1 Decode 阶段的 KV 解压缩

**问题：** MLA 在 decode 阶段需要从压缩的 latent 解压出完整的 K, V，这需要：
1. 访问整个 KV Cache
2. 执行 kv_b_proj 矩阵乘法
3. 与 Paged Attention 兼容

**解决方案：**

```python
# 方案 A: 预解压 (简单但显存占用大)
# 将 cache 中的 latent 全部解压到临时 buffer，然后用 flash_attn_with_kvcache

# 方案 B: 融合 kernel (高效但复杂)
# 编写 Triton kernel，在 attention 计算时按需解压
# 类似 DeepSeek-V2 的 absorbed attention

# 方案 C: 修改 attention 计算
# 将 Q @ K^T 拆分为 Q_nope @ K_nope^T + Q_pe @ K_pe^T
# 其中 K_nope 需要解压，K_pe 直接从 cache 获取
```

### 5.2 Flash Attention 兼容性

**问题：** 标准 flash_attn 要求 Q, K, V 的 head_dim 相同，但 MLA 中：
- `qk_head_dim = 192`
- `v_head_dim = 128`

**解决方案：**

```python
# 方案 A: V padding
# 将 V 从 128 维 padding 到 192 维，计算后截取

# 方案 B: 自定义 Triton attention kernel
# 实现支持不同 head_dim 的 attention

# 方案 C: 使用 xformers 或其他支持的库
```

### 5.3 RoPE Interleave 模式

**问题：** Youtu-LLM 使用 `rope_interleave=True`，需要修改 RoPE 实现

**解决方案：** 见 4.3 节的 `RotaryEmbeddingMLA` 实现

---

## 6. 实现计划

### Phase 1: 基础功能 (优先级高)

| 任务 | 预计代码量 | 说明 |
|------|------------|------|
| 实现 `youtu.py` 模型结构 | ~300 行 | MLA 投影、decoder layer、model |
| 实现 `attention_mla.py` | ~200 行 | MLA attention + KV cache kernel |
| 修改 `rotary_embedding.py` | ~80 行 | 添加 MLA RoPE |
| 修改 `model_runner.py` | ~50 行 | KV cache 分配 |
| 修改 `loader.py` | ~30 行 | 权重映射 |

**Phase 1 目标：** Prefill 阶段正常工作

### Phase 2: Decode 支持 (优先级高)

| 任务 | 预计代码量 | 说明 |
|------|------------|------|
| 实现 KV 解压缩 kernel | ~150 行 | Triton kernel |
| 适配 flash_attn_with_kvcache | ~100 行 | 或自定义 decode attention |

**Phase 2 目标：** 完整生成功能

### Phase 3: 性能优化 (优先级中)

| 任务 | 说明 |
|------|------|
| Absorbed attention | Q 与 kv_b_proj 融合计算，减少中间结果 |
| CUDA Graph 支持 | 确保 MLA 计算图可捕获 |
| 张量并行优化 | 多 GPU 场景下的通信优化 |

---

## 7. 测试计划

### 7.1 单元测试

```python
def test_mla_rope():
    """测试 MLA RoPE 的正确性"""
    rope = RotaryEmbeddingMLA(head_size=64, max_position=1024, base=1600000, interleave=True)
    positions = torch.arange(10)
    q = torch.randn(10, 16, 64)
    k = torch.randn(10, 1, 64)
    q_rot, k_rot = rope(positions, q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

def test_mla_projection():
    """测试 MLA 投影的形状"""
    # 验证 Q, KV 两阶段投影输出形状正确

def test_mla_kvcache():
    """测试 MLA KV Cache 存取"""
    # 验证 latent 正确存入 cache
    # 验证 decode 时可正确读取
```

### 7.2 集成测试

```python
def test_youtu_generation():
    """测试完整生成流程"""
    llm = LLM("/path/to/Youtu", enforce_eager=True)
    outputs = llm.generate(["你好"], SamplingParams(temperature=0.6, max_tokens=256))
    assert len(outputs) > 0
```

### 7.3 数值精度验证

```python
def test_match_huggingface():
    """对比 Nano-vLLM 和 HuggingFace 实现的输出"""
    # 确保数值一致性
```

---

## 8. 风险与注意事项

1. **Flash Attention 维度限制**
   - head_dim 必须是 8 的倍数
   - Q/K/V 维度不匹配需要特殊处理

2. **CUDA Graph 兼容性**
   - Decode 阶段的 KV 解压缩可能影响 graph 捕获
   - 初期建议使用 `enforce_eager=True`

3. **数值精度**
   - 多次矩阵乘法可能累积误差
   - LayerNorm 建议用 float32 计算

4. **显存占用**
   - 虽然 KV Cache 压缩了，但 kv_b_proj 需要额外显存
   - 需要权衡解压时机和显存使用

---

## 9. 参考资料

1. [DeepSeek-V2 技术报告](https://arxiv.org/abs/2405.04434)
2. [DeepSeek-V2 HuggingFace 实现](https://huggingface.co/deepseek-ai/DeepSeek-V2)
3. [vLLM DeepSeek-V2 支持 PR](https://github.com/vllm-project/vllm/pull/4650)
4. [Flash Attention 文档](https://github.com/Dao-AILab/flash-attention)
