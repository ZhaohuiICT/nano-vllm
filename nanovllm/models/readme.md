这个文件实现了完整的 Qwen3 因果语言模型，采用自底向上的层次结构。

  ---
  整体架构图

  Qwen3ForCausalLM (顶层入口)
      │
      ├── Qwen3Model (模型主体)
      │       │
      │       ├── VocabParallelEmbedding (词嵌入)
      │       │
      │       ├── Qwen3DecoderLayer × N (N 层 Decoder)
      │       │       │
      │       │       ├── RMSNorm (input_layernorm)
      │       │       ├── Qwen3Attention (自注意力)
      │       │       ├── RMSNorm (post_attention_layernorm)
      │       │       └── Qwen3MLP (前馈网络)
      │       │
      │       └── RMSNorm (最终归一化)
      │
      └── ParallelLMHead (输出层，预测下一个 token)

  ---
  1. Qwen3Attention（自注意力层）- 第 14-87 行

  初始化部分（第 16-69 行）

  def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
      # 张量并行：将 heads 分配到多个 GPU
      tp_size = dist.get_world_size()  # 获取 GPU 数量
      self.num_heads = num_heads // tp_size        # 每个 GPU 的 Q heads 数
      self.num_kv_heads = num_kv_heads // tp_size  # 每个 GPU 的 KV heads 数（GQA）

  关键组件：

  ┌───────────────────┬───────────────────────────────────────────────────┐
  │ 组件              │ 作用                                              │
  ├───────────────────┼───────────────────────────────────────────────────┤
  │ `qkv_proj`        │ 将 hidden_states 投影为 Q、K、V（合并计算更高效） │
  ├───────────────────┼───────────────────────────────────────────────────┤
  │ `o_proj`          │ 输出投影，将注意力结果映射回 hidden_size          │
  ├───────────────────┼───────────────────────────────────────────────────┤
  │ `rotary_emb`      │ RoPE 位置编码                                     │
  ├───────────────────┼───────────────────────────────────────────────────┤
  │ `attn`            │ Flash Attention 计算                              │
  ├───────────────────┼───────────────────────────────────────────────────┤
  │ `q_norm / k_norm` │ Qwen3 特有的 QK 归一化（当无 bias 时）            │
  └───────────────────┴───────────────────────────────────────────────────┘

  Forward 流程（第 71-87 行）

  def forward(self, positions, hidden_states):
      # 1. QKV 投影
      qkv = self.qkv_proj(hidden_states)  # [seq_len, q_size + 2*kv_size]
      q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

      # 2. reshape 为多头格式
      q = q.view(-1, self.num_heads, self.head_dim)     # [seq, heads, dim]
      k = k.view(-1, self.num_kv_heads, self.head_dim)  # [seq, kv_heads, dim]
      v = v.view(-1, self.num_kv_heads, self.head_dim)

      # 3. QK 归一化（Qwen3 特有）
      if not self.qkv_bias:
          q = self.q_norm(q)
          k = self.k_norm(k)

      # 4. 应用 RoPE 位置编码
      q, k = self.rotary_emb(positions, q, k)

      # 5. 注意力计算（Flash Attention）
      o = self.attn(q, k, v)

      # 6. 输出投影
      output = self.o_proj(o.flatten(1, -1))
      return output

  数据流：
  hidden_states [seq, hidden]
         ↓ qkv_proj
  qkv [seq, q_size + 2*kv_size]
         ↓ split
  q [seq, heads, dim], k [seq, kv_heads, dim], v [seq, kv_heads, dim]
         ↓ RoPE
  q', k' (带位置信息)
         ↓ Flash Attention
  o [seq, heads, dim]
         ↓ o_proj
  output [seq, hidden]

  ---
  2. Qwen3MLP（前馈网络）- 第 90-116 行

  实现 SwiGLU 激活的 FFN：

  class Qwen3MLP(nn.Module):
      def __init__(self, hidden_size, intermediate_size, hidden_act):
          # gate 和 up 合并为一个矩阵（减少一次矩阵乘法）
          self.gate_up_proj = MergedColumnParallelLinear(
              hidden_size,
              [intermediate_size] * 2,  # gate 和 up 各 intermediate_size
          )
          self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
          self.act_fn = SiluAndMul()  # SiLU(gate) * up

      def forward(self, x):
          gate_up = self.gate_up_proj(x)  # [seq, 2 * intermediate intermediate]
          x = self.down_proj(x)            # [seq, hidden]
          return x

  SwiGLU 公式：
  output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

  数据流：
  x [seq, hidden]
         ↓ gate_up_proj
  [seq, 2 * intermediate]  →  split → gate [seq, inter], up [seq, inter]
         ↓ SiluAndMul
  SiLU(gate) * up → [seq, intermediate]
         ↓ down_proj
  output [seq, hidden]

  ---
  3. Qwen3DecoderLayer（单层 Decoder）- 第 119-158 行

  class Qwen3DecoderLayer(nn.Module):
      def __init__(self, config):
          self.self_attn = Qwen3Attention(...)
          self.mlp = Qwen3MLP(...)
          self.input_layernorm = RMSNorm(...)
          self.post_attention_layernorm = RMSNorm(...)

      def forward(self, positions, hidden_states, residual):
          # Pre-Norm 架构 + 残差连接

          # 第一个残差块：Attention
          if residual is None:
              hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
          else:
              hidden_states, residual = self.input_layernorm(hidden_states, residual)
          hidden_states = self.self_attn(positions, hidden_states)

          # 第二个残差块：MLP
          hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
          hidden_states = self.mlp(hidden_states)

          return hidden_states, residual

  残差连接逻辑：
  输入: x
         ↓
  LayerNorm(x) → Attention → output1
         ↓
  x + output1 = residual1  (第一个残差)
         ↓
  LayerNorm(residual1) → MLP → output2
         ↓
  residual1 + output2 = residual2  (第二个残差，传给下一层)

  ---
  4. Qwen3Model（模型主体）- 第 161-182 行

  class Qwen3Model(nn.Module):
      def __init__(self, config):
          self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
          self.layers = nn.ModuleList([
              Qwen3DecoderLayer(config)
              for _ in range(config.num_hidden_layers)  # N 层堆叠
          ])
          self.norm = RMSNorm(hidden_size)

      def forward(self, input_ids, positions):
          # 1. 词嵌入
          hidden_states = self.embed_tokens(input_ids)

          # 2. 逐层处理
          residual = None
          for layer in self.layers:
              hidden_states, residual = layer(positions, hidden_states, residual)

          # 3. 最终归一化
          hidden_states, _ = self.norm(hidden_states, residual)
          return hidden_states

  ---
  5. Qwen3ForCausalLM（顶层入口）- 第 185-215 行

  class Qwen3ForCausalLM(nn.Module):
      # 权重映射：HuggingFace 原始权重名 → 合并后的权重名
      packed_modules_mapping = {
          "q_proj": ("qkv_proj", "q"),   # q_proj 权重加载到 qkv_proj 的 q 部分
          "k_proj": ("qkv_proj", "k"),
          "v_proj": ("qkv_proj", "v"),
          "gate_proj": ("gate_up_proj", 0),
          "up_proj": ("gate_up_proj", 1),
      }

      def __init__(self, config):
          self.model = Qwen3Model(config)
          self.lm_head = ParallelLMHead(vocab_size, hidden_size)

          # 权重共享：embedding 和 lm_head 共用权重
          if config.tie_word_embeddings:
              self.lm_head.weight.data = self.model.embed_tokens.weight.data

      def forward(self, input_ids, positions):
          return self.model(input_ids, positions)  # 返回 hidden_states

      def compute_logits(self, hidden_states):
          return self.lm_head(hidden_states)  # hidden → vocab 概率分布

  packed_modules_mapping 的作用：

  HuggingFace 原始模型权重是分开的（q_proj, k_proj, v_proj），但 Nano-vLLM 为了效率将它们合并为qkv_proj。这个映射告诉权重加载器如何正确加载。

  ---
  完整推理流程

  input_ids: [batch, seq_len]  (token IDs)
  positions: [batch, seq_len]  (位置索引)
                  ↓
          ┌───────────────────┐
          │  embed_tokens     │  词嵌入
          └───────────────────┘
                  ↓
          hidden_states: [batch, seq, hidden]
                  ↓
          ┌───────────────────┐
          │  Decoder Layer 1  │ ←─┐
          └───────────────────┘   │
                  ↓               │ × N 层
          ┌───────────────────┐   │
          │  Decoder Layer N  │ ←─┘
          └───────────────────┘
                  ↓
          ┌───────────────────┐
          │  Final RMSNorm    │
          └───────────────────┘
                  ↓
          hidden_states: [batch, seq, hidden]
                  ↓
          ┌───────────────────┐
          │  lm_head          │  线性层 hidden → vocab_size
          └───────────────────┘
                  ↓
          logits: [batch, seq, vocab_size]  (下一个 token 的概率分布)

  ---
  关键设计特点

  ┌──────────────┬────────────────────────────────────────────────┐
  │ 特点         │ 说明                                           │
  ├──────────────┼────────────────────────────────────────────────┤
  │ **GQA**      │ `num_kv_heads < num_heads`，节省 KV Cache 内存 │
  ├──────────────┼────────────────────────────────────────────────┤
  │ **QK Norm**  │ Qwen3 特有，提升训练稳定性                     │
  ├──────────────┼────────────────────────────────────────────────┤
  │ **SwiGLU**   │ 比 ReLU/GELU 效果更好的激活函数                │
  ├──────────────┼────────────────────────────────────────────────┤
  │ **Pre-Norm** │ LayerNorm 在子层之前，训练更稳定               │
  ├──────────────┼────────────────────────────────────────────────┤
  │ **张量并行** │ 所有线性层支持多 GPU 切分                      │
  ├──────────────┼────────────────────────────────────────────────┤
  │ **权重合并** │ QKV 合并、gate/up 合并，减少 kernel launch     │
  └──────────────┴────────────────────────────────────────────────┘
