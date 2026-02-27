import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul


class MoeRouter(nn.Module):
    """
    Top-K 路由器
    
    对每个 token 计算其应该路由到的 top-k 个专家
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
        
        Returns:
            router_weights: [num_tokens, top_k] 归一化的路由权重
            selected_experts: [num_tokens, top_k] 选中的专家索引
        """
        # 计算路由 logits: [num_tokens, num_experts]
        router_logits = F.linear(hidden_states, self.weight)
        
        # Softmax 归一化 (用 float32 保证数值稳定)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Top-K 选择
        router_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 归一化 top-k 权重使其和为 1
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        
        return router_weights.to(hidden_states.dtype), selected_experts


class MoeExperts(nn.Module):
    """
    专家集合
    
    所有专家权重合并为 3D 张量，提高内存局部性
    每个专家结构: gate_proj + up_proj -> SiLU -> down_proj (SwiGLU)
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # 合并 gate 和 up 为一个张量: [num_experts, 2 * intermediate_size, hidden_size]
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        # down 投影: [num_experts, hidden_size, intermediate_size]
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
            selected_experts: [num_tokens, top_k] 每个 token 选中的专家索引
            router_weights: [num_tokens, top_k] 对应的路由权重
        
        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.size(0)
        final_output = torch.zeros_like(hidden_states)
        
        # 创建专家掩码: [num_experts, num_tokens, top_k]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 0, 1)  # [num_experts, num_tokens, top_k]
        
        # 遍历每个专家
        for expert_idx in range(self.num_experts):
            # 找到路由到该专家的 (token_idx, top_k_idx)
            token_indices, topk_indices = torch.where(expert_mask[expert_idx])
            
            if token_indices.numel() == 0:
                continue
            
            # 获取分配给该专家的 tokens
            expert_input = hidden_states[token_indices]  # [num_assigned, hidden_size]
            
            # 专家计算: gate-up -> activation -> down
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_idx])
            expert_output = self.act_fn(gate_up)
            expert_output = F.linear(expert_output, self.down_proj[expert_idx])
            
            # 加权: 乘以对应的路由权重
            weights = router_weights[token_indices, topk_indices].unsqueeze(-1)
            weighted_output = expert_output * weights
            
            # 累加到最终输出
            final_output.index_add_(0, token_indices, weighted_output)
        
        return final_output


class SparseMoeBlock(nn.Module):
    """
    稀疏 MoE 块
    
    组合 Router 和 Experts，替代标准 MLP
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.router = MoeRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = MoeExperts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_tokens, hidden_size]
        
        Returns:
            output: [num_tokens, hidden_size]
        """
        original_shape = hidden_states.shape
        # 展平为 [num_tokens, hidden_size]
        hidden_states = hidden_states.view(-1, original_shape[-1])
        
        # 路由决策
        router_weights, selected_experts = self.router(hidden_states)
        
        # 专家计算
        output = self.experts(hidden_states, selected_experts, router_weights)
        
        # 恢复原始形状
        return output.view(original_shape)
