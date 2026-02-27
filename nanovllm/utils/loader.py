import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_moe_expert_weight(
    experts_module: nn.Module,
    weight_name: str,
    loaded_weight: torch.Tensor,
):
    """
    加载 MoE 专家权重
    
    HuggingFace 权重格式:
        model.layers.{layer}.mlp.experts.{expert_id}.gate_proj.weight
        model.layers.{layer}.mlp.experts.{expert_id}.up_proj.weight
        model.layers.{layer}.mlp.experts.{expert_id}.down_proj.weight
    
    Nano-vLLM 格式:
        experts.gate_up_proj: [num_experts, 2*intermediate, hidden]
        experts.down_proj: [num_experts, hidden, intermediate]
    """
    # 解析专家 ID 和权重类型
    # 匹配 experts.{expert_id}.{gate_proj|up_proj|down_proj}.weight
    match = re.search(r'experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', weight_name)
    if not match:
        return False
    
    expert_id = int(match.group(1))
    proj_type = match.group(2)
    
    if proj_type == "gate_proj":
        # gate_proj 放在 gate_up_proj 的前半部分
        intermediate_size = loaded_weight.size(0)
        experts_module.gate_up_proj.data[expert_id, :intermediate_size, :] = loaded_weight
    elif proj_type == "up_proj":
        # up_proj 放在 gate_up_proj 的后半部分
        intermediate_size = loaded_weight.size(0)
        experts_module.gate_up_proj.data[expert_id, intermediate_size:, :] = loaded_weight
    elif proj_type == "down_proj":
        experts_module.down_proj.data[expert_id] = loaded_weight
    
    return True


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                
                # 尝试处理 MoE 专家权重
                if ".experts." in weight_name and "mlp.experts" in weight_name:
                    # 获取对应的 experts 模块
                    # weight_name 格式: model.layers.{layer}.mlp.experts.{id}.xxx
                    # 需要找到 model.layers.{layer}.mlp.experts 模块
                    match = re.search(r'(model\.layers\.\d+\.mlp)\.experts\.\d+', weight_name)
                    if match:
                        mlp_path = match.group(1)
                        try:
                            mlp_module = model.get_submodule(mlp_path)
                            # SparseMoeBlock 的 experts 属性
                            if hasattr(mlp_module, 'experts'):
                                if load_moe_expert_weight(mlp_module.experts, weight_name, loaded_weight):
                                    continue
                        except AttributeError:
                            pass
                
                # 尝试处理 MoE Router 权重
                if ".gate.weight" in weight_name and "mlp.gate" in weight_name:
                    # weight_name: model.layers.{layer}.mlp.gate.weight
                    # 对应: model.layers.{layer}.mlp.router.weight
                    param_name = weight_name.replace("mlp.gate.weight", "mlp.router.weight")
                    try:
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        continue
                    except AttributeError:
                        pass
                
                # 处理 packed_modules（QKV 合并、gate/up 合并）
                packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, loaded_weight, shard_id)
                            packed = True
                            break
                        except AttributeError:
                            pass
                
                if packed:
                    continue
                
                # 默认权重加载
                try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                except AttributeError:
                    # 权重名不匹配，跳过（可能是未使用的权重）
                    pass
