"""
Qwen3 MoE 模型离线推理测试脚本
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 修改为你的模型路径
    model_path = "/apdcephfs_qy3/share_7435715/demianwang/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    # enforce_eager=True: 禁用 CUDA Graph（MoE 朴素实现可能不兼容）
    # tensor_parallel_size=1: 单 GPU
    llm = LLM(
        model_path, 
        enforce_eager=True, 
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,  # GPU 显存利用率
    )
    
    print("Model loaded successfully!")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256,
    )
    
    # 测试 prompts
    prompts = [
        "你认为大模型时代，算法工程师如何提升自身价值？",
        "what is 1 + 1?",
    ]
    
    # 应用聊天模板
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    print("\n" + "=" * 50)
    print("Starting generation...")
    print("=" * 50)
    
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # 打印结果
    for prompt, output in zip(prompts, outputs):
        print("\n" + "-" * 50)
        print(f"Prompt: {prompt}")
        print(f"Response: {output['text']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
