"""
对比 HuggingFace Transformers / vLLM / Nano-vLLM 在 Qwen3-MoE 上的推理结果

用法:
    # 用 HuggingFace transformers 做基准（无需安装 vLLM）
    python bench_vllm_vs_nano.py --backend hf
    python bench_vllm_vs_nano.py --backend nano
    python bench_vllm_vs_nano.py --backend both-hf   # hf + nano 对比

    # 如果装了 vLLM
    python bench_vllm_vs_nano.py --backend vllm
    python bench_vllm_vs_nano.py --backend both-vllm  # vllm + nano 对比
"""
import argparse
import json
import torch
from transformers import AutoTokenizer


MODEL_PATH = "/apdcephfs_qy3/share_7435715/demianwang/models/Qwen/Qwen3-30B-A3B-Instruct-2507"

TEST_PROMPTS = [
    "introduce yourself",
    "what is 1 + 1?",
    "Explain the concept of sparse mixture of experts in transformer models in 3 sentences.",
    "Write a Python function to compute fibonacci numbers.",
]

# greedy decoding 保证确定性
MAX_TOKENS = 256


def build_formatted_prompts(prompts, tokenizer):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


def run_hf(prompts, tokenizer):
    """使用 HuggingFace transformers 进行推理（作为正确性基准）"""
    from transformers import AutoModelForCausalLM

    print("=" * 60)
    print("Running HuggingFace transformers inference...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    formatted_prompts = build_formatted_prompts(prompts, tokenizer)

    results = []
    for prompt, formatted in zip(prompts, formatted_prompts):
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,  # greedy
                temperature=None,
                top_p=None,
            )

        new_ids = output_ids[0, input_len:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "text": text,
            "num_tokens": len(new_ids),
            "token_ids": new_ids[:50],
        })
        print(f"\nPrompt: {prompt}")
        print(f"Response ({len(new_ids)} tokens): {text[:200]}...")

    del model
    torch.cuda.empty_cache()
    return results


def run_vllm(prompts, tokenizer):
    """使用 vLLM 进行推理"""
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("Running vLLM inference...")
    print("=" * 60)

    llm = LLM(
        MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

    formatted_prompts = build_formatted_prompts(prompts, tokenizer)
    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        results.append({
            "prompt": prompt,
            "text": text,
            "num_tokens": len(token_ids),
            "token_ids": list(token_ids[:50]),
        })
        print(f"\nPrompt: {prompt}")
        print(f"Response ({len(token_ids)} tokens): {text[:200]}...")

    del llm
    torch.cuda.empty_cache()
    return results


def run_nano(prompts, tokenizer):
    """使用 Nano-vLLM 进行推理"""
    from nanovllm import LLM, SamplingParams

    print("=" * 60)
    print("Running Nano-vLLM inference...")
    print("=" * 60)

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # Nano-vLLM 不支持 temperature=0，用极小值近似 greedy
    sampling_params = SamplingParams(
        temperature=1e-9,
        max_tokens=MAX_TOKENS,
    )

    formatted_prompts = build_formatted_prompts(prompts, tokenizer)
    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for prompt, output in zip(prompts, outputs):
        text = output["text"]
        token_ids = output["token_ids"]
        results.append({
            "prompt": prompt,
            "text": text,
            "num_tokens": len(token_ids),
            "token_ids": list(token_ids[:50]),
        })
        print(f"\nPrompt: {prompt}")
        print(f"Response ({len(token_ids)} tokens): {text[:200]}...")

    del llm
    torch.cuda.empty_cache()
    return results


def compare_results(ref_results, nano_results, ref_name="Reference"):
    """对比两个后端的结果"""
    print("\n" + "=" * 60)
    print(f"COMPARISON: {ref_name} vs Nano-vLLM")
    print("=" * 60)

    all_match = True
    for i, (rr, nr) in enumerate(zip(ref_results, nano_results)):
        print(f"\n{'─' * 60}")
        print(f"Prompt {i+1}: {rr['prompt']}")
        print(f"{'─' * 60}")

        # 对比 token ids
        r_ids = rr["token_ids"]
        n_ids = nr["token_ids"]
        match_len = 0
        for a, b in zip(r_ids, n_ids):
            if a == b:
                match_len += 1
            else:
                break

        total = min(len(r_ids), len(n_ids))
        print(f"Token ID match: {match_len}/{total} (前 {total} 个 token)")

        if match_len < total:
            all_match = False
            print(f"  First mismatch at token {match_len}: {ref_name}={r_ids[match_len] if match_len < len(r_ids) else 'N/A'}, Nano={n_ids[match_len] if match_len < len(n_ids) else 'N/A'}")

        # 对比文本
        if rr["text"] == nr["text"]:
            print("Text: EXACT MATCH ✓")
        else:
            print("Text: DIFFERENT")
            min_len = min(len(rr["text"]), len(nr["text"]))
            diff_pos = min_len
            for j in range(min_len):
                if rr["text"][j] != nr["text"][j]:
                    diff_pos = j
                    break
            print(f"  First diff at char {diff_pos}")
            print(f"  {ref_name:6s}: ...{rr['text'][max(0,diff_pos-20):diff_pos+50]}...")
            print(f"  {'Nano':6s}: ...{nr['text'][max(0,diff_pos-20):diff_pos+50]}...")

        print(f"Length: {ref_name}={rr['num_tokens']} tokens, Nano={nr['num_tokens']} tokens")

    print(f"\n{'=' * 60}")
    if all_match:
        print("RESULT: ALL TOKENS MATCH ✓")
    else:
        print("RESULT: SOME DIFFERENCES FOUND (may be due to floating point precision)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3-MoE inference across backends")
    parser.add_argument("--backend", choices=["hf", "vllm", "nano", "both-hf", "both-vllm"], default="both-hf",
                        help="hf=transformers, vllm=vLLM, nano=Nano-vLLM, both-hf=hf+nano, both-vllm=vllm+nano")
    parser.add_argument("--save", action="store_true", help="Save results to JSON files")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    ref_results = None
    nano_results = None
    ref_name = None

    if args.backend in ("hf", "both-hf"):
        ref_results = run_hf(TEST_PROMPTS, tokenizer)
        ref_name = "HF"
        if args.save:
            with open("results_hf.json", "w") as f:
                json.dump(ref_results, f, ensure_ascii=False, indent=2)
            print("\nSaved to results_hf.json")

    if args.backend in ("vllm", "both-vllm"):
        ref_results = run_vllm(TEST_PROMPTS, tokenizer)
        ref_name = "vLLM"
        if args.save:
            with open("results_vllm.json", "w") as f:
                json.dump(ref_results, f, ensure_ascii=False, indent=2)
            print("\nSaved to results_vllm.json")

    if args.backend in ("nano", "both-hf", "both-vllm"):
        nano_results = run_nano(TEST_PROMPTS, tokenizer)
        if args.save:
            with open("results_nano.json", "w") as f:
                json.dump(nano_results, f, ensure_ascii=False, indent=2)
            print("\nSaved to results_nano.json")

    if ref_results and nano_results:
        compare_results(ref_results, nano_results, ref_name)


if __name__ == "__main__":
    main()
