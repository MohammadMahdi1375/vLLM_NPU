"""
DFlash benchmark CLIENT for a *running* vLLM server (V1).

Talks to the OpenAI-compatible /v1/completions endpoint for generation, and reads
acceptance length from the Prometheus /metrics endpoint with a before/after delta
(same pattern your sglang script used on /server_info). The server stays up between
runs — no reloading the model.

Launch the server once (see the launch command), then run this as many times as you want.
"""
import argparse
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset


DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "split": "test",
        "format": lambda x: (
            f"{x['question']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "split": "test",
        "format": lambda x: (
            f"{x['problem']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "split": "test",
        "format": lambda x: (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{x['prompt']}\n```"
        ),
    },
    # MT-Bench. The HF dataset stores `prompt` as a LIST of turns (it's the one
    # multi-turn benchmark). z-lab's server benchmark (_run_server) does NOT replay
    # the conversation — it uses only `turns[0]`. We match that here by taking the
    # first turn, so numbers line up with z-lab's vLLM/SGLang results and flow through
    # the same single-turn streaming + /metrics path as the other datasets.
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "split": "train",
        "format": lambda x: x["prompt"][0],
    },
}


def load_data(name, n):
    cfg = DATASETS[name]
    ds = load_dataset(*cfg["load_args"], split=cfg["split"])
    items = [{"text": cfg["format"](x)} for x in ds]
    random.seed(42)
    random.shuffle(items)
    return items if n is None else items[:n]


# ---- spec-decode counters from the Prometheus /metrics text endpoint ----
# Counters are exposed with a _total suffix; per-pos carries a position="N" label.
_NUM_RE = re.compile(r"\}?\s+([0-9eE.+-]+)\s*$")


def _sum_counter(text, base_name):
    """Sum all label-series for a counter (handles the model_name label)."""
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(base_name + "{") or line.split("{")[0] == base_name:
            m = _NUM_RE.search(line)
            if m:
                total += float(m.group(1))
                found = True
    return total if found else None


def _per_pos(text, base_name):
    """Return {position_index: count} for a per-position counter."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(base_name):
            continue
        pos_m = re.search(r'position="(\d+)"', line)
        val_m = _NUM_RE.search(line)
        if pos_m and val_m:
            i = int(pos_m.group(1))
            out[i] = out.get(i, 0.0) + float(val_m.group(1))
    return out


def get_spec_metrics(base_url):
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        r.raise_for_status()
        t = r.text
    except Exception as e:
        print(f"WARN: could not fetch /metrics: {e}")
        return None
    return {
        "num_drafts": _sum_counter(t, "vllm:spec_decode_num_drafts_total"),
        "num_draft_tokens": _sum_counter(t, "vllm:spec_decode_num_draft_tokens_total"),
        "num_accepted_tokens": _sum_counter(t, "vllm:spec_decode_num_accepted_tokens_total"),
        "per_pos": _per_pos(t, "vllm:spec_decode_num_accepted_tokens_per_pos_total"),
    }


def send_one(base_url, model, prompt, max_new_tokens, temperature, top_p, top_k, timeout):
    """Streaming /v1/completions request. Captures TTFT, per-chunk ITL, and output_len
    from the final usage chunk."""
    import json as _json
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k  # vLLM accepts top_k as an extra field

    st = time.perf_counter()
    ttft = 0.0
    most_recent_ts = st
    itls = []
    output_len = 0
    text_out = []

    with requests.post(
        f"{base_url}/v1/completions", json=payload, timeout=timeout, stream=True,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[5:].lstrip()
            if line == "[DONE]":
                continue
            try:
                data = _json.loads(line)
            except Exception:
                continue
            ts = time.perf_counter()
            choices = data.get("choices") or []
            if choices and choices[0].get("text"):
                text_out.append(choices[0]["text"])
                if ttft == 0.0:
                    ttft = ts - st
                else:
                    itls.append((ts - most_recent_ts) * 1000.0)  # ms
                most_recent_ts = ts
            if data.get("usage"):
                output_len = int(data["usage"].get("completion_tokens", 0))

    e2e = time.perf_counter() - st
    return {
        "text": "".join(text_out),
        "e2e_latency": e2e,
        "ttft_ms": ttft * 1000.0,
        "itls": itls,
        "output_len": output_len,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", required=True,
                    help="Path/name as passed to `vllm serve` (the served model id).")
    ap.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="If omitted, benchmarks the entire dataset.")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=3600)
    args = ap.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.num_prompts is None:
        print(f"Loading FULL {args.dataset} dataset (plus {args.concurrency} warmup)...")
        raw = load_data(args.dataset, None)
    else:
        print(f"Loading {args.num_prompts} samples (plus {args.concurrency} warmup)...")
        raw = load_data(args.dataset, args.num_prompts + args.concurrency)
    print(f"  Loaded {len(raw)} samples")

    prompts = []
    for item in raw:
        messages = [{"role": "user", "content": item["text"]}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompts.append(chat_text)

    # flush prefix cache so warmup doesn't help later requests
    try:
        requests.post(f"{args.base_url}/reset_prefix_cache", timeout=30)
    except Exception:
        pass

    bs = max(args.concurrency, 1)
    if len(prompts) > bs:
        print(f"Warmup ({bs} samples)...")
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(
                lambda p: send_one(args.base_url, args.model, p, args.max_new_tokens,
                                   args.temperature, args.top_p, args.top_k, args.timeout_s),
                prompts[:bs]))
        prompts = prompts[bs:]

    # snapshot acceptance counters AFTER warmup
    m_before = get_spec_metrics(args.base_url)
    if m_before:
        print(f"Spec metrics BEFORE: drafts={m_before['num_drafts']} "
              f"accepted={m_before['num_accepted_tokens']}")

    print(f"Benchmarking {len(prompts)} prompts, concurrency={args.concurrency}...")
    start = time.perf_counter()
    total_tokens = 0
    e2e_latencies = []
    all_itls = []
    all_ttfts = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(
            send_one, args.base_url, args.model, p, args.max_new_tokens,
            args.temperature, args.top_p, args.top_k, args.timeout_s,
        ): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futures), total=len(prompts)):
            out = fut.result()
            total_tokens += out["output_len"]
            e2e_latencies.append(out["e2e_latency"])
            all_itls.extend(out["itls"])
            if out["ttft_ms"] > 0:
                all_ttfts.append(out["ttft_ms"])

    elapsed = time.perf_counter() - start
    m_after = get_spec_metrics(args.base_url)

    # ---- acceptance length from counter delta ----
    accept_length = float("nan")
    accept_rate = float("nan")
    d_drafts = d_acc = d_draft_tok = 0.0
    if m_before and m_after and m_after["num_drafts"] is not None:
        d_drafts = m_after["num_drafts"] - m_before["num_drafts"]
        d_acc = m_after["num_accepted_tokens"] - m_before["num_accepted_tokens"]
        if m_after["num_draft_tokens"] is not None:
            d_draft_tok = m_after["num_draft_tokens"] - m_before["num_draft_tokens"]
        if d_drafts > 0:
            accept_length = 1.0 + d_acc / d_drafts
        if d_draft_tok > 0:
            accept_rate = d_acc / d_draft_tok

    print()
    print("=" * 60)
    print(f"Dataset:               {args.dataset}")
    print(f"Prompts:               {len(prompts)}")
    print(f"Concurrency:           {args.concurrency}")
    print(f"Wall clock:            {elapsed:.2f} s")
    print(f"Total output tokens:   {total_tokens}")
    print()
    print(f"** Throughput:         {total_tokens/elapsed:.2f} tok/s **")
    print(f"** Accept length:      {accept_length:.3f} (mean, incl. bonus) **")
    print(f"   Accept rate:        {100*accept_rate:.2f}%")
    print(f"   num_drafts:         {int(d_drafts)}")
    print(f"   num_draft_tokens:   {int(d_draft_tok)}")
    print(f"   num_accepted_tokens:{int(d_acc)}")
    if e2e_latencies:
        print(f"   E2E latency mean:   {statistics.mean(e2e_latencies):.2f} s")
    if all_ttfts:
        print(f"   Mean TTFT (ms):     {statistics.mean(all_ttfts):.2f}")
    if all_itls:
        print(f"   Mean ITL (ms):      {statistics.mean(all_itls):.2f}")
        print(f"   Median ITL (ms):    {statistics.median(all_itls):.2f}")

    if m_before and m_after and m_after.get("per_pos") and m_before.get("per_pos") and d_drafts > 0:
        print("   Per-position accept rate:")
        for i in sorted(m_after["per_pos"]):
            c = m_after["per_pos"].get(i, 0) - m_before["per_pos"].get(i, 0)
            print(f"     pos {i}: {100*c/d_drafts:.2f}%")
    if m_after and m_after["num_drafts"] in (None, 0):
        print()
        print("NOTE: no spec-decode counters from /metrics. Server may not be on the V1")
        print("      engine, or this vllm-ascend build feeds acceptance elsewhere.")
    print("=" * 60)


if __name__ == "__main__":
    main()