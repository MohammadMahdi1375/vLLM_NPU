#!/usr/bin/env python3
"""Independent HTTP evaluation client for an already running vLLM server.

Usage: python evaluator.py --base-url http://127.0.0.1:8100 --output results/dflash
This file does not launch vLLM. No server script imports or executes this file.
Use the same prompt manifest for all four model endpoints.


Default protocol: Appendix E of ReTrace v2 (2608.29748), one request at a
time, no thinking, temperatures 0 and 1, seed 42, at most 8192 new tokens.
Native NPU execution and stored-response training are experimental adaptations.
No generated benchmark program is executed and no task-accuracy score is claimed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_TARGET = (
    "/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/"
    "1cfa9a7208912126459214e8b04321603b3df60c"
)
COUNTS = {
    "gsm8k": 128,
    "math500": 128,
    "humaneval": 164,
    "lcb": 128,
    "aime25": 30,
    "mtbench": 80,
    "alpaca": 128,
}
DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "default", "test"),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "aime25": ("MathArena/aime_2025", "default", "train"),
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", "default", "train"),
    "alpaca": ("tatsu-lab/alpaca", "default", "train"),
}
SPEC_COUNTERS = {
    "vllm:spec_decode_num_drafts": "rounds",
    "vllm:spec_decode_num_draft_tokens": "drafted",
    "vllm:spec_decode_num_accepted_tokens": "accepted",
    "vllm:request_success": "finished",
    "vllm:generation_tokens": "generated",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temp, path)


def names_from(values):
    aliases = {
        "mt-bench": "mtbench",
        "livecodebench": "lcb",
        "aime2025": "aime25",
        "math-500": "math500",
    }
    names = [aliases.get(n.lower(), n.lower()) for s in values for n in s.split(",")]
    if names == ["all"]:
        return list(COUNTS)
    if len(set(names)) != len(names) or any(n not in COUNTS for n in names):
        raise ValueError(f"Choose distinct datasets from {list(COUNTS)}, or all")
    return names


def format_prompt(name, row):
    if name in ("gsm8k", "math500", "aime25"):
        question = (
            row.get("question")
            if name == "gsm8k"
            else row.get("problem", row.get("question"))
        )
        if not isinstance(question, str):
            raise ValueError(f"{name}: missing question/problem")
        return (
            question
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    if name == "humaneval":
        return (
            "Write a solution to the following problem and make sure that it passes the tests:\n```python\n"
            + row["prompt"]
            + "\n```"
        )
    if name == "mtbench":
        return (row.get("turns") or row["prompt"])[0]
    if name == "alpaca":
        return row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
    if name == "lcb":
        starter = row.get("starter_code", "")
        return (
            row["question_content"]
            + "\n\n"
            + (
                "Complete the following Python starter code.\n```python\n"
                + starter
                + "\n```"
                if starter
                else "Write a Python program that reads from standard input and writes to standard output."
            )
            + "\nReturn your solution in a Python code block."
        )
    raise ValueError(name)


def load_rows(name, source, offline, lcb_release):
    if source:
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            from datasets import DatasetDict, load_from_disk

            rows = load_from_disk(str(path))
            if isinstance(rows, DatasetDict):
                split = DATASETS.get(name, ("", "", "test"))[2]
                rows = rows[split]
            return rows, {"path": str(path), "fingerprint": rows._fingerprint}, False
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(
                "Local source must be JSONL, a JSON row list, or datasets.save_to_disk directory"
            )
        return rows, {"path": str(path), "sha256": file_hash(path)}, True
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if name == "lcb":
        from huggingface_hub import hf_hub_download

        rows, hashes = [], {}
        for filename in [
            "test.jsonl",
            *[f"test{i}.jsonl" for i in range(2, lcb_release + 1)],
        ]:
            local = hf_hub_download(
                "livecodebench/code_generation_lite",
                filename,
                repo_type="dataset",
                local_files_only=offline,
            )
            hashes[filename] = file_hash(local)
            with open(local) as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            k: row.get(k, "")
                            for k in (
                                "question_content",
                                "starter_code",
                                "question_id",
                                "contest_date",
                            )
                        }
                    )
        return (
            rows,
            {
                "repo": "livecodebench/code_generation_lite",
                "release": lcb_release,
                "source_sha256": hashes,
            },
            False,
        )
    from datasets import load_dataset

    repo, config, split = DATASETS[name]
    # Do not perform a separate HfApi.dataset_info call: it breaks cached/offline runs.
    rows = load_dataset(repo, config, split=split)
    return (
        rows,
        {
            "repo": repo,
            "config": config,
            "split": split,
            "fingerprint": rows._fingerprint,
        },
        False,
    )


def token_list(value):
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized prompt")
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or any(type(x) is not int or x < 0 for x in value)
    ):
        raise ValueError("Expected a nonempty list of nonnegative token IDs")
    return value


def prepare_manifest(args, names):
    """Freeze formatted token IDs once; reuse this exact file on all four servers."""
    if args.manifest.exists():
        return json.loads(args.manifest.read_text())
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    sources = {}
    for item in args.source:
        name, sep, path = item.partition("=")
        if not sep or name not in COUNTS or name in sources:
            raise ValueError("--source must be a unique DATASET=LOCAL_PATH mapping")
        sources[name] = path
    result = {
        "format": 2,
        "seed": args.seed,
        "thinking": False,
        "target": str(args.target),
        "target_config_sha256": file_hash(args.target / "config.json"),
        "tokenizer_sha256": {
            p.name: file_hash(p)
            for p in sorted(args.target.iterdir())
            if p.name
            in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "special_tokens_map.json",
            )
        },
        "datasets": {},
        "author_row_ids_available": False,
        "choices": {
            "selection": "seeded shuffle; all rows when full set requested",
            "mtbench": "first user turn only; paper does not specify turn policy",
            "lcb_release": f"release_v{args.lcb_release}",
            "formatter_sha256": file_hash(__file__),
        },
    }
    for name in names:
        rows, source_info, local_json = load_rows(
            name, sources.get(name), args.offline, args.lcb_release
        )
        indices = list(range(len(rows)))
        if len(rows) > COUNTS[name]:
            random.Random(args.seed).shuffle(indices)
        indices = indices[: COUNTS[name]]
        selected = []
        for i in indices:
            row = rows[i]
            if row.get("messages"):
                prompt = ""
            elif (
                local_json
                and isinstance(row.get("prompt"), str)
                and set(row) <= {"id", "prompt", "answer"}
            ):
                prompt = row["prompt"]
            else:
                prompt = row.get("user_prompt") or format_prompt(name, row)
            messages = row.get("messages") or [{"role": "user", "content": prompt}]
            tokens = token_list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            selected.append(
                {
                    "source_row": i,
                    "source_id": str(
                        row.get("id", row.get("task_id", row.get("question_id", i)))
                    ),
                    "prompt_token_ids": tokens,
                    "sha256": digest(tokens),
                }
            )
        if not selected:
            raise ValueError(f"No rows for {name}")
        result["datasets"][name] = {
            "source": source_info,
            "source_rows": len(rows),
            "requests": len(selected),
            "results": selected,
        }
        print(f"Prepared {name}: {len(selected)} prompts", flush=True)
    result["complete_table2_counts"] = all(
        len(result["datasets"].get(n, {}).get("results", [])) == c
        for n, c in COUNTS.items()
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing frozen set from another evaluator.
    with args.manifest.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return result


def select_prompts(manifest, names, limit):
    if manifest.get("thinking") is not False:
        raise ValueError("Manifest must explicitly use thinking=False")
    chosen = {}
    for name in names:
        if name not in manifest.get("datasets", {}):
            raise ValueError(
                f"Manifest lacks {name}; use another --manifest or select its datasets"
            )
        rows = manifest["datasets"][name]["results"]
        rows = rows[:limit] if limit else rows
        if not rows:
            raise ValueError(f"Empty prompt set: {name}")
        for row in rows:
            ids = token_list(row["prompt_token_ids"])
            if row.get("sha256") != digest(ids):
                raise ValueError(f"Manifest token hash mismatch in {name}")
        chosen[name] = rows
    return chosen


def normalize_url(value):
    p = urlsplit(value.rstrip("/"))
    if (
        p.scheme not in ("http", "https")
        or not p.netloc
        or p.query
        or p.fragment
        or p.username
    ):
        raise ValueError(
            "Use http(s)://host:port[/v1], without credentials or query parameters"
        )
    path = p.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((p.scheme, p.netloc, path, "", "")).rstrip("/")


def sampling(temperature, max_tokens, seed, eos):
    return {
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "top_k": 1 if temperature == 0 else 20,
        "max_tokens": max_tokens,
        "seed": seed,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "ignore_eos": False,
        "stop_token_ids": eos,
        "skip_special_tokens": True,
    }


def validate_spec(stats):
    def integer(key):
        value = stats.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid per-request speculative metric {key}: {value}")
        return value

    rounds, accepted, drafted, k = (
        integer(key)
        for key in (
            "num_spec_steps",
            "num_accepted_draft_tokens",
            "num_draft_tokens",
            "num_spec_tokens",
        )
    )
    if (
        accepted > drafted
        or drafted > rounds * k
        or (rounds and (k < 1 or drafted < rounds))
    ):
        raise ValueError(f"Inconsistent speculative counters: {stats}")
    hist = stats.get("acceptance_histogram")
    if (
        not isinstance(hist, list)
        or len(hist) != k + 1
        or any(type(v) is not int or v < 0 for v in hist)
        or sum(hist) != rounds
        or sum(i * n for i, n in enumerate(hist)) != accepted
    ):
        raise ValueError("Acceptance histogram disagrees with per-request counters")
    tau = 1 + accepted / rounds if rounds else None
    if rounds and not math.isclose(
        stats.get("mean_acceptance_length", -1), tau, abs_tol=1e-6
    ):
        raise ValueError("Server acceptance length disagrees with raw counts")
    return {
        "rounds": rounds,
        "accepted": accepted,
        "drafted": drafted,
        "k": k,
        "histogram": hist,
        "acceptance_length": tau,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


class Client:
    def __init__(self, base_url, model=None, timeout=3600, metrics_url=None):
        import requests

        self.session = requests.Session()
        key = os.getenv("OPENAI_API_KEY")
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"
        self.base = normalize_url(base_url)
        if urlsplit(self.base).hostname in {"localhost", "127.0.0.1", "::1"}:
            # A local vLLM request must not be routed through the HF download proxy.
            self.session.trust_env = False
        self.timeout = timeout
        response = self.session.get(self.base + "/v1/models", timeout=30)
        response.raise_for_status()
        models = response.json()["data"]
        if not model and len(models) != 1:
            raise ValueError(
                "Server exposes multiple models; select --model explicitly"
            )
        self.model = model or models[0]["id"]
        matching = [m for m in models if m["id"] == self.model]
        if len(matching) != 1:
            raise ValueError(f"Model {self.model} is not served at {self.base}")
        self.info = matching[0]
        self.metrics_url = metrics_url or self.base + "/metrics"

    def complete(self, tokens, params):
        body = dict(
            params,
            model=self.model,
            prompt=tokens,
            n=1,
            stream=True,
            stream_options={"include_usage": True},
            return_token_ids=True,
            add_special_tokens=False,
        )
        started = time.perf_counter()
        first = last = None
        token_ids, texts, usage, metrics, finish, request_id = (
            [],
            [],
            None,
            None,
            None,
            None,
        )
        done = False
        with self.session.post(
            self.base + "/v1/completions",
            json=body,
            stream=True,
            timeout=(30, self.timeout),
        ) as response:
            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1500]}"
                )
            for raw in response.iter_lines(chunk_size=1):
                if not raw or not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(f"Server error: {event['error']}")
                request_id = event.get("id", request_id)
                if event.get("usage") is not None:
                    usage = event["usage"]
                if event.get("metrics") is not None:
                    metrics = event["metrics"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("Unexpected second completion stream")
                    delta = choice.get("token_ids")
                    if delta:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        token_ids.extend(delta)
                    texts.append(choice.get("text") or "")
                    finish = choice.get("finish_reason") or finish
        latency = time.perf_counter() - started
        if not done or usage is None or finish not in ("stop", "length"):
            raise RuntimeError(
                f"Incomplete generation: done={done}, usage={usage}, finish={finish}"
            )
        if usage.get("prompt_tokens") != len(tokens) or usage.get(
            "completion_tokens"
        ) != len(token_ids):
            raise ValueError(
                "API token IDs do not match usage; use the pinned vLLM return_token_ids support"
            )
        spec = (metrics or {}).get("speculative_decoding")
        return {
            "request_id": request_id,
            "output_token_ids": token_ids,
            "text": "".join(texts),
            "prompt_tokens": len(tokens),
            "completion_tokens": len(token_ids),
            "latency_seconds": latency,
            "ttft_seconds": first - started if first else None,
            "tpot_seconds": (last - first) / (len(token_ids) - 1)
            if first and len(token_ids) > 1
            else None,
            "finish_reason": finish,
            "engine_metrics": metrics,
            "speculation": validate_spec(spec) if spec is not None else None,
        }

    def snapshot(self):
        from prometheus_client.parser import text_string_to_metric_families

        response = self.session.get(self.metrics_url, timeout=30)
        response.raise_for_status()
        counters, positions = {}, {}
        active = 0.0
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.labels.get("model_name") != self.model:
                    continue
                name = sample.name.removesuffix("_total")
                value = float(sample.value)
                if not math.isfinite(value):
                    continue
                if name in SPEC_COUNTERS:
                    key = SPEC_COUNTERS[name]
                    counters[key] = counters.get(key, 0.0) + value
                if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                    pos = sample.labels.get("position")
                    positions[pos] = positions.get(pos, 0.0) + value
                if name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
                    active += value
        if "finished" not in counters:
            raise ValueError(
                "Prometheus request counters missing for the served model; keep --disable-log-stats off"
            )
        return {
            "counters": counters,
            "positions": positions,
            "active": active,
            "raw": response.text,
        }

    def settle(self, before, expected, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            after = self.snapshot()
            difference = after["counters"]["finished"] - before["counters"]["finished"]
            if difference < 0:
                raise ValueError("Server counters reset during evaluation")
            if difference >= expected and after["active"] == 0:
                return after
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Server request metrics did not settle; no acceptance result will be fabricated"
                )
            time.sleep(0.2)


def counter_delta(before, after):
    result = {}
    for key in set(before["counters"]) | set(after["counters"]):
        if key not in before["counters"] or key not in after["counters"]:
            raise ValueError(
                f"Counter disappeared/appeared inside measured interval: {key}"
            )
        value = after["counters"][key] - before["counters"][key]
        if value < 0:
            raise ValueError(f"Counter reset: {key}")
        result[key] = value
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows, delta, wall_seconds, expect_speculative):
    specs = [r["speculation"] for r in rows]
    per_request = all(s is not None for s in specs)
    if not expect_speculative and any(s and s["rounds"] for s in specs):
        raise ValueError(
            "--target-only was specified but the server uses a draft model"
        )
    if per_request:
        counters = {
            k: sum(s[k] for s in specs) for k in ("rounds", "accepted", "drafted")
        }
        if len({s["k"] for s in specs}) != 1:
            raise ValueError("Proposal count changed during benchmark")
        source = "per_request_api"
        k = specs[0]["k"]
        hist = [sum(s["histogram"][i] for s in specs) for i in range(k + 1)]
        per_position = [
            sum(hist[i + 1 :]) / counters["rounds"] if counters["rounds"] else None
            for i in range(k)
        ]
        consistent = all(delta.get(key) == count for key, count in counters.items())
    elif expect_speculative:
        if (
            not all(k in delta for k in ("rounds", "accepted", "drafted"))
            or delta["rounds"] <= 0
        ):
            raise ValueError(
                "No speculative metrics: check draft/proposer launch; missing metrics are not zero acceptance"
            )
        counters = {k: delta[k] for k in ("rounds", "accepted", "drafted")}
        source, hist, per_position, consistent = "prometheus_interval", None, None, True
    else:
        if delta.get("rounds", 0) > 0:
            raise ValueError(
                "--target-only was specified but the server uses a draft model"
            )
        counters = {"rounds": 0, "accepted": 0, "drafted": 0}
        source, hist, per_position, consistent = "target_only", None, None, True
    d, a, p = (counters[k] for k in ("rounds", "accepted", "drafted"))
    if not 0 <= a <= p or (d > 0 and p < d):
        raise ValueError("Invalid accepted/drafted/round counters")
    tokens = sum(r["completion_tokens"] for r in rows)
    seconds = sum(r["latency_seconds"] for r in rows)
    exclusive = delta["finished"] == len(rows)
    if not exclusive and source == "prometheus_interval":
        raise ValueError(
            "Other requests contaminated aggregate acceptance metrics; use an idle server"
        )
    return {
        "requests": len(rows),
        "output_tokens": tokens,
        "wall_seconds": wall_seconds,
        "summed_request_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "request_tokens_per_second_mean": mean(
            [r["completion_tokens"] / r["latency_seconds"] for r in rows]
        ),
        "latency_seconds_mean": seconds / len(rows),
        "ttft_ms_mean": mean(
            [
                1000 * r["ttft_seconds"] if r["ttft_seconds"] is not None else None
                for r in rows
            ]
        ),
        "tpot_ms_request_mean": mean(
            [
                1000 * r["tpot_seconds"] if r["tpot_seconds"] is not None else None
                for r in rows
            ]
        ),
        "engine_ttft_ms_mean": mean(
            [(r["engine_metrics"] or {}).get("time_to_first_token_ms") for r in rows]
        ),
        "engine_mean_itl_ms_request_mean": mean(
            [(r["engine_metrics"] or {}).get("mean_itl_ms") for r in rows]
        ),
        "verification_rounds": d,
        "accepted_draft_tokens": a,
        "drafted_tokens": p,
        "accepted_per_round_pooled": a / d if d else None,
        "acceptance_length_with_bonus_pooled": 1 + a / d if d else None,
        "acceptance_length_with_bonus_request_mean": mean(
            [s["acceptance_length"] for s in specs]
        )
        if per_request
        else None,
        "draft_acceptance_rate": a / p if p else None,
        "acceptance_histogram": hist,
        "per_position_acceptance_rate": per_position,
        "acceptance_source": source,
        "counters_crosscheck": consistent,
        "exclusive_server_interval": exclusive,
        "measurement_valid": exclusive and consistent,
        "length_limited_requests": sum(r["finish_reason"] == "length" for r in rows),
        "greedy_parity": None,
        "comparison_valid": None,
        "speedup_tps": None,
        "speedup_total_latency": None,
        "speedup_request_tps_mean": None,
    }


def compare(cell, reference, reference_report, settings):
    issues = []
    if not reference_report.get("complete"):
        issues.append("reference run incomplete")
    if settings != reference_report.get("comparison_settings"):
        issues.append("target/tokenizer/generation settings differ")
    key = (cell["benchmark"], cell["temperature"])
    old = next(
        (c for c in reference if (c["benchmark"], c["temperature"]) == key), None
    )
    if old is None:
        issues.append("reference cell missing")
    elif [r["prompt_sha256"] for r in old["results"]] != [
        r["prompt_sha256"] for r in cell["results"]
    ]:
        issues.append("ordered prompts differ")
    mismatches = []
    if not issues and cell["temperature"] == 0:
        for i, (a, b) in enumerate(zip(old["results"], cell["results"])):
            if a["output_token_ids"] != b["output_token_ids"]:
                mismatches.append(i)
        cell["summary"]["greedy_parity"] = not mismatches
        if mismatches:
            issues.append("greedy output token IDs differ")
    if not cell["summary"]["measurement_valid"] or (
        old and not old["summary"]["measurement_valid"]
    ):
        issues.append("measurement not isolated or counters disagree")
    summary = cell["summary"]
    summary["comparison_valid"] = not issues
    summary["comparison_issues"] = issues
    summary["mismatching_requests"] = mismatches
    summary["reference_method"] = reference_report.get("method")
    summary["comparison_provenance_verified"] = bool(
        settings.get("target_weight_identity") and settings.get("runtime_identity")
    )
    if not issues:
        base = old["summary"]
        summary["speedup_tps"] = (
            summary["tokens_per_second"] / base["tokens_per_second"]
        )
        summary["speedup_total_latency"] = (
            base["summed_request_seconds"] / summary["summed_request_seconds"]
        )
        summary["speedup_request_tps_mean"] = (
            summary["request_tokens_per_second_mean"]
            / base["request_tokens_per_second_mean"]
        )


def write_reports(output, report):
    write_json(output / "report.json", report)
    fields = [
        "method",
        "temperature",
        "benchmark",
        "requests",
        "tokens_per_second",
        "accepted_per_round_pooled",
        "acceptance_length_with_bonus_pooled",
        "acceptance_length_with_bonus_request_mean",
        "draft_acceptance_rate",
        "ttft_ms_mean",
        "tpot_ms_request_mean",
        "length_limited_requests",
        "measurement_valid",
        "greedy_parity",
        "comparison_valid",
        "reference_method",
        "speedup_tps",
        "speedup_total_latency",
        "speedup_request_tps_mean",
    ]
    with (output / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in report["cells"]:
            w.writerow(
                dict(
                    c["summary"],
                    method=report["method"],
                    temperature=c["temperature"],
                    benchmark=c["benchmark"],
                )
            )
    macro = []
    for temp in report["temperatures"]:
        cells = [c for c in report["cells"] if c["temperature"] == temp]
        valid_speeds = [c["summary"]["speedup_tps"] for c in cells]
        macro.append(
            {
                "temperature": temp,
                "benchmarks": len(cells),
                "acceptance_length_macro_mean": mean(
                    [c["summary"]["acceptance_length_with_bonus_pooled"] for c in cells]
                ),
                "speedup_macro_mean": mean(valid_speeds)
                if cells and all(v is not None for v in valid_speeds)
                else None,
                "complete_seven_benchmarks": len(cells) == len(COUNTS),
            }
        )
    write_json(output / "macro_summary.json", macro)


def client_runtime():
    """Select this Conda environment's C++ ABI before optional data-loader imports."""
    lib = Path(sys.prefix) / "lib/libstdc++.so.6"
    if os.getenv("RETRACE_EVAL_CPP_READY") or not lib.is_file():
        return
    if b"CXXABI_1.3.15\0" not in lib.read_bytes():
        return
    selected = [
        str(p.resolve()) for p in [lib.parent / "libgcc_s.so.1", lib] if p.is_file()
    ]
    retained = [
        p
        for p in re.split(r"[:\s]+", os.getenv("LD_PRELOAD", ""))
        if p and not Path(p).name.startswith(("libstdc++.so", "libgcc_s.so"))
    ]
    env = dict(
        os.environ, LD_PRELOAD=":".join(selected + retained), RETRACE_EVAL_CPP_READY="1"
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", help="http://host:port or http://host:port/v1")
    p.add_argument("--model", help="Automatically discovered when one model is served")
    p.add_argument("--method", help="Result label; defaults to served model name")
    p.add_argument(
        "--target", type=Path, default=Path(os.getenv("RETRACE_TARGET", DEFAULT_TARGET))
    )
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().with_name("retrace_eval_prompts.json"))
    p.add_argument("--datasets", "--dataset", nargs="+", default=["all"])
    p.add_argument(
        "--source",
        action="append",
        default=[],
        help="DATASET=local JSONL/JSON/save_to_disk directory",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use local sources or existing HF cache only",
    )
    p.add_argument("--lcb-release", type=int, default=5)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Requests per dataset for smoke evaluation; 0=manifest count",
    )
    p.add_argument(
        "--temperatures", nargs="+", type=float, choices=(0, 1), default=[0, 1]
    )
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=3600)
    p.add_argument("--metrics-url")
    p.add_argument(
        "--target-only",
        action="store_true",
        help="Evaluate an AR server; no draft acceptance expected",
    )
    p.add_argument(
        "--reference",
        type=Path,
        help="Earlier report.json for paired speedup/parity; AR needed for Table 2 speedup",
    )
    p.add_argument(
        "--server-manifest",
        type=Path,
        help="Optional server_manifest.json written by the launcher",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if (
        args.limit < 0
        or args.max_new_tokens < 1
        or args.timeout_s <= 0
        or args.warmup_requests < 1
        or not 1 <= args.lcb_release <= 10
        or len(set(args.temperatures)) != len(args.temperatures)
    ):
        p.error(
            "Invalid limit, token budget, timeout, warmup count, release or repeated temperatures"
        )
    if not args.prepare_only and not args.base_url:
        p.error("--base-url is required unless --prepare-only")
    client_runtime()
    if args.offline:
        # HF captures these settings on import, before tokenizer preparation.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    names = names_from(args.datasets)
    manifest = prepare_manifest(args, names)
    selected = select_prompts(manifest, names, args.limit)
    target_config_hash = file_hash(args.target / "config.json")
    if manifest.get("target_config_sha256") != target_config_hash:
        raise ValueError("Manifest was prepared with a different target configuration")
    for name, expected in manifest.get("tokenizer_sha256", {}).items():
        if Path(name).name != name or file_hash(args.target / name) != expected:
            raise ValueError("Manifest tokenizer files differ from the local target")
    if args.prepare_only:
        print(f"Frozen prompts: {args.manifest.resolve()}")
        return
    client = Client(args.base_url, args.model, args.timeout_s, args.metrics_url)
    eos = json.loads((args.target / "generation_config.json").read_text()).get(
        "eos_token_id", 151645
    )
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or any(type(t) is not int or t < 0 for t in eos):
        raise ValueError("Invalid target EOS configuration")
    max_len = client.info.get("max_model_len")
    if max_len and any(
        len(r["prompt_token_ids"]) + args.max_new_tokens + 16 > max_len
        for rows in selected.values()
        for r in rows
    ):
        raise ValueError(
            "A full prompt + output + draft buffer exceeds server max_model_len; increase it without truncating prompts"
        )
    method = args.method or ("ar" if args.target_only else client.model)
    output = args.output or Path(
        f"eval_{method}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "prompts_used.json", selected)
    server_manifest = (
        json.loads(args.server_manifest.read_text()) if args.server_manifest else None
    )
    if server_manifest and server_manifest["served_model"] != client.model:
        raise ValueError("Server provenance file belongs to a different served model")
    settings = {
        "target_config_sha256": target_config_hash,
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_ids": eos,
        "thinking": False,
        "concurrency": 1,
        "sampling_protocol": "ReTrace-v2-Appendix-E",
        "stream": True,
        "target_weight_identity": (server_manifest or {}).get("target_sha256"),
        "runtime_identity": (
            {
                k: server_manifest.get(k)
                for k in (
                    "device",
                    "tp_size",
                    "dtype",
                    "max_model_len",
                    "prefix_caching",
                    "enforce_eager",
                    "model_runner",
                    "commits",
                )
            }
            if server_manifest
            else None
        ),
    }
    reference = json.loads(args.reference.read_text()) if args.reference else None
    report = {
        "format": 3,
        "complete": False,
        "method": method,
        "base_url": client.base,
        "model": client.model,
        "model_info": client.info,
        "manifest_sha256": file_hash(args.manifest),
        "comparison_settings": settings,
        "temperatures": args.temperatures,
        "datasets": names,
        "server_manifest": server_manifest,
        "eval_code_sha256": file_hash(__file__),
        "argv": sys.argv,
        "paper": "https://arxiv.org/pdf/2608.29748",
        "cells": [],
        "limits": [
            "Stored-response training and Ascend vLLM are adaptations, not an exact Table 2 reproduction.",
            "DSpark block7 is an extension; Table 2 uses DFlash b16.",
            "Acceptance length=1+accepted draft tokens/verification rounds (bonus convention); terminal clipping can differ from actual emitted tokens.",
            "Prompt IDs/templates and MT-Bench first-turn policy are frozen choices; author prompt IDs are unavailable.",
            "Streaming TTFT measures first returned token chunk; speculative chunks can contain multiple tokens.",
            "T=1 comparison does not test sampling-distribution equivalence; generated programs are never executed.",
            "Without server provenance, target weight/runtime identity is user-asserted; run matching launchers on the same idle hardware.",
        ],
    }
    write_reports(output, report)
    try:
        for temperature in args.temperatures:
            params = sampling(temperature, args.max_new_tokens, args.seed, eos)
            for name, prompts in selected.items():
                prefix = f"{name}_t{temperature:g}"
                before_warmup = client.snapshot()
                if before_warmup["active"]:
                    raise ValueError(
                        "Server is busy; evaluate each method in isolation"
                    )
                for i in range(args.warmup_requests):
                    client.complete(
                        prompts[i % len(prompts)]["prompt_token_ids"], params
                    )
                client.settle(before_warmup, args.warmup_requests)
                # Launchers disable prefix caching; no prompt warmup cache can be reused.
                before = client.snapshot()
                (output / f"{prefix}_before.prom").write_text(before["raw"])
                rows = []
                started = time.perf_counter()
                with (output / f"{prefix}_requests.jsonl").open("x") as f:
                    for i, prompt in enumerate(prompts):
                        row = client.complete(prompt["prompt_token_ids"], params)
                        row.update(
                            prompt_sha256=prompt["sha256"],
                            source_row=prompt.get("source_row"),
                            source_id=prompt.get("source_id"),
                        )
                        rows.append(row)
                        f.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                        f.flush()
                        tau = (row["speculation"] or {}).get("acceptance_length")
                        print(
                            f"{name} T={temperature:g} {i + 1}/{len(prompts)}: "
                            f"{row['completion_tokens']} tokens, {row['latency_seconds']:.2f}s, tau={tau}",
                            flush=True,
                        )
                wall_seconds = time.perf_counter() - started
                after = client.settle(before, len(prompts))
                (output / f"{prefix}_after.prom").write_text(after["raw"])
                delta = counter_delta(before, after)
                cell = {
                    "benchmark": name,
                    "temperature": temperature,
                    "sampling": params,
                    "prometheus_delta": delta,
                    "results": rows,
                    "summary": aggregate(
                        rows, delta, wall_seconds, not args.target_only
                    ),
                }
                if reference is not None:
                    compare(cell, reference["cells"], reference, settings)
                report["cells"].append(cell)
                write_reports(output, report)
                print(json.dumps(cell["summary"], indent=2), flush=True)
        report["complete"] = True
        write_reports(output, report)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_reports(output, report)
        raise
    print(f"Results: {output.resolve()}/summary.csv", flush=True)


if __name__ == "__main__":
    main()
