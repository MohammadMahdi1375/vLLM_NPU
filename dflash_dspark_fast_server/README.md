# DFlash, DFlash-Prefix and DSpark: configurable fast serving

This update gives the three methods the same target-serving configuration and
the same evaluator. `target_graph_server.py` accepts the speculative method as
an argument; it no longer always constructs `method="dflash"`.

| Launcher | Speculative method | Checkpoint family |
| --- | --- | --- |
| `Evaluator/run_server_dflash.sh` | `dflash` | vanilla DFlash |
| `Evaluator/run_server_prefix_dflash.sh` | `dflash` | `local_prefix_v2` or `compiled_prefix_v3` |
| `Evaluator/run_server_dspark.sh` | `dspark` | DSpark |

The prefix and DSpark scripts pass editable defaults into the shared launcher.
All settings can be supplied through environment variables. An explicit method
that disagrees with the selected checkpoint is rejected. `SPECULATIVE_METHOD=auto`
can resolve the method from `config.json` instead.

## Install

Run inside your existing A2 tree, with the previous target-graph and prefix
runtime installed. This is an entrypoint update, not a replacement vLLM installation.

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
unzip -o dflash_dspark_fast_server.zip
python dflash_dspark_fast_server/install.py --spec-main "$PWD" --check &&
python dflash_dspark_fast_server/install.py --spec-main "$PWD"
```

The installer checks all destinations before writing, merges unrelated edits,
and stops if edits overlap. It backs up replaced files and prints a rollback
command. It does not change checkpoints, training code, kernels, or Git history.
Restart serving to use the updated helper. Reinstalling the same package retains
settings you edit after installation.

## Select checkpoints and start servers

Edit `DRAFT_MODEL` in each launcher, or override it when starting that launcher.
The vanilla default is `dflash_qwen3_4b_openperfectblend_bs16/checkpoints/4`.
The prefix default is your completed joint run's `checkpoints/4`.
The DSpark default is the previously supplied **`checkpoints/0` (epoch 1)**.
For an epoch-5 comparison, replace that DSpark path with your trained epoch-5
checkpoint. A launch record identifies a checkpoint but cannot certify that its
training budget matches another model.

Run **one server at a time** on the same idle NPU. These commands deliberately
use the same device, port and served model name. Stop the current server before
running the next command. They are alternatives, not a single shell batch.

Vanilla DFlash:

```bash
NPU=6 PORT=8213 SERVED_MODEL_NAME=qwen3-4b-spec \
bash Evaluator/run_server_dflash.sh
```

DFlash-Prefix:

```bash
NPU=6 PORT=8213 SERVED_MODEL_NAME=qwen3-4b-spec \
bash Evaluator/run_server_prefix_dflash.sh
```

DSpark, after setting its checkpoint in `Evaluator/run_server_dspark.sh`:

```bash
NPU=6 PORT=8213 SERVED_MODEL_NAME=qwen3-4b-spec \
bash Evaluator/run_server_dspark.sh
```

For a configuration-only check, prepend `CHECK_ONLY=1`. It reads local configs
and source files and saves the proposed command without loading CANN or starting
a server. Each invocation gets a fresh run directory unless `RUN_DIR` is set.

## Execution settings shared by the three launchers

| Setting | Default |
| --- | --- |
| Target model | Original Qwen3-4B target, never a draft checkpoint |
| `TARGET_EXECUTION` | `graph` |
| Target graph | Mode 3, `FULL_DECODE_ONLY`, capture size `K + 1` |
| `DRAFT_ENFORCE_EAGER` | `omit` |
| `NUM_SPECULATIVE_TOKENS` | 15 |
| Draft sampling | Greedy |
| Adaptive verification | Disabled; full target verification retained |
| TP / DP / max sequences | 1 / 1 / 1 |
| Precision / seed | BF16 / 42 |
| Maximum sequence / batched tokens | 32768 / 32768 |
| KV page size / memory utilization | 128 / 0.96 |
| Prefix caching / chunked prefill / async scheduling | Disabled |
| API servers / renderer workers | 1 / 1 |
| Graph metrics / selector profiling | Off during timing |

There is no top-level `--enforce-eager` in graph mode. By default the speculative
JSON also omits `enforce_eager`, leaving draft execution to the backend. This is
different from your earlier measured configuration, which set draft eager true.

**DSpark's current Ascend proposer explicitly disables its draft graph.** Omitting
the option does not remove that backend limitation. DSpark's target can still use
the target graph. This package does not claim to add a DSpark draft graph.

For a controlled comparison with all draft backbones requested eager, or to
reproduce the earlier DFlash execution configuration, set
`DRAFT_ENFORCE_EAGER=1` on **all three** launches. The target graph and prefix
selector graph remain enabled. `DRAFT_ENFORCE_EAGER=0` explicitly requests false;
`omit` leaves the key absent. Use one policy consistently in a comparison, and
report the actual runtime restrictions. Do not compare a new graph-enabled run
against an older eager baseline and attribute the entire difference to training.

Prefix selectors retain their optimized path: compiled selector graph for v3,
or `local_unpaired` fusion for local-prefix v2. Token caching stays off, score
checks stay strict, all trained proposals remain, and 32 startup checks precede
steady-state fast mode. These operations have no counterpart in vanilla or DSpark.

For block-8 DFlash/Prefix, use its block-8 checkpoint and
`NUM_SPECULATIVE_TOKENS=7`. Their proposal count must equal `block_size - 1`.
DSpark's available positions depend on `sample_from_anchor`; the helper checks
this and allows an explicit count within its trained support. The common
comparison uses exactly 15 proposals and 16 verifier positions. The launcher
never silently truncates `auto` to fit this Ascend recipe's 16-position limit.

## One evaluation command for all three

In a second terminal, while the chosen server is running:

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/Evaluator
python evaluator.py \
  --base-url http://127.0.0.1:8213 \
  --model qwen3-4b-spec \
  --datasets gsm8k \
  --manifest ./retrace_eval_prompts_all_v2.json \
  --num-prompts 128 \
  --temperatures 0 --concurrencies 1 \
  --top-p 1.0 --top-k -1 --max-new-tokens 2048 \
  --seed 42 --prompt-seed 42 \
  --warmup-requests 1 \
  --selector-warmup-tokens 512 --selector-warmup-steps 32 \
  --no-thinking \
  --output "results/spec_compare_$(date +%Y%m%d_%H%M%S)"
```

The evaluator automatically reads the current launch record and labels the
actual family, block, proposal count and execution request. You can still use
`--method` for a custom label; that option does not choose a decoding algorithm.
`--model` must match the served name: the commands above set the same name for
all methods so the evaluation command can remain identical.

Warmup is excluded from timing and acceptance counters for every method, including
DSpark. Prefix parity checks apply only to prefix checkpoints. Metric and sampling
functions are unchanged. Compare pooled acceptance length to pooled acceptance
length. The launch and evaluation records retain checkpoint paths and source/config
hashes; record training epochs, token budgets and draft architecture alongside them.

Use the same frozen manifest, target weights, NPU, output limits and sampling.
Run sequentially with no competing device workload. For paper timing, repeat
each method at least three times with rotated ordering; report variation and
compare matched training budgets. These scripts make serving settings consistent;
they cannot guarantee identical numerical behavior or hardware throughput.

## Validate the newly requested graph path on A2

The included checker now accepts all three methods and only requires prefix
markers for prefix checkpoints. It starts an eager control and a graph server
sequentially, compares observed output token IDs and acceptance counters, and
checks for actual FULL target dispatch. Stop your benchmark server first.

To avoid typing a placeholder checkpoint, read the path explicitly:

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
read -r -p "DSpark checkpoint directory: " DSPARK_CHECKPOINT
test -f "$DSPARK_CHECKPOINT/config.json" &&
python speculators/scripts/dflash_prefix_compiled/check_target_graph.py \
  --spec-main "$PWD" --draft "$DSPARK_CHECKPOINT" \
  --speculative-method dspark --num-speculative-tokens 15 \
  --draft-enforce-eager omit --npu 6 --port 8213
```

Use `--speculative-method dflash` and the corresponding checkpoint for DFlash or
Prefix. Use the same `--draft-enforce-eager` choice as the intended benchmark.
This is a diagnostic, not a speed run. Passing covers only the observed prompts.

CPU tests cover command construction, common settings, checkpoint/method matching,
real localhost HTTP plumbing with synthetic counters, and reversible installation.
They do not execute vLLM kernels or certify A2 graph parity/speedup. Test details
and results are included in `VALIDATION.json` and `tests/CPU_TESTS.txt`.
