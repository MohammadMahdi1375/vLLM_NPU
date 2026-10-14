# Matched DFlash, DFlash-Prefix and DSpark comparison on Ascend A2

This package prepares the three existing methods for a controlled comparison. It does not certify NPU speed, accuracy, or publication readiness before the hardware checks run. No model weights, training settings, proposal budget, scoring arithmetic, or target verification rule are changed.

## What changes

- `Evaluator/run_server_dflash.sh`, `run_server_prefix_dflash.sh`, and `run_server_dspark.sh` use one configurable command builder. The speculative method is explicitly passed and checked against the checkpoint. Prefix uses method `dflash`; DSpark uses `dspark`.
- The default `matched_fast` profile requests the same target decode graph and explicitly keeps **all three draft backbones eager**. The historical fast prefix recipe already used this division of work.
- Prefix retains its existing strict selector graph. A new opt-in DSpark graph captures its original sequential full-vocabulary Markov loop, including vanilla, gated, and RNN heads. It does not enable unsupported DSpark attention graphs. LM-head computation remains outside this new graph. The recurrent state resets on every block.
- DSpark checks exact scores and selected tokens for the first 32 real blocks in fast mode, or every block in validate mode. Capture is deferred past dummy/profile runs. Failure stops the process; there is no silent fallback, approximation, shortlist, confidence pruning, or skipped verification. Graph outputs are cloned to avoid replay aliasing.
- `evaluator.py` retains the existing metric and sampling formulas. It confirms DSpark's head parity marker and saves ordered prompt/output/counter fingerprints, enabling comparisons without storing full generated text.
- `compare_speculative.py` and `run_paper_comparison.sh` perform preflight, per-method reference/graph validation, and repeated timings sequentially on one NPU. They hash weights and source, save commands and launch records, reject incomplete or contaminated results, rotate timing order, and summarize all repeats.

The supported recipe is Qwen3, BF16, one NPU, TP=DP=1, concurrency=1, K=7 or 15 for the corresponding DFlash/Prefix block-8/16 checkpoints, and full-vocabulary DSpark. Reduced-vocabulary DSpark and adaptive verification are outside this comparison profile. A trained confidence head can remain in a DSpark checkpoint; adaptive proposal truncation is disabled for every method so K is fixed.

## Execution profiles

| Setting | Vanilla DFlash | DFlash-Prefix | DSpark |
|---|---|---|---|
| Default target decode | graph | graph | graph |
| Default draft backbone | eager | eager | eager |
| Default token selection | original argmax | existing selector graph | new Markov-loop graph |
| Full target verification | yes | yes | yes |
| Startup parity checks for added head graph | not applicable | 32 blocks | 32 blocks |
| Measured requests | same frozen manifest | same frozen manifest | same frozen manifest |

`HEAD_EXECUTION` controls learned-head execution; it does not mean vanilla DFlash has a prefix fast mode. Vanilla records `head_execution_requested=argmax`.

There are **two distinct comparison policies**:

1. `BACKBONE_POLICY=matched_eager` (default): controls backbone execution across all methods. This is a matched implementation comparison, not a claim that every baseline uses its absolute fastest possible backend.
2. `BACKBONE_POLICY=backend_default`: removes the speculative `enforce_eager` override for all methods. DFlash/Prefix may use supported draft graphs; DSpark still explicitly keeps its draft attention eager and uses its separate Markov graph. This is an additional deployment comparison. The same strict end-to-end validation is required. Record this policy separately; do not label it "all draft backbones graphed" or pool its timings with the first policy.

For a strong paper, report both policies, or at least check the second policy before claiming a speed advantage over the fastest available DFlash baseline. Actual fastest execution must be measured; `omit` alone is not evidence of graph dispatch or greater speed. This package makes neither an all-graphs nor a universally-fastest claim.

## Install

From your project root:

```bash
cd /home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
unzip -o speculative_paper_comparison.zip
python speculative_paper_comparison/install.py --spec-main "$PWD" --check &&
python speculative_paper_comparison/install.py --spec-main "$PWD"
```

The installer supports the earlier fast-server package and published baselines, merges non-overlapping changes, checks all destinations before writing, and creates a rollback copy. Existing checkpoints and training code are untouched. No pip reinstall is required when using the launcher's source PYTHONPATH.

If a shell launcher has overlapping local edits, inspect the printed diff and explicitly replace only that launcher with the new defaults, retaining its backup:

```bash
python speculative_paper_comparison/install.py --spec-main "$PWD" \
  --replace-launcher Evaluator/run_server_dspark.sh --check &&
python speculative_paper_comparison/install.py --spec-main "$PWD" \
  --replace-launcher Evaluator/run_server_dspark.sh
```

Repeat `--replace-launcher` for another conflicting run_server shell file if needed. Python conflicts cannot be bypassed by this option. Do not overwrite the whole source tree or reset Git. The installer prints its exact rollback command.

## Select checkpoints

Activate your existing A2 environment. Set the DSpark checkpoint explicitly; the previous epoch-1 default is intentionally removed.

```bash
export SPEC_MAIN=/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main
export ENV_ROOT=/home/n84449292/m84379596/conda/vllm-ascend-0202
export PYTHON_BIN="$ENV_ROOT/bin/python"
export CANN_ROOT=/home/n84449292/m84379596/CANN/9.1.0
export DFLASH_DRAFT="$SPEC_MAIN/output/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/4"
export PREFIX_DRAFT="$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4"
read -r -p 'Full path to your matched DSpark checkpoint: ' DSPARK_DRAFT
export DSPARK_DRAFT
cd "$SPEC_MAIN"
```

The default target is your original Qwen3-4B snapshot, not a draft checkpoint. Override `TARGET_MODEL`, `MANIFEST`, or any path if needed. No A3 SOC identifier is forced on A2.

Same epoch directory numbers do NOT prove equal training budgets. Record training examples/tokens, optimizer steps, epochs, data split, model depth/width/vocabulary, and checkpoint-selection rule for each method. Optional `TRAINING_RECORDS=/path/to/your_training_records.json` copies your declarations into provenance; it is not independently verified. Do not compare an epoch-1 DSpark checkpoint against epoch-5 models as a training-matched result.

## First check, then run

Use an idle NPU. The runner starts/stops only its own servers and rejects an occupied port. It cannot reserve the NPU against other users or detect every unrelated device workload.

```bash
CHECK_ONLY=1 NPU=6 PORT=8213 \
  bash Evaluator/run_paper_comparison.sh
```

Preflight uses no Torch/CANN/NPU server, but hashes weight files for reproducibility; this may take a few minutes. A successful preflight is not hardware validation. Use a new RUN_DIR for the actual experiment (the default automatically does this).

```bash
NPU=6 PORT=8213 DATASETS=gsm8k REPEATS=3 \
  bash Evaluator/run_paper_comparison.sh
```

The default `PHASE=all` does the following:

1. Runs each checkpoint with target eager + reference head, then target graph + continuous head validation, on all selected evaluation requests with the same generation limits. Both are diagnostics, not timed results.
2. Requires exact output and per-request acceptance-counter agreement between each checkpoint's reference and optimized runs. Requires actual target FULL dispatch in validation logs and head parity markers where applicable. Requires output token fingerprints to agree across the three methods at temperature zero; acceptance is expected to differ across methods.
3. Restarts servers and benchmarks each method three times. Order rotates DFlash/Prefix/DSpark, Prefix/DSpark/DFlash, DSpark/DFlash/Prefix. Every timing run excludes warmup and initial 32-block checks, disables continuous checks/profilers, and must retain the validated outputs and counters.
4. Saves `comparison.csv`, `comparison.json`, `identity.json`, `result.json`, hardware information, per-run logs, `launch.json`, commands, and evaluator `run_info.json`.

At default settings, validation is six full dataset passes plus nine timing passes. This is deliberately more than the earlier 8-prompt spot check. Each path is validated on the selected inputs only; there is no claim of universal numerical equivalence. A mismatch leaves a diagnostic result and blocks subsequent timings. Do not weaken strict checks to obtain a table.

For all five datasets:

```bash
NPU=6 PORT=8213 \
DATASETS='gsm8k math500 humaneval mbpp mtbench' REPEATS=3 \
  bash Evaluator/run_paper_comparison.sh
```

Use the same settings and checkpoints for the additional backend-default comparison:

```bash
NPU=6 PORT=8213 BACKBONE_POLICY=backend_default \
DATASETS='gsm8k math500 humaneval mbpp mtbench' REPEATS=3 \
  bash Evaluator/run_paper_comparison.sh
```

The backend-default candidate can fail graph validation on your fork/hardware. A failure is an unsupported candidate, not permission to silently substitute a slower mode under the same label. The matched policy remains separately available.

## Split validation and timing

```bash
PHASE=validate NPU=6 PORT=8213 \
RUN_DIR="$SPEC_MAIN/output/paper_validate_bs16" \
  bash Evaluator/run_paper_comparison.sh

PHASE=benchmark NPU=6 PORT=8213 \
VALIDATION_DIR="$SPEC_MAIN/output/paper_validate_bs16" \
RUN_DIR="$SPEC_MAIN/output/paper_timing_bs16" \
  bash Evaluator/run_paper_comparison.sh
```

Both commands must use the same dataset list, prompt count, token limit, seed, NPU, backbone policy, weights, installed code, and package versions. Reusing stale validation after changing these settings is rejected. Output directories must be new. There is no arbitrary "skip validation" flag for the benchmark phase.

## Individual servers remain available

All settings can be edited or overridden directly. Examples:

```bash
DRAFT_MODEL="$DFLASH_DRAFT" NPU=6 PORT=8213 bash Evaluator/run_server_dflash.sh
DRAFT_MODEL="$PREFIX_DRAFT" NPU=6 PORT=8213 bash Evaluator/run_server_prefix_dflash.sh
DRAFT_MODEL="$DSPARK_DRAFT" NPU=6 PORT=8213 bash Evaluator/run_server_dspark.sh
```

Run one at a time. Stop a manually started server before starting the comparison runner. `HEAD_EXECUTION=validate` performs continuous checks; its throughput is diagnostic. `COMPARISON_PROFILE=matched_reference HEAD_EXECUTION=reference` retains learned prefix/Markov heads but executes them eagerly. `TARGET_EXECUTION=eager` changes the target only. For manually selecting backend defaults, use `COMPARISON_PROFILE=backend_default DRAFT_ENFORCE_EAGER=omit HEAD_EXECUTION=fast` consistently for each method.

`COMPARISON_PROFILE=custom` is for experiments outside the controlled suite and should not be used to produce a matched-paper table. The older `check_target_graph.py` remains a target-only spot-check utility; use the full comparison runner for the new Markov-graph validation.

## Reading the results

- Tau is the existing pooled metric: `1 + accepted_draft_tokens / verification_rounds`, including the verifier bonus convention.
- Tokens/s is the unchanged evaluator metric: total generated tokens divided by summed request latency, at concurrency one. It includes request/prefill cost, and is distinct from wall-clock tokens/s and inverse mean TPOT.
- `tokens_per_second_mean/std` summarize all independent server timing repeats. Standard deviation is between runs, not a confidence interval or significance test.
- `relative_tps_vs_dflash` is the ratio of repeat means within the same dataset/policy. `paired_speed_ratio_mean` averages within-repeat ratios. Neither is speedup over target-only autoregressive decoding; that requires a separate target-only baseline.
- Failed/incomplete/diagnostic runs never enter the comparison table. The runner does not select only the fastest repeat or infer labels from a served model name.
- `training_budget_independently_verified=false` remains explicit. Hardware parity passing does not validate training-budget fairness, dataset independence, generalization, novelty, statistical significance, or a claim of universal speedup.

## Validation status and rollback

CPU tests cover the original DSpark Markov loop against the new loop, RNN reset and output lifetime, exact parity rejection, dummy exclusion, method/config routing, installer merges/rollback, evaluator metric invariance, and complete HTTP orchestration with deliberate failure injection. HTTP fixture timings are synthetic and are not NPU performance results.

Actual NPUGraph capture, Ascend kernels, memory behavior, full output parity, and speed are untested locally. Run the A2 validation before using the new DSpark graph in a result. The code is experimental until that passes.

The new environment variable `VLLM_ASCEND_DSPARK_HEAD_GRAPH` is centralized in `vllm_ascend/envs.py`: `off` (default outside these launchers), `validate`, or `fast`. It is not sensitive. The scope excludes training, tensor parallelism, multiple concurrent requests, reduced vocabulary, probabilistic proposals, and adaptive verification.

Use the exact rollback command printed by the installer. It restores modified files byte-for-byte and refuses to overwrite subsequent edits. No remote Git push is performed by this package.
