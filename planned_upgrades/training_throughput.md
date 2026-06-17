# Training Throughput Upgrade Plan

## Context

The supervised trainer currently runs in ordinary fp32 unless PyTorch or the
device backend applies lower-level defaults. The training path already has some
throughput-oriented pieces:

- CUDA device selection from config
- tensor-cache loading for repeated training runs
- configurable `num_workers`
- pinned memory on CUDA
- non-blocking CUDA transfers

The missing training-throughput features are mixed precision, gradient
accumulation, `torch.compile`, and more explicit benchmark reporting.

These should be added as planned upgrades, not silent defaults. Precision,
batching, and compiler settings can change numerical behavior, failure modes,
and comparability between runs.

## Constraints

Must preserve:

- fp32 as the default behavior
- reproducibility from config
- copied run config in the output directory
- checkpoint metadata that records the training settings
- policy/value output shapes: `policy_logits: [batch, 4672]`, `value: [batch]`
- legal move masking outside the model
- no Stockfish, Syzygy, tablebase, Leela, or external engine labels

Any throughput result should report speed and correctness checks separately
from playing strength. Faster training does not by itself imply a stronger
checkpoint.

## Stage 0: Baseline Benchmark

Goal: record the current fp32 training speed before changing trainer behavior.

Use an existing cached supervised config and record:

```text
config
git commit
python version
torch version
platform
GPU model
CUDA version if available
dataset/cache paths
model preset
batch size
num_workers
train samples
validation samples
epoch count
elapsed seconds
samples/sec
peak GPU memory if available
train and validation losses
failure status if the run does not complete
```

Acceptance:

- A baseline report exists.
- The benchmark command is reproducible from a committed config.
- No trainer behavior changes are made in this stage.

## Stage 1: Opt-In AMP

Goal: add mixed precision without changing default fp32 training.

Planned config field:

```yaml
precision: fp32
```

Allowed values:

- `fp32`
- `amp_fp16`
- `amp_bf16`

Initial behavior:

- `fp32` keeps the current training path.
- `amp_fp16` uses CUDA autocast and `GradScaler`.
- `amp_bf16` uses CUDA autocast without `GradScaler`.
- AMP modes should fail early with a clear error if the selected device does
  not support them.

Implementation notes:

- Use AMP in the train forward/loss section.
- Use AMP in validation forward/loss sections.
- Use `GradScaler` only for CUDA fp16.
- Keep optimizer, scheduler behavior, checkpoint format, and model outputs
  otherwise unchanged.
- Record the resolved precision mode in `status.json`.
- Keep the copied config and checkpoint `train_config` sufficient to identify
  the precision mode used for a run.

Acceptance:

- Existing fp32 tests still pass.
- A smoke training run works with `precision: fp32`.
- A CUDA smoke training run works with `precision: amp_fp16` when CUDA is
  available.
- A CUDA smoke training run works with `precision: amp_bf16` only on hardware
  that supports bfloat16.
- Training fails clearly for unsupported precision/device combinations.
- Checkpoints trained with AMP still load through the normal checkpoint path.
- No NaNs or infinite losses occur in the smoke runs.

## Stage 2: Benchmark AMP

Goal: decide whether AMP is useful for the current models and hardware.

Compare on the same cached dataset:

- fp32
- amp_fp16
- amp_bf16 if supported

Hold constant:

- dataset/cache
- train/validation split
- seed
- model preset
- batch size for the first comparison
- epoch count
- `num_workers`

Report:

```text
precision
elapsed seconds
samples/sec
peak GPU memory
train total loss
validation total loss
status
notes
```

Acceptance:

- A short report compares fp32 and AMP on the same config family.
- Any recommendation is based on measured speed and loss sanity, not assumed
  GPU behavior.
- Previous fp32 results remain comparable because AMP is opt-in.

## Stage 3: Gradient Accumulation

Goal: allow larger effective batches when memory limits the physical batch
size.

Planned config field:

```yaml
gradient_accumulation_steps: 1
```

Notes:

- The default keeps current behavior.
- Effective batch size is `batch_size * gradient_accumulation_steps`.
- Metrics should distinguish sample count, batch count, and optimizer step
  count.
- Learning-rate comparisons need care because accumulation changes optimizer
  step cadence.

Acceptance:

- `gradient_accumulation_steps: 1` matches current behavior.
- Accumulation divides loss correctly before backward.
- Optimizer steps occur at the expected interval.
- The final partial accumulation step is handled explicitly.
- Metrics and status files record accumulation settings.

## Stage 4: `torch.compile`

Goal: test PyTorch compilation as a separate opt-in optimization.

Planned config field:

```yaml
compile_model: false
```

Optional later fields:

```yaml
compile_backend: null
compile_mode: null
```

Notes:

- Keep this separate from AMP so failures are easy to isolate.
- Record compile settings in run metadata.
- Benchmark includes compile warmup cost and steady-state epoch speed.
- The default remains uncompiled eager execution.

Acceptance:

- Eager execution remains the default.
- Compiled runs produce finite losses in a smoke run.
- Compilation failures report a clear error or fall back only if explicitly
  configured to do so.
- Benchmark reports both startup cost and per-epoch speed.

## Stage 5: Data Loader And Optimizer Tuning

Goal: tune remaining throughput knobs after AMP is measured.

Candidate changes:

- benchmark `num_workers` values per platform
- benchmark `prefetch_factor` when `num_workers > 0`
- keep pinned memory enabled for CUDA
- consider fused `AdamW` only on supported PyTorch/CUDA versions

Notes:

- Windows worker behavior can differ from Linux because workers are spawned.
- The tensor cache should remain the preferred path for repeated CUDA training.
- Optimizer changes should be benchmarked separately from precision changes.

Acceptance:

- Configs record any loader or optimizer changes.
- Benchmarks report platform and hardware.
- No data semantics or model contracts change.

## Suggested Implementation Order

1. Record an fp32 baseline benchmark.
2. Add the `precision` config field with default `fp32`.
3. Implement autocast and `GradScaler` behind the precision setting.
4. Add tests for config parsing, metadata, and unsupported device handling.
5. Run fp32 and AMP smoke training.
6. Benchmark fp32 versus AMP on the same cached dataset.
7. Add gradient accumulation only after AMP behavior is stable.
8. Test `torch.compile` only after AMP and accumulation are independently
   understood.

## Non-Goals

- No automatic precision changes.
- No unsupported strength or Elo claims.
- No multi-GPU or distributed training in this upgrade.
- No model architecture changes.
- No change to legal move masking or chess-rule handling.
- No new experiment-tracking framework unless the current run artifacts become
  insufficient.
