# Training Throughput Plan

Status: planned. The trainer uses ordinary fp32 with optional tensor caches,
configurable workers, CUDA pinned memory, and non-blocking transfers. AMP,
gradient accumulation, and compilation are not implemented.

Keep fp32 and eager execution as defaults. Add each optimization separately
behind config so numerical changes, failures, and speed can be compared.

## Baseline And Measurement

Use an existing cached supervised config. Record config/commit, seed,
Python/PyTorch/CUDA versions, platform/GPU, data/cache identity, model preset,
batch size, worker count, sample/epoch counts, elapsed time, samples/sec,
peak GPU memory, train/validation losses, and failure status.

For comparisons, hold dataset, split, seed, model, initial batch size, epoch
count, and workers constant. Report speed and finite-loss checks separately
from playing strength.

## Implementation Order

| Step | Proposed setting | Required behavior and validation |
|---|---|---|
| 1. AMP | `precision: fp32\|amp_fp16\|amp_bf16` | Autocast in training and validation; CUDA fp16 uses GradScaler, bf16 does not. Fail early on unsupported device/precision combinations. |
| 2. AMP comparison | Same config family | Measure fp32, fp16, and supported bf16; finite losses, memory, speed, and checkpoint loading. |
| 3. Accumulation | `gradient_accumulation_steps: 1` | Scale losses correctly, handle the final partial group, and distinguish physical batches, samples, and optimizer steps. |
| 4. Compilation | `compile_model: false` | Measure startup/warmup and steady-state speed; fail clearly unless fallback is explicitly configured. |
| 5. Loader/optimizer tuning | Explicit later settings | Benchmark workers and prefetching per platform; assess fused AdamW separately from precision. |

These are proposed fields, not currently accepted training options. Add config,
metadata, and unsupported-device tests with each implementation. Existing fp32
tests and smoke runs must keep passing; CUDA smoke runs should cover only
supported modes and produce finite losses through the normal checkpoint path.

Effective batch size with accumulation is physical batch size times
accumulation steps, apart from the final partial group. Changing optimizer-step
cadence also changes learning-rate scheduling, so record it explicitly.
Default accumulation of 1 must retain current behavior.

## Acceptance And Boundaries

Copied configs, checkpoint training metadata, metrics, and status must identify
the resolved precision, accumulation, and compilation settings. Decide and test
any scaler/resume-state requirements before claiming equivalent continuation.
Use hardware-specific measurements to recommend settings; Windows workers may
behave differently from Linux workers.

Preserve model shapes, legal masking, supervision rules, and data semantics.
No automatic precision changes, distributed training, model changes, or new
tracking framework are part of this plan.
