# Reproducibility

Every experiment starts from a config and has a stable run ID. A changed seed,
dataset, checkpoint, or protocol gets a new run ID and output directory.
Use validation data for tuning and checkpoint selection; reserve test data for
the final fixed evaluation. External engine results must not guide training or
checkpoint selection.

## Run Record

Keep these alongside the local run under ignored `runs/`:

- question, hypothesis, and status: planned, running, completed, failed, aborted,
  or inconclusive; record the failure reason when applicable;
- config copy, random seed, Git commit if available, start/end times, runtime;
- dataset manifest, model/training/evaluation configs, checkpoint identifiers;
- metrics, evaluation results, hardware notes, and exact opponents;
- scope and limitations, including missing metadata as explicit `null` values.

Failed, negative, and inconclusive results remain part of the record. A result
is publishable only when its evidence meets the [evaluation protocol](docs/EVALUATION_PROTOCOL.md).
Do not silently replace runs or publish only favorable measurements.

## Training And Resume Artifacts

Supervised training writes `metrics.jsonl`, `batch_metrics.jsonl`, loss curves,
per-epoch `checkpoint_epoch_###.pt`, and `checkpoint_latest.pt`. The final
`checkpoint.pt` is written after the configured run completes. Interrupted runs
retain `status.json` and recoverable metrics/checkpoints.

Current training checkpoints include optimizer state and `global_step` for
resume through `resume_from_checkpoint`. Older weights-only checkpoints can
warm-start a run but reset the optimizer. Keep the exact continuation lineage;
missing ancestors must be disclosed rather than reconstructed as if original.

## Publishing Evidence

Publish a dated report under `reports/`, with machine-readable evidence under
`reports/data/` where available, and add a short entry to [RESULTS.md](RESULTS.md).
Each report owns the detailed protocol and limitations; the result index links
to it. Preserve older reports and the configs that reproduce them.

Raw datasets, caches, training checkpoints, and entire run directories stay
untracked. A published result may remove machine-specific paths or secrets;
record each transformation and the source checksum. Preserve game records,
scores, settings, and failure information. Do not fabricate missing hardware
or Git metadata. Research configs and historical source paths may refer to
local artifacts that must be regenerated or obtained separately.

## Published Model

The canonical inference artifact is `models_archive/resnet_c_epoch_030.pt`.
Its [manifest](models_archive/manifest.json) records selection, source and export
checksums, training lineage, dataset provenance, and evaluation references.
Verify it against `models_archive/SHA256SUMS`. The final scheduled epoch was
selected independently of Stockfish results.

Keep one compact default inference checkpoint in ordinary Git. Additional or
historical model binaries belong in GitHub Release assets, with checksums and
references in the model manifest and README. Dataset and training artifacts
remain outside Git.

The exporter keeps model state, configuration, and provenance, omits optimizer
state, rejects machine-absolute training paths, and refuses overwrites by default.
The recorded export command is:

```powershell
poetry run python scripts/export_model_artifact.py `
  runs/lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics/checkpoint_epoch_030.pt `
  models_archive/resnet_c_epoch_030.pt `
  --artifact-id resnet_c_epoch_030 `
  --exported-at 2026-08-11T20:46:01.694963+00:00
```

Published model filenames are immutable. `--overwrite` is reserved for local
byte-for-byte reproduction, never replacing a published model with new weights.
Publication checks cover CPU loading, finite tensor outputs, their documented
shapes, legal move selection, and held-out supervised evaluation. Model export
does not erase the source run's missing Git commit or overwritten ancestor
limitation; both remain in the manifest.
