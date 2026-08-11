# Reproducibility

McChess will be easy to rerun at three scales:

1. smoke: fast enough for local development
2. reportable: large enough to support a result table
3. archival: enough metadata to revisit a run later

## Required Run Artifacts

Each training or evaluation run should create a run directory containing:

- `config.yaml` or `config.json`
- `metrics.jsonl`
- `manifest.json` for datasets, when applicable
- checkpoint path or explicit `null` if no checkpoint is produced
- evaluation result JSON, when applicable
- random seed
- git commit if available
- hardware notes when available
- start time, end time, and status

Supervised training writes recoverable artifacts after every completed epoch:

- `batch_metrics.jsonl` for train loss at `log_every_steps` intervals
- `batch_loss.svg`, refreshed during epochs from batch-level train metrics
- `checkpoint_epoch_###.pt` for the model after that epoch
- `checkpoint_latest.pt`, overwritten with the newest completed epoch
- `loss.svg`, refreshed after each completed epoch
- `checkpoint.pt`, written only after the configured run completes

Current supervised checkpoints also store `optimizer_state_dict` and
`global_step`, so a later run can resume optimizer moments and epoch numbering
when `resume_from_checkpoint` is set. Older checkpoints without optimizer state
remain usable as model-weight warm starts, but the optimizer is reset.

If a run is interrupted between epochs or during a later epoch, `status.json`
should still identify the latest completed epoch and the latest checkpoint path.
The batch-level metrics may include logged train loss from the interrupted epoch.

## Published Inference Artifacts

Training checkpoints are local resumable state and remain outside normal Git
history. A published model is exported separately with
`scripts/export_model_artifact.py`. The exporter keeps model state and audit
metadata, omits optimizer and global resume state, rejects machine-absolute
training paths, and refuses to overwrite an existing artifact by default.

The canonical artifact is `models_archive/resnet_c_epoch_030.pt`. Verify it
against both `models_archive/SHA256SUMS` and `models_archive/manifest.json`.
The manifest records the source-checkpoint checksum as well as the exported
checksum, so the transformation remains auditable without committing the full
resumable checkpoint.

Recreate the artifact from the named local source checkpoint with:

```powershell
poetry run python scripts/export_model_artifact.py `
  runs/lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics/checkpoint_epoch_030.pt `
  models_archive/resnet_c_epoch_030.pt `
  --artifact-id resnet_c_epoch_030 `
  --exported-at 2026-08-11T20:46:01.694963+00:00
```

Published artifact filenames are immutable. Use `--overwrite` only when
reproducing a byte-for-byte artifact locally, never to replace a published model
under the same identifier.

## Tiny End-To-End Reproduction

The repository will maintain a tiny path that can run on CPU:

1. build a small dataset from a fixture PGN
2. train a model for a few steps or overfit a tiny sample
3. load the checkpoint
4. run a short arena against a simple baseline
5. write metrics and results to disk

This path is for correctness and reproducibility, not strength.

## Reportable Result Reproduction

A reportable result should include:

- dataset manifest path
- model config path
- training config path
- evaluation config path
- checkpoint path
- command used to run the job
- seed
- git commit if available
- date run
- hardware notes

The result should be reproducible from repository files plus the referenced dataset source.

## Result Status

Every experiment should end with one of:

- `completed`
- `failed`
- `aborted`
- `inconclusive`

Failed or inconclusive runs should record the reason. Do not remove them only because the result is weak.

## Expected Limits

The project should not require large-scale compute for its core claims.

If a run requires GPU hardware, large storage, or long wall-clock time, document that requirement before presenting the result as reproducible.
