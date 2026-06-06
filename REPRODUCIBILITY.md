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

If a run is interrupted between epochs or during a later epoch, `status.json`
should still identify the latest completed epoch and the latest checkpoint path.
The batch-level metrics may include logged train loss from the interrupted epoch.

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
