# Model Sharing Plan

Status: pending

## Goal

Publish one curated ResNet-C inference checkpoint with the repository so a
fresh clone can load a playable model without first running supervised
training or downloading a separate training artifact.

The shared artifact is for inference and reproducibility. It must not be
described as the strongest checkpoint unless that claim is supported by a
documented comparison.

## Publication Gate

Do not publish the moving `checkpoint_latest.pt` file. Finish the current
ResNet-C run, then choose a publication checkpoint using a rule declared from
validation metrics. Record whether the selected artifact is the final epoch-30
checkpoint or an earlier validation-selected checkpoint.

Stockfish benchmark results must not be used for checkpoint selection. They
may be linked as external evaluation results after the checkpoint has been
selected.

The publication candidate must pass:

- CPU checkpoint loading
- finite policy and value output checks
- the `[batch, 4672]` policy and `[batch]` value shape contract
- a policy-only legal-move smoke test
- a recorded supervised evaluation on the intended held-out split

## Inference-Only Checkpoint

Export a new checkpoint from the selected training checkpoint. Do not copy the
training checkpoint directly.

Keep the fields required by `load_policy_value_checkpoint`:

- `model_state_dict`
- `model_config`
- `epoch`
- `metrics`
- `train_config`
- `saved_at`
- `completed_at`

Add an artifact schema version and provenance metadata if the loader is
extended to preserve them. Omit:

- `optimizer_state_dict`
- mutable resume state that is not needed for inference
- machine-specific absolute paths

The current ResNet-C training checkpoints are approximately 43 MB because they
contain optimizer state. The model tensors are approximately 14.4 MB, so an
inference-only artifact should be substantially smaller.

Add an export command, for example:

```powershell
poetry run python scripts/export_model_artifact.py `
  runs/<run-id>/checkpoint_epoch_030.pt `
  models_archive/resnet_c_epoch_030.pt
```

The exporter should refuse to overwrite an existing artifact unless an
explicit overwrite option is provided. Published filenames are immutable; a
new checkpoint gets a new filename.

## Repository Layout

Use:

```text
models_archive/
  README.md
  manifest.json
  resnet_c_epoch_030.pt
  SHA256SUMS
```

The final filename must reflect the checkpoint that was actually selected. Do
not retain `epoch_030` in the name if an earlier checkpoint is published.

`manifest.json` should record:

- artifact schema version and artifact ID
- checkpoint filename and SHA-256
- model family, preset, and complete model config
- source run ID and source checkpoint path
- epoch and validation metrics
- dataset manifest path and dataset description
- training config path, seed, and git commit when available
- export command and export time
- supervised and arena evaluation references
- intended input and output tensor contracts
- known limitations

`models_archive/README.md` should contain the shortest supported load and play
commands and explain that legal move masking remains mandatory.

## Git And GitHub Storage Policy

The repository currently ignores `*.pt`, so implementation must add a narrow
`.gitignore` exception for curated files under `models_archive/`. Do not
unignore training checkpoints under `runs/`.

Track one compact default inference checkpoint in ordinary Git so a normal
clone is immediately usable without Git LFS or a separate download. Keep
historical or additional checkpoints as GitHub Release assets and reference
them from `models_archive/README.md` and the manifest rather than adding every
binary to Git history.

If future model artifacts no longer fit this policy, reassess Git LFS or
release-only downloads. GitHub currently warns for regular Git files above
50 MiB and blocks files above 100 MiB:

<https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>

GitHub Releases are the preferred location for additional versioned binaries:

<https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases>

## Documentation And Config Integration

Update:

- `README.md` with the included model and a minimal play command
- `docs/guides/HOW_TO_PLAY.md` with the archive checkpoint path
- `docs/ARCHITECTURES.md` with the published preset and artifact identifier
- `EXPERIMENTS.md` with the source run and evaluation references
- `REPRODUCIBILITY.md` with the export and checksum procedure

Add new demo configs that reference the archived checkpoint when useful. Do not
rewrite historical evaluation configs whose original paths are part of their
recorded protocol.

## Tests

Add focused tests for:

- inference export omits optimizer state
- exported and source models produce identical outputs for the same input
- the exported artifact loads on CPU through the public checkpoint loader
- manifest model config and checksum match the binary
- a policy-only bot using the archived artifact returns a legal move
- the exporter rejects accidental overwrite by default

Exporter tests should use a small temporary checkpoint. A separate archive
smoke test may load the committed artifact without duplicating it in test
fixtures.

## Integrity And Provenance

Generate the checksum after the final file has been written and verify it
before publishing. Record the exact source checkpoint checksum as well as the
exported artifact checksum so the transformation can be audited.

Document the human-game dataset provenance and applicable license or terms
note. Do not imply that the artifact contains Stockfish, Syzygy, external
engine labels, or pretrained engine weights.

## Done Criteria

- one immutable inference-only ResNet-C checkpoint is tracked under
  `models_archive/`
- the artifact contains no optimizer state and loads with the public loader
- `manifest.json` and `SHA256SUMS` match the committed file
- a fresh clone can run the documented policy-only example without training
- relevant tests, `poetry run pytest`, Ruff, and mypy pass when configured
- README, play, architecture, experiment, and reproducibility docs reference
  the artifact
- additional checkpoints are kept out of ordinary Git history
- checkpoint selection and evaluation claims remain documented and measured
