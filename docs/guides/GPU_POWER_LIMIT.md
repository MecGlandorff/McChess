# Limiting GPU Power During Training

This guide uses `nvidia-smi` to cap an NVIDIA GPU power limit during local
training. It is a hardware protection and noise/heat control helper; it does
not change model code or training results by itself.

Requirements:

- NVIDIA GPU and driver
- `nvidia-smi` available on `PATH`
- Administrator PowerShell on Windows when setting power limits

## Check GPU Status

```powershell
poetry run python gpu_protect --status
```

## Turn Protection On

The default protected setting is 75% of the GPU default power limit:

```powershell
poetry run python gpu_protect --on
```

For a specific GPU or percent:

```powershell
poetry run python gpu_protect --on --gpu 0 --percent 75
```

Preview the command without changing the GPU:

```powershell
poetry run python gpu_protect --on --dry-run
```

## Turn Protection Off

This restores the GPU default power limit reported by `nvidia-smi`:

```powershell
poetry run python gpu_protect --off
```

## Training Workflow

Use the limiter before starting a CUDA training run:

```powershell
poetry run python gpu_protect --on
poetry run python scripts\train_supervised.py configs\train\lichess_2013_01_full_epoch1.yaml
poetry run python gpu_protect --off
```

Confirm that training actually uses the GPU. The training script prints the
resolved device near startup; the limiter only matters when it prints
`device=cuda`.

Power limits can reset after reboot, driver restart, or some system sleep/resume
events, so check `--status` before long runs.
