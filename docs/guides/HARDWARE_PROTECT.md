# Limiting CPU And GPU Hardware During Training

`hardware_protect` applies CPU and NVIDIA GPU limits as one local workflow. A
single restore-state file records the original Windows CPU power-plan values and
the selected GPU limit mode so `--off` can restore both components without
requiring the original command again.

The helper controls frequency, boost, power, and clock settings. It is not a
closed-loop temperature controller. Continue to monitor temperatures while a
long run is active; firmware thermal limits remain the final protection layer.

Requirements:

- Windows 10 or Windows 11 for CPU limiting through `powercfg`
- NVIDIA GPU and driver for GPU limiting through `nvidia-smi`
- Administrator PowerShell when the driver or power plan requires elevation

## Check Status

```powershell
poetry run python hardware_protect --status
```

Inspect only one component when needed:

```powershell
poetry run python hardware_protect --status --component cpu
poetry run python hardware_protect --status --component gpu
```

Status shows the active CPU power plan, its AC/DC maximum-frequency and boost
values, NVIDIA temperature/power/current-clock data, and whether a unified
restore state exists.

For additional live NVIDIA data:

```powershell
nvidia-smi dmon -s pucvmt -d 1
```

Use the laptop vendor's utility or another hardware monitor for CPU package
temperature because `powercfg` does not expose a reliable CPU temperature.

## Preview The Local Laptop Limits

For the local Ryzen 7 7735HS and RTX 4060 Laptop GPU, preview a 3200 MHz CPU cap
and 1500 MHz GPU graphics-clock cap:

```powershell
poetry run python hardware_protect --on --cpu-max-frequency-mhz 3200 --gpu-limit-mode clock --gpu-clock-mhz 1500 --dry-run
```

Dry-run mode inspects the required values but does not change hardware or write
`.local/hardware_protect.json`.

## Turn Protection On

Apply both limits before a long CUDA training run:

```powershell
poetry run python hardware_protect --on --cpu-max-frequency-mhz 3200 --gpu-limit-mode clock --gpu-clock-mhz 1500
```

The CPU path:

- saves the active plan's exact AC/DC maximum-frequency and boost-mode values
- requests a 3200 MHz maximum for AC and DC
- disables processor boost for AC and DC
- activates the updated plan and verifies the reported values

The GPU clock path sends `nvidia-smi -lgc 1500`. On this laptop, 1500 MHz is a
practical starting point for a long run. A 1800 MHz target is a faster option if
temperature and noise are acceptable:

```powershell
poetry run python hardware_protect --on --cpu-max-frequency-mhz 3200 --gpu-limit-mode clock --gpu-clock-mhz 1800
```

Only one protected session may be active. Run `--off` before applying a
different target. This prevents the saved original values from being
overwritten by already-limited values.

## GPU Power-Limit Mode

Power mode is the default and computes a watt target from the GPU's default
power limit:

```powershell
poetry run python hardware_protect --on --cpu-max-frequency-mhz 3200 --gpu-percent 75
```

This sends `nvidia-smi -pl <watts>`. On some laptop GeForce systems under
Windows WDDM and OEM firmware control, the command can report success without
enforcing the requested watt cap. Actual draw may then exceed the target. Use
explicit clock mode when that occurs.

Clock mode can also derive a supported clock from a percentage of the maximum:

```powershell
poetry run python hardware_protect --on --gpu-limit-mode clock --gpu-percent 50
```

For the local RTX 4060 Laptop GPU, explicit targets are easier to audit than a
percentage once a suitable clock is known.

## Protect One Component

The default `--component all` applies CPU and GPU protection together. CPU-only
and GPU-only sessions are available for machines or workloads that need one
side:

```powershell
poetry run python hardware_protect --on --component cpu --cpu-max-frequency-mhz 3200
poetry run python hardware_protect --on --component gpu --gpu-limit-mode clock --gpu-clock-mhz 1500
```

The restore file records which components were selected.

## Turn Protection Off

```powershell
poetry run python hardware_protect --off
```

The command reads `.local/hardware_protect.json`, restores the CPU plan's saved
AC/DC values, and uses the recorded GPU mode:

- power mode restores the NVIDIA default power limit
- clock mode resets the NVIDIA graphics-clock lock

If the active Windows power plan changed after protection was enabled, the
saved plan is restored without switching the machine back to it.

The command attempts both restores even if one component fails. It removes the
state file only after all selected components restore successfully. On partial
failure, fix the reported permission or driver problem and run `--off` again.

## Training Workflow

```powershell
poetry run python hardware_protect --on --cpu-max-frequency-mhz 3200 --gpu-limit-mode clock --gpu-clock-mhz 1500
poetry run python scripts\train_supervised.py configs\train\lichess_2013_01_full_epoch1.yaml
poetry run python hardware_protect --off
```

Confirm that training prints `device=cuda` when applying a GPU limit. CPU
protection can still matter during data loading, cache construction, CPU-only
training, and evaluation.

Power and clock settings can be reset by reboot, driver restart, GPU
reinitialization, or OEM power-profile changes. Check status and measured values
before a long run.

## Local CPU Temperature Context

The Ryzen 7 7735HS has a 3.2 GHz base clock, boost up to 4.75 GHz, and an AMD
maximum operating temperature of 95°C. An observed 85°C is below that specified
maximum, but reducing sustained frequency can lower fan noise and chassis heat
during long runs. If 3200 MHz does not lower temperature enough, reduce the cap
in small steps and record both training throughput and temperature.

Windows treats `PROCFREQMAX` as a requested maximum processor performance state.
Firmware and hardware control still decide measured instantaneous clocks, so
verify the behavior under load.

References:

- [Microsoft MaxFrequency setting](https://learn.microsoft.com/en-us/windows-hardware/customize/power-settings/options-for-perf-state-engine-maxfrequency)
- [Microsoft powercfg command-line options](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/powercfg-command-line-options)
- [AMD Ryzen 7 7735HS specifications](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-7-7735hs.html)
