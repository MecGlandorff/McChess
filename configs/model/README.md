# Model Configs

Model architecture configs belong here.

Current built-in presets are exported by `mcchess.model`:

- `resnet_baseline`: existing default single-board `PolicyValueResNet`
  configuration.
- `resnet_b`: deeper single-board ResNet with `channels=64`, `num_blocks=6`,
  and `value_hidden_dim=128`.

Training configs may select a preset with:

```yaml
model_preset: resnet_b
```

Do not set both `model_preset` and an inline `model` mapping in the same
training config.
