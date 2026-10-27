# Model Selection

Training configs choose one built-in preset or an inline `model` mapping:

```yaml
model_preset: resnet_c
```

Do not set both. [ARCHITECTURES.md](../../docs/ARCHITECTURES.md) lists current
preset dimensions and normalization; the model config saved in a checkpoint
is authoritative for historical runs.
