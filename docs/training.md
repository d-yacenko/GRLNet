# Training Recipe

The packaged recipe is available as:

```bash
grlnet-train-imagenet --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml
```

The recipe expects ImageFolder-style directories:

```text
train_root/class_name/*.JPEG
eval_root/class_name/*.JPEG
```

## Default ImageNet-1K Recipe

- optimizer: SGD with Nesterov momentum;
- base learning rate: `0.08`;
- warmup: `5` epochs;
- scheduler: warmup plus cosine decay with non-zero tail;
- weight decay: `1e-4`;
- label smoothing: `0.05`;
- mixup: `alpha=0.2`, `prob=0.5`;
- EMA decay: `0.999`;
- AMP: enabled on CUDA;
- channels-last memory format: enabled;
- auxiliary supervision: last recurrent steps, weight decays from `0.2` to `0.05`;
- checkpoint payload includes `model`, `ema_model`, optimizer, scheduler, scaler and history.

The validation path evaluates the EMA model. For inference and published
weights, prefer `ema_model` from training checkpoints.

## Resume

```bash
grlnet-train-imagenet \
  --config src/grlnet/recipes/imagenet/configs/stabhrec40_a100_single_50e.yaml \
  --resume auto \
  --output-dir runs/stabhrec40_a100_single_50e
```

`--resume auto` looks for `<checkpoint_prefix>_latest.pth` in the output
directory.
