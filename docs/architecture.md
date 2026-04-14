# GRLNet / StabHRec40 Architecture

The current GRLNet release is centered on the StabHRec40 architecture. It is no
longer the older lattice/track model.

## Forward Path

```text
image [B, 3, H, W]
  -> stem: ConvGNAct(3->64, k5, s2)
  -> stem: ConvGNAct(64->64, k3, s1)
  -> max pool
  -> stem: ConvGNAct(64->64, k3, s1)
  -> h_seed: Conv2d(64->192) + GroupNorm + SiLU
  -> recurrent cell, shared weights, 12 steps
  -> GAP(H_final) concat GAP(C_final)
  -> LayerNorm + MLP classifier
```

The recurrent cell computes ConvLSTM-like gates from the H stream only:

```text
i, f, o, g = split(Conv(SiLU(GroupNorm(H_{t-1}))))
C_t = sigmoid(f + forget_bias) * C_{t-1} + sigmoid(i) * tanh(g)
hidden_t = sigmoid(o) * tanh(GroupNorm(C_t))
delta_t = DeltaBranch(hidden_t)
H_t = H_{t-1} + sigmoid(a) * hidden_t + sigmoid(b) * delta_t
```

Where `a = hidden_scale` and `b = delta_scale` are learnable scalar gates.

## Inference Contract

```python
logits = model(images)  # images: [B, 3, H, W]
```

Training can request auxiliary logits:

```python
main_logits, aux_logits = model(images, return_aux=True)
```

Auxiliary logits are produced from the last recurrent readouts and are used only
for training supervision.
