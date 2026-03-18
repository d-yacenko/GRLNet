# Gold Improvement Recommendations

## Purpose

This note is a working document for improving `gold` accuracy in `GRLNet` under expensive `ImageNet-1000` training.

The main constraints are:

- long runs on expensive cluster hardware should not be wasted;
- interventions should be ranked by strength, not by curiosity value;
- it is highly desirable to reuse saved checkpoints and continue training instead of restarting from scratch;
- the main scientific question is not clean `val`, but whether the GRL mechanism actually improves the single-anchor `gold` protocol.

## Why `gold` is the key metric

For GRL, clean `val` is useful but secondary.

- `val` uses grouped tracks from `eval_root`;
- `gold` always applies notebook-style `prep_batch()` and converts the active third of the track into augmented views of a single anchor image;
- therefore `gold` is the closest metric to the claim that GRL can extract more useful signal from one image transformed into a pseudo-track.

If the comparison target is a classical single-image classifier, then `gold` is the important protocol, not plain grouped `val`.

## Current evidence

### Full `ImageNet-1000` run

Current trajectory on the large run:

```python
arr = np.array([
    [0.012, 0.055, 0.012],  # ep1
    [0.090, 0.199, 0.039],  # ep2
    [0.185, 0.316, 0.059],  # ep3
    [0.260, 0.361, 0.079],  # ep4
    [0.307, 0.402, 0.106],  # ep5
    [0.352, 0.441, 0.105],  # ep6
    [0.368, 0.450, 0.113],  # ep7
    [0.396, 0.493, 0.126],  # ep8
    [0.416, 0.474, 0.137],  # ep9
    [0.419, 0.536, 0.150],  # ep10
])
```

Columns are:

- `train`
- `val`
- `gold`

Important observations:

- `val` grows strongly: `0.055 -> 0.536` by epoch 10.
- `gold` also grows: `0.012 -> 0.150` by epoch 10.
- `gold / val` remains low:
  - epoch 5: `0.106 / 0.402 = 0.264`
  - epoch 10: `0.150 / 0.536 = 0.280`
- the ratio improves slightly, but the absolute gap `val - gold` stays large.

Interpretation:

- the model definitely learns class discrimination;
- the model also learns something useful for `gold`;
- but `gold` robustness/invariance is learned much more slowly than clean classification.

### Old `20-class` run

Old report:

- file: `rnn_conv_report_2026.03.06.19.09.55.pkl`
- same core architecture, much easier task: `20` classes from `ImageNet-1000`

Key values:

- best `val_acc`: `0.8359788` at epoch `20`
- best `gold_acc`: `0.6301370` at epoch `17`
- last epoch `20`: `train=0.7161`, `val=0.8360`, `gold=0.5890`

Interpretation:

- the architecture is capable of achieving strong `gold` on an easier problem;
- therefore the current weak `gold` on full `ImageNet-1000` is not enough evidence to say that GRL is fundamentally broken;
- but it is evidence that the current full-1000 recipe does not yet force strong enough `gold` behavior.

## Main concern: does GRL work at all?

### The concern

The concern can be phrased like this:

> Are we actually training GRL, or are we effectively training a plain shallow convolutional classifier with a classifier head, while the recurrent/track part contributes little?

This concern is valid.

### What the current curves do say

The current curves support the following claims:

- the model is not dead;
- the model is not failing to optimize on full `ImageNet-1000`;
- `gold` is not flat, it rises from `1.2%` to `15.0%` in 10 epochs;
- therefore the system is learning at least some behavior relevant to the `gold` protocol.

### What the current curves do not prove

The current curves do **not** prove that the GRL mechanism is being used strongly enough.

Why:

- `gold` is still much lower than `val`;
- a model can improve on `gold` even if it mostly relies on one strong anchor frame and only weakly uses track-style aggregation;
- therefore current `gold` growth is compatible with a partially degenerate regime where the model behaves closer to a shallow image classifier than desired.

### Bottom-line interpretation

The most defensible interpretation is:

- current data do **not** support the statement “GRL does not work at all”;
- current data **do** support the statement “the present training recipe does not yet make GRL-style invariance strong enough on full `ImageNet-1000`”.

In other words:

- the mechanism is likely not absent;
- the mechanism is likely under-realized.

## Can current numbers be explained by a plain 5-layer classifier?

### Short answer

Not conclusively, but the concern cannot be dismissed from the learning curves alone.

### What can be inferred without a new expensive training run

A plain image-dominant model would show a signature like this:

- clean `val` grows well;
- `gold` also grows somewhat because the anchor image still contains most class information;
- but additional frames in the pseudo-track add little real value.

This is close to what the current full-1000 curves look like:

- `val` is already strong;
- `gold` grows, but remains far behind.

Therefore:

- the current curves are **consistent with a weak-GRL regime**;
- they are **not sufficient** to prove that the track mechanism contributes strongly.

### Why this is still not enough to conclude “it is only a plain CNN”

There are two reasons not to jump to that conclusion.

1. On the easier `20-class` problem, the same core architecture reached strong `gold`.

That means the architecture can, in principle, learn a meaningful `gold` protocol.

2. `gold` on full-1000 is not flat.

If the recurrent/track machinery were completely useless, then one would expect either:

- very early saturation of `gold`, or
- much weaker continued growth.

Instead, `gold` keeps improving, just too slowly.

### Strongest low-cost falsification test

The best next test is **not** a new expensive training run of a new baseline from scratch.

The best next test is a **checkpoint-only evaluation ablation** on already trained weights.

Use the same saved checkpoint and compare:

1. normal `gold` protocol;
2. anchor-only protocol:
   - first active frame is kept;
   - all other active frames are replaced by exact copies of the anchor without augmentation;
3. reduced-track protocol:
   - keep only the anchor informative;
   - replace the rest with zeros or exact repeats;
4. shuffled-active-frames protocol:
   - same content, permuted order inside the active third.

Interpretation:

- if `gold` is close to anchor-only and close to reduced-track, then GRL contribution is weak;
- if `gold` is materially higher than anchor-only, then track aggregation contributes;
- if frame order hardly matters but multiple transformed frames matter, that still supports GRL as an order-light track integrator rather than a pure image classifier.

This test is cheap, reuses weights, and directly answers the scientific concern.

## Recommendation ranking

The list below is ordered by practical value under expensive training.

### Tier A: strong actions that preserve current expensive progress

#### A1. Do not interrupt the current expensive run only to chase speculative changes

Reason:

- the current run already provides useful signal;
- checkpointing works;
- the run is teaching us how `gold` evolves on full `ImageNet-1000`.

Practical conclusion:

- let the current long run continue to a meaningful checkpoint boundary;
- do not restart only because `gold` is currently lower than desired.

#### A2. Save and track `best_gold`, not only `best_val`

Current behavior:

- best checkpoint selection is by `val_loss`.

Problem:

- if `gold` is the main scientific target, then selecting only by `val_loss` can keep the wrong checkpoint.

Recommendation:

- keep `best_val`, but add `best_gold` as a separate artifact;
- do not replace the current logic, add a parallel `best_gold` checkpoint.

Why this is strong:

- cheap change;
- compatible with continuing from current weights;
- directly aligned with the real target metric.

#### A3. After the current run, perform checkpoint-only `gold` ablation tests

This is the highest-value diagnostic action.

Why:

- no retraining cost;
- directly addresses the “plain conv classifier” hypothesis;
- uses exactly the weights already paid for.

This should be done before any major architectural conclusion is made.

### Tier B: strong resume-compatible training actions

#### B1. Stage-2 `gold` fine-tuning from existing weights

This is the strongest training intervention that preserves current investment.

Recipe:

- take the best current checkpoint;
- reduce learning rate significantly, for example by `x10`;
- increase `train_gold_prob` from `0.5` to `0.8-1.0`;
- continue training for an additional late stage;
- monitor and save `best_gold`.

Why this is strong:

- it directly reduces the train/eval mismatch between train and `gold`;
- it reuses the current checkpoint;
- it focuses the model on the exact target protocol.

Expected effect:

- lower risk than restarting from scratch with a new recipe;
- best chance to lift `gold` materially with minimal wasted compute.

#### B2. Make the scheduler useful for late-stage `gold`

Current problem:

- if the scheduler acts too late, the model may spend too much time in high-LR classification mode and not enough time in a refinement regime.

Recommendation:

- when switching to stage-2 `gold` fine-tuning, use a lower LR from the beginning of that stage;
- optionally apply a more responsive LR schedule during the fine-tuning stage.

Why this matters:

- `gold` is typically a refinement objective, not just a raw optimization objective;
- lower LR is often where invariance improves.

#### B3. Choose final model for publication by `gold`, not by `val`

If the paper claim is about the GRL single-anchor protocol, the final reported model should be selected by that metric.

This does not prevent also reporting the best clean `val` checkpoint.

### Tier C: stronger but more invasive recipe changes

#### C1. Gold curriculum over the full run

Instead of `train_gold_prob = 0.5` for all epochs, use a schedule such as:

- early phase: lower or medium `gold` mixing;
- middle phase: stronger `gold` mixing;
- late phase: near-full `gold` mixing.

Why:

- early training should still learn class structure;
- later training should force invariance.

This can help, but it is less attractive than stage-2 fine-tuning because it is harder to apply without restarting from the beginning.

#### C2. More aggressive protocol-specific recipe tuning

Examples:

- modify augmentation strength inside `prep_batch()` or gold-like transforms;
- adjust optimizer/scheduler policy specifically for the `gold` stage;
- add additional diagnostics on how much the model uses non-anchor frames.

This is useful, but should come after checkpoint-only diagnostics and stage-2 fine-tuning.

## What should not be concluded too early

The following statements are currently too strong:

- “GRL is broken.”
- “This is just a plain CNN.”
- “`val` is enough, so `gold` can be ignored.”
- “If `val` is lower than AlexNet top-1, the approach is invalid.”

The most defensible current statement is:

- clean grouped classification is already working well;
- `gold` remains the bottleneck;
- the next step should be to measure how much of `gold` comes from true track behavior versus anchor-only behavior.

## Realistic outlook for `gold`

### Current recipe

If training continues with the same general behavior, then:

- `gold = 80%` on full `ImageNet-1000` is not a realistic expectation;
- even by `100` epochs, the current recipe more plausibly points to a much lower range.

This does not mean the model is bad.

It means:

- the current recipe is not yet optimized for the target metric.

### Practical expectation

The practical goal should be:

1. prove whether GRL contributes beyond anchor-only behavior;
2. lift `gold` with resume-compatible fine-tuning;
3. only then decide whether deeper recipe or architecture changes are justified.

## Recommended next sequence

Recommended order of work:

1. Keep the current expensive run as a baseline and finish a meaningful tranche of training.
2. Add `best_gold` checkpointing.
3. Run checkpoint-only `gold` ablation diagnostics on saved weights.
4. If GRL contribution is present but weak, launch stage-2 `gold` fine-tuning from the saved checkpoint.
5. If GRL contribution is negligible, then the next step is not “more epochs”, but a targeted protocol/architecture ablation.

## Bottom line

The current evidence suggests:

- the model is learning strongly on full `ImageNet-1000`;
- `gold` is improving, so GRL-like behavior is not absent;
- but `gold` is still too weak to claim that the current recipe fully realizes the GRL idea at scale;
- the most valuable next move is a cheap checkpoint-based falsification test and then a resume-compatible `gold` fine-tuning stage.
