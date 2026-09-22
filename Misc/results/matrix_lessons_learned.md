# Comprehensive Ablation Matrix — Lessons Learned (2026-08-24)

Anchor: `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv`, best val_miou **0.8292 @ epoch 194**.
Config: `--num_classes 17 --dilation_gating on --color_style mosaic --area_thresholds 20 90 180 --early_bias_power 1.5 --whole_tooth_patch_prob 0.0`.

Each matrix run changes exactly ONE axis from this anchor. Full per-run metrics are in `patch_experiment_results.csv`; this file is the synthesis/interpretation layer on top of that data.

## Complete, clean comparisons (both runs finished 200 epochs)

| Change from anchor | Result | Δ vs anchor | Lesson |
|---|---|---|---|
| **Gating OFF** | 0.8396 | **+0.010** | Area-gating the dilated blocks isn't earning its keep — turning it off is at least as good, maybe slightly better. Since gating exists purely to save compute on small patches, if it's not helping accuracy, that's a real case for dropping it as unnecessary complexity, not just tolerating it. |
| **RGB = flat** | 0.8353 | **+0.006** | The simpler flat color (one shared draw per class-group) does *at least* as well as mosaic on this validation metric — the added complexity of per-face HSV noise isn't clearly buying anything here. Worth remembering this is val_miou only, though — mosaic's whole rationale was matching real per-face *live* variance, which this synthetic validation set can't test either way. |
| **Dilation_ks widened** | 0.8370 | **+0.008** | Modest, real gain from wider local context. Doesn't by itself confirm the premolar-confusion hypothesis it was designed to test — that needs a live/per-class check — but the wider context isn't hurting and mildly helps overall. |
| **Whole-tooth ON** | 0.8210 | **−0.008** | Now a *clean* answer to a question that was inconclusive at 100 epochs: with both configs given the full 200-epoch budget, whole-tooth curriculum doesn't help scan-sweep validation performance, and mildly hurts. Reasonably conclusive this time — the earlier ambiguity was genuinely about epoch budget, not about whole-tooth secretly winning later. |
| **Color dropout ON** | 0.8208 | −0.008 (not apples-to-apples) | This run's val set *also* gets color dropout (by design), so it's being scored on a harder task, not a fair equal-footing comparison. Reading it as "only −0.008 despite a meaningfully harder validation distribution" is actually a decent sign that the robustness fix isn't costing much. |
| **Transformer early-only** | **0.8472** | **+0.018** | Best result in the whole matrix. Real, clean win from adding global attention. |
| **Transformer early+late** | 0.8400 | +0.011 | Beats anchor, but *loses* to early-only (0.8472). At 100 epochs early+late was ahead of early-only (0.8200 vs 0.7992) — that ordering **flips** by 200 epochs. The second (late-stage) attention pass looks like it may be adding capacity the model doesn't need, or converges slower/less cleanly than the simpler early-only version. |

## Incomplete runs — compared fairly against the anchor's *own* value at the same epoch, not its final 200-epoch best

| Change from anchor | Stopped at | Its best-so-far | Anchor's best within same epoch budget | Lesson |
|---|---|---|---|---|
| **RGB = none** | epoch 118 | 0.6713 | 0.7635 | **Real, large gap (−0.092) even at equal training budget.** This isn't an artifact of stopping early — RGB is doing serious work. Worth finishing this run to get the final number, but the direction is already clear. |
| **area_thresholds = default** (early_bias stays tuned) | epoch 133 | 0.8168 | 0.7887 | **Ahead of anchor by +0.028 at the same point.** Reverting area_thresholds alone isn't hurting — it may be helping more than the "fully tuned" combo at this stage of training. |
| **early_bias_power = default** (area_thresholds stays tuned) | epoch 116 | 0.7777 | 0.7635 | **Slightly ahead too (+0.014).** Same story: reverting just this one knob isn't a loss at this point in training. |

That last pair is the most surprising finding in the whole set: **the original "tuned" combination of area_thresholds + early_bias_power together doesn't clearly beat reverting either one alone**, at least through ~120-130 epochs. The historical comparison that justified tuning both (0.6505 baseline → 0.77-0.83 tuned) also changed the color scheme at the same time, so it never isolated these two knobs from each other or from color. This matrix is the first time it's been isolated, and the early signal says the interaction might not be what was assumed. Both runs need to finish for a confident answer, but resuming these two and RGB-none should be prioritized over anything else — they're the ones actually changing the picture, not just confirming it.

## Bottom line

Transformer early-only is the strongest single change (+0.018, and now the best-known config on this metric), gating looks safely droppable, and the area_thresholds/early_bias_power tuning is the one piece of received wisdom this matrix is actively calling into question rather than confirming.
