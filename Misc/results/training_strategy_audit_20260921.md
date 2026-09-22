# Training strategy audit — 2026-09-21

The strongest next step is to retain HSV-generated color and the early-global transformer, then improve rare-class loss weighting, deployment-specific sampling, and tooth-instance separation. Do not combine all previously promising switches at once: the existing matrix does not establish their interactions.

Scope: inspected all 20 local checkpoint configurations, the 19 available TensorBoard histories, launch configurations, dataset/sampling/loss/model code, and live guidance code. This is an audit of recorded results and implementations, not a new common-set evaluation of model predictions. No training was launched and no training/inference code was changed. Per-class and live failure metrics are not available in the existing event logs.

## What the experiments support

The server HSV dilated baseline has best validation mIoU 82.92%, final 71.62%, and last-20 mean 79.30%. The following are single-run observations, not statistically established improvements. Class weights were independently estimated from random patches, which further weakens strict one-variable causal attribution.

| Change | Evidence | Decision |
|---|---|---|
| Early-global transformer | Best 84.72%; last-20 82.29% vs baseline 79.30% | Strongest existing architecture signal; keep as reference |
| Add late-global transformer too | Best 84.00%; last-20 81.50%; worse than early-only at 200 epochs, although it won the local 100-epoch comparison | Do not add depth by default |
| Gating off | Best 83.96%; last-20 81.01% | Promising on the dilated model, but transformer × gating-off is untested |
| Larger dilation neighborhoods (200/1200/3000 vs 200/900/1800) | Best 83.70%; last-20 79.05% vs 79.30% baseline; final falls to 70.81% | No sustained advantage demonstrated; secondary priority |
| Flat color instead of HSV mosaic | Best 83.53%; last-20 80.76% | Synthetic validation does not prove HSV superior to flat; retain HSV based on live-domain plausibility and user observations |
| Remove color | Best 67.13% through epoch 117 vs HSV baseline 76.35% over that same epoch budget | Strong evidence color matters on this synthetic evaluation; not proof it resolves tooth identity |
| Whole-tooth probability 0.3 | Best 82.10%; last-20 78.97% vs baseline 79.30% | No demonstrated aggregate gain; no instance/boundary metric was measured |
| Color dropout 0.6 | Best 82.08%, but validation also has dropout | Robustness intervention, not a directly comparable lower score; test all candidates on the same color conditions |
| Default area thresholds 40/180/360 instead of 20/90/180 | `earlyBiasTunedOnly`: best 77.77% through epoch 115 vs baseline 76.35% at that budget | Modest promising incomplete observation; no support for claiming tuned lower thresholds are superior |
| Early-bias power 2.5 instead of 1.5 | `areaThreshTunedOnly`: best logged 81.68% through epoch 132 vs baseline 78.87% at that budget | Inconclusive: it changes validation patches as well as training. Available best checkpoint is only 80.14% at epoch 114 |
| Midline continuation | Best 84.78%; last-20 81.34% | Strong local checkpoint, but not proof midline improved over the server transformer (different dataset sizes) |
| Midline 0.3 from scratch | Best 78.29%; last-20 75.85% | Worse than local midline continuation by 6.49 best-mIoU points / 5.49 mean points; do not repeat this recipe unchanged |

HSV is used to synthesize RGB input channels; the network is not directly given HSV channels. All tooth IDs share the tooth color distribution, distinct from gum. This helps tooth-vs-gum discrimination but does not directly distinguish premolar 4 from 5 or molar 15 from 16. The synthetic color comes from ground-truth labels, so validation on synthetic color alone is an optimistic proxy for real missing/specular/contaminated color.

**Correction to the previous results summary:** `Misc/results/patch_experiment_results.csv` and `matrix_lessons_learned.md` reverse the interpretation of the two TunedOnly runs. Console Namespace arguments, launch scripts, and checkpoint area_thresholds agree: `areaThreshTunedOnly` means tuned area 20/90/180 with DEFAULT power 2.5; `earlyBiasTunedOnly` means DEFAULT area 40/180/360 with tuned power 1.5. Interpret the metrics using these actual settings. The old files are retained as historical artifacts and are not authoritative on these two descriptions.

## What the four uncertain knobs actually do

**area_thresholds:** hard on/off thresholds for three dilated blocks using accumulated surface area in mm². At 50 mm², 20/90/180 and 40/180/360 both activate only block 1. At 100 mm² the tuned setting activates two blocks, defaults one. Neither creates geometry that has not been scanned. The transformer is always active, independently of these gates. Documented premolar confusion at >4,000 mm² occurs with all gates already on, so lowering thresholds cannot directly address that example.

**early_bias_power:** samples `int(U**p * (n_stages-1))`. Higher p means more early/small patches, not later ones. On the continuous normalized stage coordinate, p=1.5 places about 63% of probability in the first half, p=2.5 about 76%; actual discrete bins differ. This emphasizes early scans while reducing late/posterior exposure. It applies to the chosen midline, whole-tooth, or scan-sweep branch, so sampling knobs interact. It currently also changes validation, which prevents a clean training ablation.

**midline_seed_prob:** per-example chance of choosing a symmetric 8/9-centered footprint (8–22 mm nominal widths, 8 growth stages), falling back to scan-sweep when a central incisor is missing. It is not a midline loss, auxiliary head, epoch schedule, or guaranteed balanced number of examples. At 0.3, it substitutes for roughly 30% of scan-sweep opportunities in eligible arches. Combined with early bias, it concentrates exposure on incisors and can reduce posterior coverage. A lower or scheduled probability is a hypothesis to test, not an established optimum.

**whole_tooth_patch_prob:** replaces some footprint-based examples with cumulative unions of ground-truth tooth masks plus a 1.5 mm margin. Growth still begins at an incisor and uses the same early-stage bias. It does not uniformly target difficult posterior pairs, add tooth counting, or train an instance-separation objective. The current CLI rejects positive whole-tooth and midline probabilities together; that combination has not been evaluated.

## Findings that matter before another architecture sweep

1. **Rare-class weighting is a plausible driver of terminal-molar false positives.** In the scratch checkpoint, alpha(1)/alpha(2)=37.4 and alpha(16)/alpha(15)=32.7. In the continuation these ratios are 29.3 and 24.8; the no-midline server transformer has 20.1 and 18.2. `dataset/patch_losses.py` estimates uncapped inverse frequency from 200 random training patches. For equally difficult faces, a true third-molar face can contribute tens of times the loss of a true neighboring second-molar face. Importantly, alpha is indexed by the TRUE label: falsely predicting 16 on a true-15 face uses alpha(15), not alpha(16). This creates an asymmetric training pressure, but it is not proof of the cause. Test softened weighting while measuring both terminal precision and recall.
2. **Current train/validation split is not case-independent.** There are 1,200 train / 600 validation arches, with 791 / 491 filename case IDs and 382 IDs in both. There are zero identical mesh filenames across splits; overlap is the upper/lower partner arch. If these filename IDs are patient IDs, this is patient-level leakage. Only 109 validation case IDs (218 arches) are absent from the current training split. Preserve exact manifests and verify patient metadata before rebuilding splits. These counts describe current local data, not an independently verified historical server manifest.
3. **Training randomness is not controlled by the printed seed alone.** `_rng_for` uses entropy-seeded `np.random.default_rng()` for training. The collator does likewise without a seed; class-alpha sampling seeds only arch selection, not the resulting patches. Store RNG states and use explicit per-worker/per-epoch/per-sample seeds for reproducible but diverse draws. Keep one fixed alpha vector when isolating architecture or sampler changes.
4. **The final growth stage is unreachable in the current sampler.** For U<1 and n>1, floor(U**p*(n-1)) cannot return n-1. The 8-stage midline sampler never chooses stage 7; the whole-tooth branch never selects its final cumulative stage. Other stages can still contain terminal teeth, so this does not establish that molars are never seen. Use an explicit distribution over all stages, then audit actual tooth/area/visibility exposure.
5. **Validation changes with some training switches.** Early-bias power changes the patch, as verified on actual validation examples (one arch changes from 1,780 to 665 mm²). Color style/dropout changes the input as well. Midline and whole-tooth validation probabilities are zero. Create a fixed independent validation manifest covering all relevant modes rather than letting experiment flags define its contents.
6. **The live decision discards uncertainty.** `realtime/mesh_viewer_segmented.py` takes argmax and passes labels to guidance. Existing confirmation counts thresholded faces over consecutive cycles, then retains confirmations permanently. A persistent false region can be confirmed; reobserving the same accumulated faces is not independent new evidence. Keep probability margins, physical area, connected support, visibility, and new-observation evidence; allow revision. Class presence is not scan completeness. Missing terminal teeth must be allowed rather than forcing a 16-tooth completion target.
7. **Temporal instability can arise without new geometry.** Live preprocessing/capping and neighbor precomputation use unseeded sampling. Test repeated identical mesh inference before attributing every label flip to the learned model. Use stable voxel/face sampling or deterministic seeds for fixed geometry. The installed FPS uses the compiled operator; the random-start Python fallback should be separately controlled if used elsewhere.
8. **Checkpoint bookkeeping needs verification.** Scratch `last.ckpt` contains epoch 186, although logs reach 199; continuation `last.ckpt` contains 188. The older source transformer `last.ckpt` behavior explains why continuation logs begin at 89 despite comments claiming a start at 100. Also, the power-2.5 (`areaThreshTunedOnly`) run logs 81.68% but the available best file is 80.14%. Do not assume a filename implies a recoverable final/best state; verify epoch, optimizer, scheduler and callback state on save and resume.

## Direct interventions for the three live failures

| Failure | Training intervention | Architecture/inference intervention | Primary evaluation |
|---|---|---|---|
| Early wrong tooth identities | Fixed small-patch/midline benchmark; varied offsets, incomplete crowns, real color failures; explicit early-stage strata rather than one global bias | Keep early transformer; allow uncertain tooth identity; optional coarse tooth-type head and visible-class head | False tooth IDs per early patch; non-gum precision; coverage-vs-error curve |
| Premature 1/16 | Soften inverse-frequency alpha; include true-present, absent-from-patient, and present-but-not-yet-visible cases; mine false 1/16 predictions on other teeth/gum from training data only | Calibrated visible-presence score and spatial support; tentative vs confirmed identity; no hard posterior ban based on scan order | Absent-class false-positive area, region precision/recall for 1/16, premature guidance-completion rate |
| Two teeth merged under one label | Partial adjacent pairs 4/5, 12/13, 1/2, 2/3, 14/15, 15/16 with realistic scanner margins; pair-separation/boundary supervision | Auxiliary instance embedding or boundary/center head, followed by anatomically informed numbering with missing teeth allowed | Pair-merge rate, instance matching, per-tooth IoU, boundary F1, number-assignment accuracy |

An auxiliary presence head must predict which teeth are actually VISIBLE in the patch, not which exist in the complete ground-truth arch. Hard-negative mining must not teach that 1/16 can never appear in a small patch: small true molar captures also need representation. Morphology/position priors should allow missing teeth and uncertain arch orientation.

For merging, tooth-number semantic labels already provide different targets for neighboring teeth, but the current model has no explicit instance-separation objective. Distinguish two cases in replay: an actual boundary failure versus two geometrically separate crowns receiving the same number. Instance embeddings/boundaries address separation; relative position and instance-level numbering address identity. A coarse type head alone cannot distinguish two premolars of the same type.

## Architecture order of priority

1. Keep the current early-global transformer and local geometry backbone. Add a lightweight boundary or discriminative instance-embedding head before adding a second large transformer. Use existing face labels and mesh adjacency to derive supervision on training arches. Prefer boundary-aware sampling that can also operate without ground-truth labels at inference; do not leak GT boundaries into test-time sampling.
2. Give the existing attention block explicit RELATIVE spatial relations: pairwise distances, normal relations, and/or relative coordinates in an appropriate local frame. It currently uses positions for FPS but does not explicitly add positional relations inside attention. Geometry is already present in upstream features, so it is not geometry-blind. This is a measured ablation, not a promise; distances are rotation-invariant, coordinate offsets are not and need augmentation/frame handling. Avoid tying identity to robot/world coordinates.
3. Add optional visible-tooth presence / coarse tooth-type auxiliary heads. Use soft priors rather than irreversible logit masks. An upper/lower arch token is inexpensive if reliable arch identity is available at inference, but has no supporting ablation yet; report upper-arch results separately before splitting models.
4. If same-geometry replay is stable but growing scans remain unstable, test temporal consistency on registered shared surfaces, masking newly revealed/ambiguous boundaries. Do not enforce unchanged predictions where new context correctly resolves an earlier mistake. Stable sampling and probability fusion are simpler first steps than a recurrent model.
5. Transformer + gating-off is a useful small architecture experiment after measurement fixes. Wider neighborhoods and a second transformer are lower priority: they have not demonstrated sustained gains over the early-only model. Track latency and GPU memory as well as accuracy.

## Training and evaluation methodology

- Split by verified patient/case, keeping both jaws and all crops, sequences, augmentations and live captures together. Use separate train, tuning/calibration, and untouched test cases. Existing checkpoints can first be compared on the 218 locally unseen-case validation arches, but check historical training manifests before calling these unseen for every model. Independently held-out real captures are the stronger deployment test.
- Freeze evaluation patches independently of training flags: tiny 8/9 patches; ordinary growing sweeps; partial adjacent posterior teeth; terminal absent/not-yet-visible/visible; missing teeth; real or measured noisy/missing color. Include upper-arch-specific and stage/area breakdowns. Maintain one legacy set for historical continuity, but use the fixed deployment suite for selection.
- Record per-class precision, recall and IoU, including confusion pairs; tooth-instance merge/split errors; false-positive physical area when a class is absent; label flips on registered common surfaces; guidance mistakes and detection delay. Macro mIoU across the entire epoch can conceal the few faces that prematurely stop a robot.
- Use true physical area as well as face-count metrics: density/capping changes the number of faces. For early patches without GT labels 1/16, separately report total predicted 1/16 mm² and whether an erroneous region would cross the deployed confirmation rule.
- Evaluate class weighting as its own factor: existing inverse frequency vs square-root inverse frequency, then consider a capped or effective-number alternative. Normalize consistently, estimate from a fixed sufficiently large training sample with coverage counts, and report the actual alpha vector. Preserve terminal recall. Keep gamma=2 initially so alpha and focal focusing are not changed together.
- Replace only relying on one early-bias exponent with explicit measured exposure buckets (area, tooth pair, terminal presence, new-visible surface). Target hard examples while maintaining the full scan distribution. Test lower midline probability (e.g. 0.1) and delayed introduction separately; neither number nor schedule is established by current evidence.
- Use repeatable training randomness and at least three seeds for finalists. Compare equal optimizer-update budgets; historical local epochs contained 1,200 steps, server epochs 942. Do not select early-vs-late transformers from 100-epoch rankings, because your own ranking reversed by 200.
- Calibrate confidence on a separate held-out subset. Temperature scaling changes confidence, not argmax labels by itself; its benefit comes from better rejection/confirmation decisions. Select thresholds against false-terminal events and missed/late detections, not merely a visually pleasing softmax score.
- Save exact code revision, actual file hashes/manifests, dependencies, weights/sampling settings and optimizer schedule. Validate best/last checkpoints and emit per-class/per-stratum metrics. Resume state should not silently be the best epoch masquerading as last.

## Minimal staged experiment plan (proposal only)

**Stage 0 — common replay, no training:** reevaluate the available early transformer, midline continuation, new scratch model, gate-off dilated, and whole-tooth models on the same frozen cases/patches. Include all other checkpoints as time permits, with their own compatible preprocessing; each must see the same geometric crop and observation color condition. Record model-native synthetic validation separately. Add boundary/terminal error annotations to held-out live sequences. Measure repeated identical-mesh stability. This is needed before claiming any historical model is best for the robot.

**Stage 1 — four runs to isolate two high-priority factors:** same early-transformer/HSV backbone and fixed evaluation; use a 2×2 design: (current inverse alpha vs softened fixed alpha) × (ordinary sampling vs targeted adjacent-pair/terminal sampling). Keep gating and other settings fixed. Fix split/sampler/reproducibility defects equally for all four. Compare at equal full update budget, then repeat the top two with additional seeds. This tests the loss hypothesis and data-exposure hypothesis independently and their interaction.

**Stage 2 — midline sampling:** on the selected stage-1 recipe, compare no dedicated midline, low constant probability (initial test 0.1), and a predeclared warm-start schedule introducing the same low probability. Retain scan-sweep/posterior exposure floors. Only increase to 0.3 if early-patch benefit is demonstrated without unacceptable terminal/pair regression. This isolates timing from simply reusing a pretrained checkpoint whose optimizer/loss history differs.

**Stage 3 — architecture:** compare selected baseline vs +boundary/instance head vs +relative-position attention. Combine them only after separate gains. Evaluate transformer + gating-off separately. Keep a full budget for finalists and do not run a combinatorial grid of every historical knob.

Success should mean fewer false early/terminal identities and fewer merged teeth at acceptable detection delay, while maintaining broad segmentation quality and real-time latency. Higher aggregate mIoU alone is insufficient.

## Complete local model inventory

Best/final values below come from TensorBoard when available; epoch is the actual zero-based epoch scalar, not the CSV row count. A local file may not contain the best logged state. Restarted histories can have repeated epoch entries; historical sequence averages should be treated cautiously outside the clean server matrix and new scratch run.

| Run | Best logged mIoU | Epoch | Final logged mIoU | Best checkpoint present |
|---|---:|---:|---:|---|

| `Patch_Seg_081226` | 56.76% | 90 | 49.89% | `best-epoch=90-val_miou=0.5676.ckpt` |
| `Patch_Seg_17class_gateOff_color_mosaic_tuned_hsv` | 83.96% | 194 | 82.46% | `best-epoch=194-val_miou=0.8396.ckpt` |
| `Patch_Seg_17class_gateOn_colorFlat_tuned` | 83.53% | 194 | 81.04% | `best-epoch=194-val_miou=0.8353.ckpt` |
| `Patch_Seg_17class_gateOn_colorNone_tuned` | 67.13% | 111 | 61.58% | `best-epoch=111-val_miou=0.6713.ckpt` |
| `Patch_Seg_17class_gateOn_color_baseline` | 65.05% | 93 | 57.00% | `best-epoch=93-val_miou=0.6505.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_areaThreshTunedOnly` | 81.68% | 131 | 80.72% | `best-epoch=114-val_miou=0.8014.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_earlyBiasTunedOnly` | 77.77% | 101 | 76.70% | `best-epoch=101-val_miou=0.7777.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned` | 83.17% | 199 | 83.17% | `best-epoch=199-val_miou=0.8317.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv` | 82.92% | 194 | 71.62% | `best-epoch=194-val_miou=0.8292.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_colordropout` | 82.08% | 177 | 79.73% | `best-epoch=177-val_miou=0.8208.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_dilationks` | 83.70% | 189 | 70.81% | `best-epoch=189-val_miou=0.8370.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer` | 84.72% | 181 | 83.57% | `best-epoch=181-val_miou=0.8472.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_100ep` | 79.92% | 88 | 74.43% | `best-epoch=88-val_miou=0.7992.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_lateglobal` | 84.00% | 189 | 81.69% | `best-epoch=189-val_miou=0.8400.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_lateglobal_100ep` | 82.00% | 95 | 79.29% | `best-epoch=95-val_miou=0.8200.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_midlineseed` | 84.78% | 188 | 81.22% | `best-epoch=188-val_miou=0.8478.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_transformer_midlineseed_scratch_200ep_20260919_102500` | 78.29% | 186 | 76.61% | `best-epoch=186-val_miou=0.7829.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_hsv_wholetooth` | 82.10% | 188 | 75.96% | `best-epoch=188-val_miou=0.8210.ckpt` |
| `Patch_Seg_17class_gateOn_color_mosaic_tuned_wholetooth` | 74.71% | 90 | 68.85% | `best-epoch=90-val_miou=0.7471.ckpt` |
| `Patch_Seg_17class_gateOn_color_tuned` | unavailable event history | — | — | `best-epoch=44-val_miou=0.7045.ckpt` |

A separate attempted `wholetooth_dilationks` combination was interrupted during class-frequency estimation before training began; it has no trained checkpoint or performance evidence. It must not be counted as a failed learned combination.

The extra `color_tuned` run is an older 100-epoch-target run whose console log ends with a CUDA launch timeout; its saved best score is 70.45%. It is not a clean modern HSV ablation. Early pre-HSV runs and the original project checkpoint are retained for completeness, not used as controlled comparisons.

## Local evidence pointers

- `dataset/patch_losses.py`: weighted focal loss and 200-patch inverse-frequency estimator.
- `dataset/patch_dataset.py:114`: split selection and missing-cache filtering; `:133` stage sampler; `:173` entropy-seeded training RNG.
- `train_patch_network_color.py:177`: training and validation both receive early_bias_power and the same color transform factory.
- `dataset/patch_dataset_midline.py` and `dataset/patch_dataset_whole_tooth.py`: curriculum selection and fallback.
- `models/patch_transformer_layer.py`: FPS token selection and attention without explicit relative positions.
- `models/patch_dilated_tooth_seg_transformer_network.py`: hard area gates, unconditional early attention.
- `realtime/mesh_viewer_segmented.py:94`: persistent confirmation; `:739` forward/argmax; `realtime/live_preprocessing.py:215`: face capping.
- `Docs/REALTIME.md:845`: historical live premolar/terminal confusion, including large-area failures.
- Companion `training_strategy_audit_20260921.json`: scalar histories, saved checkpoint alpha/epoch metadata and split counts.

## Research grounding for proposed experiments

These papers motivate candidates, not evidence that they will improve this particular scanner:

- [Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html): an alternative to raw inverse-frequency weighting.
- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html): held-out confidence calibration, including temperature scaling.
- [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html): point-cloud attention with explicit spatial structure.
- [Joint Semantic-Instance Segmentation of 3D Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/html/Pham_JSIS3D_Joint_Semantic-Instance_Segmentation_of_3D_Point_Clouds_With_Multi-Task_CVPR_2019_paper.html): joint semantic predictions and instance embeddings.
- [3D Dental Model Segmentation with Geometrical Boundary Preserving](https://arxiv.org/abs/2503.23702): preserving dental boundary detail during downsampling; its complete-scan setting does not itself validate partial-scan robotics.
