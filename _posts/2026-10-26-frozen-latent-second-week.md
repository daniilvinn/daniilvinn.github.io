# Fixing the recipe first: what starts freezing, and why the latent question is still open
## Where the last post ended

The [previous post](/2026/10/16/frozen-latent-first-week-studies.html) looked for a cheap measurement of a frozen latent space that predicts how well an autoregressive video generator trained on it will roll out. No metric beat slowness $$\rho_{\mathrm{copy}}$$. The most actionable finding was on the outcome side: every generator froze, and part of the freezing came from the training recipe (sampler under-dispersion and collapse under self-conditioning), not from the latent space.

This week's questions follow from that:

- What exactly starts the freeze?
- If the recipe is repaired with a fix that does not depend on the latent space, do different latents still behave differently?

The work had two parts: an overnight debate run that pruned the hypothesis list using only existing data, and a GPU follow-up of about 8.5 GPU-hours on the same RTX 5070 Ti.

## Setup

Everything is the same as last week unless stated otherwise.

- **Data, encoders and generator:** Kinetics-400 at 15 fps; Wan2.2 VAE, FLUX.2 VAE, DINOv3 ViT-B (with a PCA bottleneck) and DC-AE; a 39M-parameter DiT with flow matching, x-prediction and diffusion forcing over 16 frames, trained for 6000 steps.
- **Rollouts:** 8 real context frames, then 112 generated frames, on the 330 test clips that are long enough (a "light" set of 160 clips for the repair pilot). Scored in two windows: horizons 25–48 and 89–112.
- **Motion at lag $$k$$:** $$M_k$$ is the generated displacement power between frames $$k$$ apart, divided by the real one. $$M_1$$ is last week's motion ratio. About 1 means real-like motion, near 0 means frozen.
- **Coherence:** $$C = M_8 / M_1$$. Real-like motion gives about 1. Flicker, where frames change a lot but the changes do not add up over time, gives values below 0.7.
- **Energy** $$E$$: generated latent power over real latent power. Values well below 1 mean blur or collapse.
- **Pre-registration:** the plan, with predictions and kill criteria, was committed before any run that produced an outcome. Every deviation was logged before the affected result was seen. The rollout code reproduces last week's stored numbers to 4 decimals.

## Part 1. An overnight debate run on existing data

Last week's list of next steps was long, and most items cost 5–12 GPU-hours each. Before spending GPU time, I ran an unattended overnight session of LLM agents (about 10.5 hours). Literature researchers mapped the field and proposed 26 hypotheses. 15 of them went through structured debates between an attacker, a defender and a judge, for up to 5 rounds, and a fresh attacker then re-read every survivor. The agents could read the stored evaluation files, training logs and code, but ran no experiments.

The verdicts: none survived as stated, 11 survived in a narrower form, 3 were unresolved and 1 was killed. The killed one was the idea that training on a temporally whitened "innovation" latent would remove freezing. For an ideal denoiser, that latent is just the native one on a shifted noise schedule, and no rescaling of the latent gets the 12-step sampler past a one-step variance ratio of about 0.81.

The most useful output was not the verdicts but facts that the debaters re-derived from data already on disk. I checked them against the files:

- **Correction to last week: slowness does not order the encoders.** Across the four base encoders, the Spearman correlation of $$\rho_{\mathrm{copy}}$$ with late freezing is 0.0. Wan2.2 and DC-AE are about equally slow, yet they freeze the least and the most. The clip-level correlation of 0.91 reported last week has a negative sign (faster clips freeze more) and comes mostly from the motion ratio's own denominator. Controlling for each clip's real motion leaves partial correlations of −0.22 to +0.004.
- **About 60% of the spread between encoders at 6000 steps is training stage.** Generators start over-dispersed, calibrate, then freeze, and encoders move along that path at different speeds (DINOv3 is the slowest). Matching runs on stage cuts the range between encoders from 1.06 to 0.40 log.
- **Freezing deepens as the fit improves.** On Wan2.2, late motion follows one power law in the training loss $$L$$ across two learning-rate schedules, $$m \approx 0.295\,(L/0.591)^{5.8}$$, within 9% at all 12 checkpoints.
- **Real generators and synthetic latents fail differently.** Real generators freeze by over-persistence: read as AR(1), their persistence is 0.98 and 0.95 against 0.87 and 0.79 in the data, while energy stays at 0.76–0.86. Last week's synthetic latents instead collapse in energy (0.15). These are two separate failures.
- **Most motion is lost early, while real frames are still in the context.** Wan2.2 keeps 0.89, 0.64, 0.50 and 0.41 of real motion at horizons 1–4, and 0.21 at horizon 8.

Together, these made the next step clear: before comparing latents again, find what starts the freeze and whether a latent-blind repair removes it.

## Part 2. Short-horizon error is mostly step size

Last week's short-horizon accuracy O1 (error at horizons 1–4 relative to copying the last frame) splits exactly into two parts:

$$
\mathrm{O1} = \underbrace{(1 - c^2)}_{\text{skill}} + \underbrace{(\sqrt{M} - c)^2}_{\text{step size}}
$$

Here $$c$$ is the cosine between the generated and the true displacement, and $$M$$ is the displacement power ratio. Skill says whether the model moves in the right direction; step size says whether it moves too much or too little.

I ran 4-frame rollouts at 2000 and 6000 training steps for 21 Wan2.2 and FLUX.2 family runs, and for DC-AE at 1000–6000 steps:

- **Directional skill improves from 2000 to 6000 steps in 22 of 23 runs.**
- **The change in O1 is almost all step size:** a median of 93% in the Wan2.2 family and 99% in FLUX.2.
- **DC-AE's late "worsening" is step size too.** Its O1 rises from 1.36 to 1.81 while its directional skill keeps improving.

So O1 on its own mostly measures step-size calibration. From now on I report it only with its split.

## Part 3. What starts the freeze

Between horizons 1 and 2, motion drops by about 0.3 log: $$\Delta_{12} = \log m_1 - \log m_2$$. There were four candidate causes:

- **Junction:** a scoring artefact. The horizon-1 step starts from the last real frame, which could inflate $$m_1$$.
- **Window:** the position of the frame inside the 16-frame window.
- **Activity:** the model sets its next step from how much its recent context moves.
- **Exposure:** the model steps less after a frame it generated itself than after a real frame with the same amount of change.

To separate them, I took 7 real frames, regenerated the next frame from noise level 0.2, and measured the spread of the following step over 4 samples while swapping what sits in the context. The last context frame was either the model's own sample or a real frame with matched increment power. This used 409 clips, 12 sampler steps and 95% clip-bootstrap CIs. Two controls were fixed in advance: synthetic Gaussian latents, where none of the terms should appear, and an early FLUX.2 checkpoint, where exposure should already be visible.

| Run | $$\Delta_{12}$$ | Junction | Window | Activity | Exposure |
|---|---|---|---|---|---|
| Wan2.2, 6k steps | 0.338 | 0.031 | 0.002 | 0.058 | 0.159 [0.135, 0.184] |
| FLUX.2, 6k steps | 0.363 | 0.032 | −0.004 | 0.040 | 0.138 [0.126, 0.149] |
| Wan2.2, 15k steps | 0.297 | 0.019 | 0.004 | 0.038 | 0.153 [0.105, 0.205] |
| FLUX.2, 1500 steps (positive control) | 0.188 | 0.014 | −0.005 | −0.010 | 0.078 [0.062, 0.094] |
| Synthetic latents (negative control) | 0.091 | 0.015 | 0.006 | 0.007 | −0.000 [−0.002, 0.001] |

Both controls passed. The results:

- **Exposure is the largest term in every run.** After its own frame, the model's next step is 0.14–0.16 log smaller than after a real frame with the same amount of change.
- **It is not a scoring artefact.** The junction term is 0.02–0.03 log, under 10% of the drop. The window position does nothing.
- **The step follows context activity.** Across real contexts at 30, 15 and 7.5 fps, the elasticity of the next-step spread with respect to context motion is 0.53–0.71. A context whose last 4 frames are frozen cuts the spread to 0.22–0.46 of normal.
- **Energy is not the driver.** Lowering the context's energy to about half barely moves the step (elasticity −0.08 to 0.17).
- **Activity sustains the freeze but does not start it.** A gain map fitted on real contexts and run freely from horizon 1 reproduces only 5–23% of the decay from horizon 1 to 8.
- **At 48 sampler steps, exposure halves** (0.077 of a 0.135 drop on Wan2.2). It is still the largest term, but part of it comes from the 12-step sampler's own samples.
- **The activity response is learned during training.** Its elasticity is 0.12 in FLUX.2 at 1500 steps and 0.53 at 6000. This is an observation, not a test, but it is one candidate explanation for why freezing deepens as the fit improves.

In short: exposure starts the freeze, and the learned response to context activity compounds it. That points to training on the model's own outputs as the right kind of fix, rather than an inference-time trick.

## Part 4. More sampler steps do not equalise latents

Last week, 48 sampler steps fixed one-step under-dispersion on synthetic latents. On real latents, it does not fix freezing. Values are seed means ($$M_1$$ / $$M_8$$ / $$C$$ / $$E$$ at horizons 25–48), plus late $$M_1$$; DINOv3 has one seed.

| Encoder | 12 steps | 48 steps | Late $$M_1$$ (12 / 48 steps) |
|---|---|---|---|
| Wan2.2 | 0.11 / 0.24 / 2.1 / 0.79 | 0.12 / 0.32 / 2.9 / 0.76 | 0.08 / 0.09 |
| FLUX.2 | 0.18 / 0.30 / 1.7 / 0.79 | 0.14 / 0.23 / 1.8 / 0.52 | 0.12 / 0.09 |
| DINOv3 | 0.32 / 0.34 / 1.1 / 0.86 | 0.42 / 0.60 / 1.4 / 0.79 | 0.29 / 0.39 |

- **The gap between encoders grows.** The log range of $$M_1$$ goes from 1.02 to 1.22 at horizons 25–48, and from 1.22 to 1.51 late. DINOv3 keeps about three times the late motion of the two VAEs at both step counts.
- **48 steps makes FLUX.2's energy collapse worse** (energy 0.52 against 0.79).
- **What a frozen rollout looks like.** At horizons 25–48, $$C$$ is 1.1–3.6: motion is slow but drifts persistently. By horizons 89–112 the rollout is nearly static, with small per-frame jitter.
- **The coherence measure works.** The 500-step checkpoints, which move too much ($$M_1$$ of 1.9–9.1), all have $$C \le 0.45$$, while a perfect sampler on synthetic latents gives $$C = 1.000$$.

## Part 5. Repairing the recipe

The repair is borrowed from [Resampling Forcing](https://arxiv.org/abs/2512.15702). Each 6000-step generator is fine-tuned for 1000 steps with its context frames replaced by its own re-generations of the real frames: a real frame is noised and then denoised by the model, while the targets stay ground truth. A sham fine-tune with the same steps and no corruption is the baseline. Rollouts use 48 sampler steps. The same rule applies to every latent and was fixed before running.

A pre-check confirmed that this degraded context contains the slowing cue. It cuts the next step by 0.74 log, more than the model's own frames do (0.32).

The pre-registered gate for "repaired" was $$M_1$$ and $$M_8$$ in [0.5, 1.5], $$C \ge 0.7$$ and $$E$$ in [0.8, 1.2]. Only if the pilot passed would the main arm run: 10 new base runs, 15 repairs and 15 shams, testing whether the latents become equivalent within ±0.4 log. Late motion was still changing after 1000 fine-tune steps, so the pilot was extended to 2000 steps, as pre-registered. The table shows $$M_1$$ at both windows, late $$M_8$$ and $$C$$, and the gain over the same-seed sham at horizons 25–48.

| Run | $$M_1$$ after 1000 | $$M_1$$ after 2000 | Late $$M_8$$ / $$C$$ (2000) | Gain over sham |
|---|---|---|---|---|
| Wan2.2, seed 0 | 0.32 / 0.16 | 0.77 / 0.20 | 0.19 / 0.97 | +1.56 log |
| Wan2.2, seed 0, other fine-tune seed | 0.37 / 0.15 | 0.86 / 0.21 | 0.19 / 0.93 | |
| Wan2.2, seed 1 | 0.91 / 0.35 | 1.83 / 1.37 | 0.82 / 0.59 | +2.94 log |
| DINOv3, seed 0 | 0.60 / 0.54 | 0.58 / 0.59 | 0.35 / 0.60 | +0.42 log |
| FLUX.2, seed 0 | 0.22 / 0.09 | 0.27 / 0.10 | 0.07 / 0.68 | +0.98 log |
| FLUX.2, seed 1 | 0.26 / 0.16 | 0.30 / 0.20 | 0.14 / 0.70 | +0.57 log |
| Wan2.2, seed 0, from step 3000 | 0.27 / 0.19 | 0.36 / 0.21 | 0.15 / 0.75 | |
| DINOv3, seed 0, from step 3000 | 0.78 / 0.81 | 0.79 / 0.82 | 0.36 / 0.44 | |

- **The repair has a strong effect.** Late motion rises by 0.4–2.9 log over sham.
- **No run passes the gate, each for a different reason.**
  - Wan2.2 seed 0 is healthy and coherent at horizons 25–48 but freezes again by 89–112. The repair delays the freeze; it does not remove it.
  - Wan2.2 seed 1 overshoots into jitter ($$M_1 = 1.83$$) and flickers late ($$C = 0.59$$).
  - DINOv3 keeps frame-to-frame motion but loses coarse motion late ($$C = 0.60$$).
  - FLUX.2 improves over sham but stays frozen.
- **The starting checkpoint matters.** Wan2.2 seed 0 repaired from step 3000 ends at 0.36, against 0.77 from step 6000.
- **The spread comes from the base model, not the fine-tune.** Re-running the Wan2.2 seed 0 repair with a different fine-tune seed reproduces it within 0.09. Two base models trained identically except for the seed respond to the same repair with a 2.4-fold difference in motion at horizons 25–48 and about 7-fold late.

The pre-registered verdict is "repair ineffective". The main arm did not run, so the question of whether latents become equivalent after a repair stays open.

## Part 6. Two more attempts

**Decode and re-encode.** Generated FLUX.2 frames sit off the codec's manifold: a decode–encode round trip moves them by 0.50 of a real step, against 0.07 for real frames. In a one-step test, putting the re-encoded frame in the context recovered about 90% of the exposure gap. In a closed loop, where every generated frame is round-tripped before it enters the context, this does not hold up (FLUX.2, both seeds):

- Energy falls (−0.19 and −0.20 log).
- The lag-1 motion gain (+0.33 and +0.49 log) is matched by a perturbation of the same size with random signs, and neither adds lag-8 motion.

Refuted. Wan2.2's generated frames are already on its codec manifold, so the round trip does nothing there.

**Free-running repair (exploratory).** This is the same fine-tune, but the degraded context comes from a free-running 12-step rollout of 1–8 frames after a ground-truth prefix.

| Run | $$M_1$$ (25–48 / 89–112) | Late $$C$$ | Late $$E$$ |
|---|---|---|---|
| Wan2.2, seed 0 | 2.46 / 2.22 | 0.62 | 0.92 |
| Wan2.2, seed 1 | 3.14 / 3.15 | 0.54 | 0.87 |
| DINOv3, seed 0 | 0.91 / 1.07 | 0.39 | 0.84 |

- **The late freeze is gone.** Late motion equals early motion in every run, and energy holds at 0.84–0.92.
- **But every run flickers** ($$C$$ of 0.39–0.63), and Wan2.2 overshoots real motion two to three times.

The two kinds of degradation bracket the target: anchored resampling under-corrects and the freeze comes back, while free-running rollouts over-correct and produce flicker.

## Results

Positive:

- The start of the freeze is located. Exposure starts it, and the model's learned response to context activity compounds it. It is not energy and not a scoring artefact.
- Training on the model's own outputs moves late motion a lot (+0.4 to +2.9 log over sham), so it is the right family of fix.
- Two measurement fixes: O1 is reported as skill plus step size, and coherence $$C$$ separates real motion from flicker.
- The debate run corrected last week's slowness result and found a fit–freezing law in existing data, without any GPU time.

Negative (results that prune the search):

- No repair reached a stable, coherent endpoint, so latent equivalence after a repair is still untested.
- 48 sampler steps neither fix freezing nor equalise latents, and they worsen FLUX.2's energy collapse.
- Decoding and re-encoding generated frames does not remove exposure.
- Base models that differ only in seed respond to the same repair up to about 7-fold differently.

## Limitations

- **Same small recipe.** Everything is still a 39M DiT trained on Kinetics-400 clips.
- **Few seeds.** One or two base seeds per encoder (DINOv3 has one), and the repair pilot used 160 clips.
- **A provisional mechanism.** "Activity compounds exposure" is not final: the direct test of the activity lever at fixed energy was not run.
- **The debate run is argument, not experiment.** Its literature search ran out of budget partway through, and the judges' calibration was not checked. The facts quoted in Part 1 were re-checked against the stored files.
- **Descriptive latent differences.** DINOv3 keeps more late motion before repair, and FLUX.2 responds least to the repair, but with 1–2 seeds and no calibrated endpoint neither is a test.

## Conclusion

The main question, whether the choice of latent still matters once freezing is repaired, is still open, and the reason is itself the main result. At 39M, a latent-blind repair either under-corrects (the freeze returns) or over-corrects (flicker), and its outcome depends more on the base seed than on anything else. Until a repair passes the gate reproducibly, comparing latents on this recipe mostly measures seed luck and repair strength.

The mechanism, however, is clearer than a week ago: the model takes smaller steps after its own frames than after real ones, and its learned response to slow context turns that into a freeze.

Next steps:

- Try a fixed ladder of degradation strengths between anchored resampling and free-running rollouts on one encoder with 4–5 base seeds, with the rule chosen in advance and scored by the coherence gate at horizons 89–112. Run the latent comparison only if the post-repair spread between seeds falls below about 0.1. Roughly 4–6 GPU-hours.
- If no strength works, stop comparing latents at this scale and lean on larger published studies such as [CADRE](https://arxiv.org/abs/2610.30129).
- Two cheap mechanism checks: the activity lever at fixed energy, and whether the activity response tracks the fit–freezing law across stored checkpoints (about 0.5 GPU-hours).

Thanks for reading!
