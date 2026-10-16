# Testing latent straightness as a predictor of video rollout stability
## The problem

When you train an autoregressive video generator or a world model, one of the first decisions is which latent space to work in: a VAE, a representation encoder like DINOv3, or something else. Each option costs a full training run to evaluate, so it would be very useful to have a cheap metric that can be computed on the frozen latent space and tells in advance how stable the rollouts will be.

For image diffusion this question is close to solved. Frozen-latent metrics such as velocity irreducible variance, iFID or AffineTok's $$M_{\mathrm{SAC}}$$ correlate with generation quality at roughly 0.85–0.96 across 13 to 86 tokenizers. For autoregressive video and world models, however, I could not find a study that validates a frozen predictor of *rollout stability* across many encoders. The closest works change the latent space and show that rollouts change, which is not the same thing.

So the question for this series of experiments was:

- Can a cheap measurement of a frozen latent space predict how well an autoregressive generator trained on it will roll out?
- And does any such measurement say more than the simplest baseline?

All experiments below ran over about a week, roughly 40 GPU-hours, on a single RTX 5070 Ti.

## Setup

- **Data:** Kinetics-400 clips, center-cropped to $$256 \times 256$$. 1300 training clips at 15 fps, 409 test clips. Clips with duplicated (telecined) frames were removed, since repeated frames create fake uneven spacing.
- **Encoders:** DINOv3 ViT-B (with a PCA bottleneck), FLUX.2 VAE, Wan2.2 VAE and DC-AE f32. Every encoder produces $$8 \times 8 = 64$$ tokens per frame.
- **Generator:** the same recipe for every latent space. A 39M-parameter DiT, flow matching with x-prediction, diffusion forcing, block-causal over 16 frames, 6000 training steps. One run takes about 17 minutes.
- **Rollouts:** 8 real context frames, then 48 generated frames.
- **Outcomes:** short-horizon accuracy (error relative to copying the last frame), late drift (Fréchet distance at horizons 32–48), blur (energy ratio) and freezing (motion ratio). Outcome definitions were fixed before looking at any generator result.

## The baseline: slowness

Any candidate metric has to beat *slowness*, which is the error of predicting the next frame as a copy of the current one, normalized by the spread of the frames $$V$$:

$$
\rho_{\mathrm{copy}} = \frac{\mathbb{E}\lVert z_t - z_{t-1} \rVert^2}{V}
$$

It is trivial to compute and it turned out to be very hard to beat. At clip level it predicts freezing with Spearman correlations up to 0.91.

## Part 1. Trajectory straightness

The starting hypothesis was that straighter latent trajectories should be easier to roll out. If consecutive latents lay on a straight line at constant speed, the next frame would be predicted exactly by $$z_{t+1} = 2 z_t - z_{t-1}$$.

To measure it, I used an energy version of straightness, with steps $$v_t = z_t - z_{t-1}$$, accelerations $$a_t = v_{t+1} - v_t$$ and encoder noise floor $$\tau$$:

$$
S = \frac{\mathbb{E}\lVert a_t \rVert^2 - 6\tau}{\mathbb{E}\lVert v_t \rVert^2 - 2\tau}
$$

$$S$$ equals 0 for a straight line, 2 for a random walk and 3 for white noise.

Results:

- **Natural-video latents are not straight.** At 15 fps, $$S$$ is 2.24–2.44 for all four encoders, and consecutive steps turn by 92–101 degrees.
- **Curvature is tied to slowness.** There is an exact identity $$S \cdot \rho_{\mathrm{copy}} = \rho_{\mathrm{line}}$$, where $$\rho_{\mathrm{line}}$$ is the error of straight-line extrapolation. In the data, $$S$$ and $$\rho_{\mathrm{copy}}$$ rise together as the frame rate drops, and comparing encoders at matched displacement changes their order.
- **No predictive power beyond slowness.** Across 409 clips and five generators, the partial correlation of curvature with short-horizon error after removing $$\rho_{\mathrm{copy}}$$ is −0.04 to +0.13. After also controlling for each clip's local motion, the residual effect runs *against* the hypothesis: at equal displacement, straighter clips are relatively harder (−0.20 to −0.28). The likely reason is that low curvature at a fixed step size means directed motion, which is exactly what these generators fail to continue.

The next step was to intervene. I applied invertible per-token maps to the Wan2.2 latent (a rotation as a control, random warps and learned flows that straighten or curve trajectories) and trained the same generator on each variant. Since the maps are invertible, every variant carries the same information.

- **Straightening collapses the space.** Effective dimension drops from about 10 to 1.5–4 on Wan2.2, and from 63 to 1.3–3.9 on FLUX.2.
- **Every reparametrization costs accuracy, in either direction.** Error relative to copy goes from 1.11–1.21 for the controls to 1.34–14.05 for the variants. The native geometry is the most accurate at every checkpoint, on both encoders.
- **Rough maps mostly amplify errors.** The strongest warp reaches 8× the real energy after mapping back, but stays at 1.06 in the generator's own space.

To summarize:

- Curvature at a fixed frame rate mostly re-measures slowness
- Straightening a latent space with invertible maps collapses it and hurts accuracy
- No curvature or spacing metric predicted outcomes beyond $$\rho_{\mathrm{copy}}$$

## Part 2. Why second-order metrics are redundant

After these results I wanted to understand why every metric behaved like slowness. For a stationary latent sequence with temporal power spectrum $$P(\omega)$$, both baselines are averages over the spectrum:

$$
\rho_{\mathrm{copy}} = \big\langle 4 \sin^2(\omega/2) \big\rangle_P, \qquad
\rho_{\mathrm{line}} = \big\langle 16 \sin^4(\omega/2) \big\rangle_P
$$

This means curvature is essentially the spectral centroid of the velocity spectrum. Even the error of the best linear predictor is fixed by $$P$$ (Kolmogorov–Szegő). Every second-order statistic is a different summary of the same spectrum. The derivation explains why these metrics correlate; that they are redundant across real encoders is the empirical finding from Part 1.

Because of this, I shelved a planned pre-registered stage 3 (at least 36 latent spaces with 3 seeds, about 15 GPU-days). It would have tested more summaries of the same spectrum.

## Part 3. Reproducing tFVD on the V-RAE benchmark

The V-RAE paper reports that tFVD (decode the midpoint between neighbouring latents and compute FVD) predicts generation quality across 10 video autoencoders. I re-ran it on 9 of them (5 VAEs and 4 V-RAEs) on UCF101, against the paper's published gFVD.

| | Correlation with gFVD |
|---|---|
| tFVD, pooled (Pearson) | 0.63 (paper: 0.621) |
| Family label alone (Spearman) | 0.87 |
| tFVD within VAEs (Spearman) | −0.30 |
| tFVD within V-RAEs (Spearman) | −0.20 |

The size of the correlation reproduces, but in my analysis it comes from the split between the two families, not from ranking autoencoders within a family. It is also not slowness, since $$\rho_{\mathrm{copy}}$$ has the opposite sign.

One lead came out of this study. Adding isotropic noise of the same size and decoding (iso-tFVD) gives Spearman 0.95 pooled and 0.80 within both families, which points to **decoder tolerance to latent noise**. It is only a lead, though: within the VAEs it ranks almost like PSNR, and the V-RAE decoder was trained with latent noise.

## Part 4. Properties beyond the spectrum

If second-order statistics are exhausted, the next candidates are properties the spectrum cannot capture. I tested two of them on all 26 existing generators, with the tests pre-registered:

- **Temporal neighbourhood preservation is invalid as defined.** Its shuffle check fails even after the one allowed fix. Frames of one clip share content, so no metric based on frame content can pass that check.
- **Decoded oracle error is a good measuring tool, not a predictor.** It measures how a linear-Gaussian predictor's error looks after decoding. Within the Wan2.2 family of 13 spaces it ranks the variants well (Spearman 0.87 with accuracy), and a training-free decoded line extrapolation does even better (0.92). It also tracks the damage along the warp ladder exactly. However, on FLUX.2 it does not replicate (−0.03 and 0.43, neither significant).

## Part 5. Where freezing actually comes from

All generators in this work froze: motion was mostly 9–32% of real. To check whether the latent space causes it, I built synthetic latents with one fixed AR(2) spectrum and three innovation shapes (Gaussian, bimodal and Laplace), hidden behind a random rotation, and trained the same recipe on each with 3 seeds per arm. The hypothesis was that multimodal conditionals cause freezing and blur.

- **Freezing appears with exactly Gaussian conditionals.** The generator regresses to the mean (one-step variance 0.77 of the truth), freezes (motion 0.60) and blurs (energy 0.16).
- **Bimodal innovations froze less, not more** (log motion −0.30 against −0.51 for Gaussian). This is the opposite of the hypothesis.
- **The generator does not reproduce the innovation shape.** One-step kurtosis is 3.00 in every arm, while the true values are 3.00, 1.23 and 5.98.

An additional sampler check (one seed per arm) separated the failure into two parts:

- **Sampler under-dispersion.** With 48 sampling steps instead of 12, one-step variance reaches 0.96–0.99 of the truth in every arm.
- **Collapse under self-conditioning.** From horizon 16 on, the context window holds only generated frames, and energy collapses even at 48 steps: 0.92 at horizon 16, 0.32 at 32 and 0.13 at 48 for Gaussian innovations.

**Important note:** this means that a large part of what the earlier experiments measured as an outcome is a property of the training recipe, not of the latent space.

## Re-testing the one positive result

Part 1 had one result that looked like a benefit of straightening. At 15k training steps, the straightened space kept 0.60 of real motion against 0.09 for the native one, with a single seed. I re-ran both with two more seeds and a pre-registered rule: the Welch 95% CI of the difference must exclude 0.

- Straightened: 0.60, 0.33, 0.21. Native: 0.09, 0.11, 0.09.
- Difference: +0.28, 95% CI [−0.21, +0.77]. **Inconclusive.**
- The direction is consistent (every straightened seed moves more than every native seed), but the original seed was the best of the three, and straightening still costs accuracy.

## Results

Positive:

- A reproducible pipeline. An independent re-implementation matched the clip-level partial correlations exactly, and the published tFVD correlation reproduced.
- Two measuring tools that track error amplification through the decoder: decoded oracle error and decoded line extrapolation.
- A located recipe fault: sampler under-dispersion and collapse under self-conditioning.
- One open lead: decoder tolerance to latent noise.

Negative (results that prune the search):

- Curvature, spacing and other second-order statistics add nothing beyond $$\rho_{\mathrm{copy}}$$.
- Straightening a latent space costs accuracy and collapses it; its freezing benefit is not reliable across seeds.
- Neighbourhood preservation as defined is invalid.
- Multimodal conditionals are not what causes freezing in this setup.
- Decoded errors do not transfer across encoder families.

No frozen-latent predictor beat slowness in these experiments, so the original question remains open.

## Limitations

- **Weak outcomes.** Every generator is a small 39M DiT trained for 6000 steps on 1300 clips. They all freeze, so one failure mode dominates the outcomes.
- **Few encoders.** Four real encoders in Part 1 and nine in two families in Part 3. Most of the intervention variance comes from invertible maps of two encoders.
- **Mostly one seed per run**, except the synthetic study and the re-test above.
- **No actions.** The action-conditioned world-model case has not been tested yet.

## Conclusion

The main takeaway is that slowness is a much stronger baseline than it looks, and second-order geometry of a frozen latent space does not add information on top of it. The most actionable finding, however, is on the outcome side: the recipe used for all tests has a sampler fault and a self-conditioning collapse, and until these are fixed, further predictor tests do not mean much.

Next steps:

- Fix the recipe first: more sampler steps, plus a training-side fix for self-conditioning (context-noise augmentation, longer windows or training on the model's own rollouts). Test on the synthetic latents first, where a run takes about 18 minutes, then on Wan2.2 and FLUX.2.
- Run a causal test of decoder tolerance: fine-tune one decoder with and without latent-noise augmentation and check whether rollout quality moves with it.
- Once the recipe passes, scale up to 15–20 real encoders with 2 seeds, scored in decoded space.

Thanks for reading!
