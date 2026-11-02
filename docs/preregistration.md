# Pre-registration of the benchmark

The engine was designed and tuned on the development half of the data. Before it was run on the test
half, the protocol below was written down and committed: the engine, the designs it is compared with, the
measures, the success criteria and my own numerical predictions. The test half was then used once. This
page is an English translation of that protocol, unchanged, followed by the results as they came out,
including the predictions that turned out wrong.

## Timeline

| When (2026-10-29, CET) | What |
|---|---|
| until 21:45 | Engine designed on the development half only (exploratory scripts, no criterion judged). A dry run of the benchmark code, 5 replicates, also on the development half. |
| 21:45:56 | Protocol committed (private notes repository, commit `cb769d0`) |
| 21:46:00 | Engine and benchmark code committed (private code repository, commit `b4b75e1`) |
| 22:03:34 | Benchmark run once on the test half (2,000 replicates); results file written |
| after | Results appended to the protocol; the protocol itself was not edited |

The two commits live in private repositories (they also hold unrelated work in progress); they can be shown
on request. This public repository is a cleaned re-implementation of the same engine, published
afterwards (see the last section).

## The engine, frozen

1. **Waves**: a campaign of n observations runs in 5 waves of n/5.
2. **Design of each wave**: strata by quintiles of the current error model f_t; Neyman allocation,
   n_h ∝ N_h · sqrt(mean of f_t(1 − f_t) in stratum h), at least 2 observations per stratum; uniform
   sampling with replacement within each stratum.
3. **Estimation**: in each wave, a stratified difference estimator,
   θ̂_t = Σ_h W_h [mean of f_t in h + mean of (y − f_t) over the sample in h]; final estimate
   θ̂ = Σ_t (b_t / n) θ̂_t, variance Σ_t (b_t / n)² Σ_h W_h² s_h² / n_h, 95% interval θ̂ ± 1.96 √V̂.
   Each wave is a design fixed before its labels, so its estimate is unbiased given the past, and so is
   the final estimate.
4. **Learning after each wave**: maximum a posteriori logistic model on all labels collected so far.
   * *With a transferred model* f_0: logit f_t = a + b · logit f_0 + γ_class, priors a ~ N(0, 0.5²),
     b ~ N(1, 0.25²), γ ~ N(0, 0.25²).
   * *Cold start*: logistic model on the standardised "enriched" features, coefficients ~ N(0, 0.5²),
     intercept ~ N(logit 0.3, 1.5²); the first wave is uniform.

## Split and populations

Each 20 km spatial block goes entirely to one half, by hashing its identifier. **Test half**: Occitanie
1,019 points, Brittany 1,322, Grand Est 2,036, Massif Central 1,187, i.e. 5,564 points; **sixteen
cases** (four study areas × WorldCover, Dynamic World, OSO, CLC+). Target: the error rate of the product on
the test half of the area. Sampling with replacement (the pool stands for a territory where N ≫ n).

## Prior error models

| Scenario | Model f_0 | Trained on |
|---|---|---|
| **Transfer** (same product, other areas) | L2 logistic regression on the enriched features (six indicators, class stated by the product, agreement with each other product and their classes, OSO validity) **plus the fifth opinion** | All points of the three other areas |
| **Transfer without embeddings** (comparison) | Same, enriched features only | Same |
| **New product** | One logistic model on features that name no product (stated class, number of agreeing products, majority class of the others, six indicators, OSO validity, opinion) | Errors of the **three other products** in the **three other areas** |
| **Cold start** | None: the engine learns everything from its own observations | — |
| *Previous design* | Logistic regression on the six indicators, one-shot importance sampling | Three other areas |

**The fifth opinion**: a logistic classifier (L2, C = 0.5) predicts the true LUCAS class from the
AlphaEarth 2021 annual satellite embedding (64 dimensions); the features are its probability for the class
stated by each product, its maximum confidence and its entropy. In the training areas, the opinion is
predicted out of area. All these features exist at every pixel, without a field visit.

## Designs compared

| Code | Name in this repository | Sampling | Estimator | Model |
|---|---|---|---|---|
| D0 | `srs` | Simple random | Mean | none (the reference) |
| D1 | `class` | Strata by map class, proportional allocation | Stratified mean | none (good practice, Olofsson et al. 2014) |
| D4-six | `one_shot_indicators` | Importance sampling ∝ sqrt(f_0(1 − f_0)), 10% uniform | Weighted difference | six indicators: **the previous design** |
| D2 | `srs+diff` | Simple random | Difference | same f_0 as the engine |
| D4 | `importance` | Importance sampling, 10% uniform | Weighted difference | same |
| D5 | `neyman` | Quintiles of f_0, Neyman | Stratified mean | same |
| D5d | `neyman+diff` | Quintiles of f_0, Neyman | Stratified difference | same |
| D6 | `class+diff` | Map classes, proportional | Stratified difference | same |
| **Engine** | `engine` | Five stratified waves, learning between waves | Stratified difference per wave | f_0 of the scenario, recalibrated |

In cold start only D0, D1 and the engine exist: no model is available for the guided fixed designs.

## Measures

Sizes n ∈ {100, 200, 500}; **criteria judged at n = 200**. Saving at equal precision
S = 1 − MSE_design / MSE_D0, with the **exact** MSE for fixed designs and, for the engine, the mean over
replicates of the exact conditional variances of its waves. Empirical coverage of the 95% interval and
bias over 2,000 simulated campaigns per case, scenario, design and size.

## Criteria and predictions, as written before the run

| ID | Criterion, at n = 200 | My prediction |
|---|---|---|
| **XB-001** transfer, C1 | Median saving of the engine ≥ **28%** | Met: 31% to 35% |
| XB-001, C2 | Engine better than the previous design (D4-six) in ≥ **14** cases out of 16 | Met: 16 out of 16 |
| XB-001, C3 | Engine within **1 point** of the best fixed design using the same f_0 (max of D2, D4, D5, D5d, D6), or better, in ≥ **12** cases out of 16 | Met; but the engine will not clearly beat any well-designed fixed design (expected gap 0 to 1 point) |
| XB-001, C4 | Coverage of the engine between **93% and 97%** in all 16 cases | Met |
| XB-001, C5 | Absolute bias of the engine < **0.3 point** in all 16 cases | Met |
| **XB-002** new product, C1 | Median saving ≥ **25%** | Met: 29% to 33% |
| XB-002, C2 | Coverage between 93% and 97% in all 16 cases | Met |
| **XB-003** cold start, C1 | Median saving ≥ **12%** | Met: 14% to 19% |
| XB-003, C2 | Engine better than D1 in ≥ **14** cases out of 16 | Met: 15 or 16 |
| XB-003, C3 | Coverage between 93% and 97% in all 16 cases | Met, narrowly in one or two cases |
| **XB-004** ambiguity-driven observer noise, C1 | With an observer confusing 20% of labels on average, concentrated on ambiguous units: bias of the engine equal to that of D0 within **0.3 point** in all 16 cases | Met: every design-unbiased estimator targets the same noisy mean; the guided design adds no bias |
| XB-004, C2 | *Descriptive*: saving of the engine under this noise | It drops by a few points |
| **XB-005** one campaign for the four products | *Descriptive*: labels needed for ±2 points on all four products of an area, shared engine vs shared random sample vs four separate engines | The shared engine saves 20% to 30% over the shared random sample; four separate campaigns cost about three times more |

Declared in advance: the development half shaped the choices (feature set, logistic rather than boosting,
fifth opinion, stratified version of the engine, five waves, narrow prior), and it showed that local labels
do not rank errors better than the transferred model, so with a good transferred model the engine can only
gain by recalibration. The prior models use all points of the other areas, both halves; only the test half
of the target area is untouched. A failed criterion is reported as such.

## Results of the pre-registered run

Twelve criteria out of twelve met; three numerical predictions wrong.

| ID | Result (n = 200) | Criterion | Prediction |
|---|---|---|---|
| XB-001 C1 | **32.8%** | met | right |
| XB-001 C2 | **16 out of 16** | met | right |
| XB-001 C3 | 16 out of 16; median gap −0.10 point (−0.83 to +0.47), strictly better in 6 cases | met | right |
| XB-001 C4 | 93.6% to 95.5% | met | right |
| XB-001 C5 | at most 0.10 point | met | right |
| XB-002 C1 | **32.1%** | met | right |
| XB-002 C2 | 93.5% to 95.4% | met | right |
| XB-003 C1 | **17.6%** (20.2% at n = 500) | met | right |
| XB-003 C2 | **14 out of 16** | met, narrowly | **wrong** (15 or 16 expected) |
| XB-003 C3 | 93.2% to 95.0% | met | right |
| XB-004 C1 | largest gap 0.15 point; bias of +10.4 to +15.2 points for every design | met | right |
| XB-004 C2 | saving **18.3%** (vs 32.8% without noise) | descriptive | **wrong** ("a few points less") |
| XB-005 | **931 to 1,534** labels vs 1,936 to 2,315 (shared random, 34% to 52% fewer) and 3,617 to 5,878 (four separate engines, ×3.8 to ×3.9); coverage ≥ 94.2% | descriptive | **wrong** (20% to 30% and ×3 expected) |

Reading: the engine keeps its promises of validity and saving in all three scenarios; with a good
transferred model a fixed stratified design using the same model does as well (the gain comes from the
model); the loop is what produces the gain in cold start. The XB-004 result matters most for citizen
science: observer noise biases every design alike, and halves the saving, so label quality comes before
label quantity.

## Relation to this repository

This repository re-implements the pre-registered engine and benchmark in a smaller, documented package.
Differences with the pre-registered run:

* **1,000 replicates instead of 2,000**, and different random seeds. Re-running the pre-registered
  criteria on `reports/results/benchmark.csv` gives the same verdicts: XB-001 C1 32.8%, C2 16/16, C3 16/16
  (median gap −0.11 point, strictly better in 5 cases), C4 93.0% to 95.2%, C5 at most 0.16 point;
  XB-002 32.0%, coverage 93.2% to 95.3%; XB-003 17.5% (20.1% at n = 500), 14/16 better than D1, coverage
  93.7% to 95.8%.
* **XB-004 (observer noise) and XB-005 (one campaign for four products) are not included here.**
* **The two-phase baseline in cold start was added on 2026-11-02, after the pre-registered run.** It was
  not pre-registered and should be read as an exploratory comparison: it shows that most of the cold-start
  gain comes from the first model update (15.0% for two phases vs 17.5% for five waves).
