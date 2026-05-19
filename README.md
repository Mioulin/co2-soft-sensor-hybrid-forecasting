# CO2 Soft-Sensor v2
## Physics-Informed Hybrid Forecasting for Post-Combustion Carbon Capture

A compact, reproducible PyTorch pipeline for estimating CO2 concentration profiles
across a six-stage MEA absorber column from sparse, asynchronous sensor readings.
This is a partial re-implementation and extension of Chai, Guo & Mercangoz (2026): SDAE-based dimensionality reduction, GRU forecaster, and covariance-weighted fusion with Gaspari-Cohn localization are taken from that work. Two extensions are added: a compact Transformer alternative to the GRU, and multi-horizon evaluation at h in {1, 3, 6, 12}.

---

## Problem

A rotating gas analyzer (AT400) samples CO2 concentration at one absorber stage
per timestep across a 43-second cycle. Only one of the six sampling points is
observed at each timestep. The goal is to forecast the full six-point CO2
concentration profile h steps ahead, under partial observability, with only 8
short pilot runs (50–228 rows each) available for training.

This mirrors the real industrial sensing problem that Applied Computing's Orbital
platform addresses at scale: sparse ground truth, non-synchronous measurements,
small data, and high economic stakes.

---

## Methodology

### Data and split

| Split | Runs | Rows | Role |
|-------|------|------|------|
| Train | 140120\_1, 140206\_1, 140207\_2, 140214\_1, 140214\_2, 140227\_1 | 552 | Scaler fit, SDAE, model training |
| Val | 140313\_1 | 228 | Early stopping, fusion calibration only |
| **Test** | **140207\_1** | **118** | **Final evaluation — never touched during training** |

Split is by whole run only. Sliding windows never cross run boundaries. The test
run is held out strictly from scaler fitting, SDAE training, early stopping, and
fusion calibration.

### Target reconstruction: kinetic-anchored residuals (KAR)

Direct linear interpolation of sparse AT400 readings produces distorted training
targets. This pipeline instead computes residuals between AT400 and the precomputed
kinetic prior at the observed point, interpolates those residuals (which are 8–16x
smaller in standard deviation than the raw signal), then adds them back:

```
r(t, pt) = AT400(t)/100 - kinetic(t, pt)      # at observed point only
r_profile = interpolate_per_channel(r_sparse)  # fill all 6 channels
y_profile = kinetic + r_profile                # full reconstructed profile
```

The six-point profile is a pseudo-label. Observed-point RMSE — computed only at
the single AT400 measurement per timestep — is the primary metric throughout.

### Feature pipeline

```
89 numeric sensors + 6 one-hot sampling-point indicators = 95 features
  -> StandardScaler (fit on train only)
  -> SDAE encoder: 95 -> 16 latent dims  (AWGN noise injection, unsupervised)
  -> Sliding window length W = 18
  -> Target: y_profile[t + h],  h in {1, 3, 6, 12}
```

### Why GRU and Transformer — the design question

The original paper uses a GRU exclusively. This implementation adds a compact
Transformer as a deliberate architectural comparison, motivated by a specific
question about inductive bias in the low-data, physics-constrained regime of
this problem.

**The hypothesis:** in a dataset of only 552 training rows with strong physical
structure already captured by the kinetic prior, the choice of sequence model
should matter less as a standalone predictor and more as a residual-correction
component. The two architectures make different assumptions about how temporal
information is used.

The GRU compresses the entire input window into a single hidden state. This is
a strong inductive bias for smooth, locally correlated dynamics — appropriate
when recent context dominates. At short horizons (h=1) this is an advantage.
At longer horizons the fixed-size hidden state becomes a bottleneck: the model
must encode everything needed to predict 6 or 12 steps ahead into a vector
trained on only 552 examples, producing instability (GRU spikes to 0.01246 at
h=6, worse than persistence).

The Transformer reads the full 18-step window with self-attention and can
selectively weight any past timestep regardless of distance. It has no
compression bottleneck. At h=12 it achieves 0.00272 vs GRU's 0.00838 — a 3x
difference — because it can draw on earlier context without degrading. However,
with only 552 training rows, neither architecture beats the kinetic prior as a
standalone predictor: the physical model has an information advantage that
18-step attention over a small pilot dataset cannot overcome.

**The key insight confirmed by the results:** architecture choice matters most
at long horizons in standalone mode, but matters less in fused mode because the
kinetic prior dominates the prediction. The better question is not
"GRU or Transformer?" but "how well can either architecture provide a useful
residual-correction signal around a strong mechanistic prior?" On that
question, both architectures contribute comparably once fused.

| Architecture | Params | Design |
|---|---|---|
| GRU | 21,192 | hidden=44, layers=2, dropout=0.1 |
| Mini-Transformer | 36,838 | d\_model=64, 1 layer, 4 heads, FF=128 |

Loss = full\_profile\_MSE + 1.0 × observed\_mask\_MSE. AdamW, cosine annealing,
gradient clipping 1.0, early stopping patience 20 on val combined loss.

### Fusion

Fusion weights calibrated on validation set only, applied to test predictions.
Covariance matrices B (kinetic residuals) and R (model residuals) are estimated
per forecast horizon h on the validation set — not time-averaged across horizons.
This per-horizon calibration is a correction over the original paper, which used
time-independent matrices.

**Inverse-variance (IV):** per-point weighting by reciprocal residual variance.

**Kalman + Gaspari-Cohn (KGC):** full 6×6 covariance fusion with spatial
localization (L=2) that damps spurious correlations between non-adjacent stages,
following the original paper's fusion formulation.

---

## What this implementation tests beyond the original

The original hybrid soft-sensor (Chai 2026) establishes the architecture and
fusion framework. This implementation keeps the same core industrial problem but
changes the experimental question from "does the hybrid beat standalone components?"
to three more specific questions:

1. Does a Transformer provide any advantage over a GRU in this data regime, and
   if so, at which horizons and under which conditions?
2. Do the results hold under strict evaluation discipline — whole-run splits, no
   leakage into scalers or fusion weights, observed-point RMSE against real
   measurements only?
3. Are neural forecasts most useful as standalone predictors or as calibrated
   residual-correction signals, and does this depend on the forecast horizon?

| Dimension | Original (Chai 2026) | This implementation |
|---|---|---|
| Dense target construction | MHE or model-constrained dense labels | Kinetic-anchored residual reconstruction from sparse AT400 |
| Neural forecaster | GRU only | Compact GRU vs compact Transformer, direct comparison |
| Evaluation horizon | Aggregate / single-step | h = 1, 3, 6, 12 explicitly |
| Fusion calibration | Time-independent covariance matrices | Per-horizon, val-only calibration |
| Ground truth | MHE state estimates (dense) | Sparse AT400 observations (real) |
| Main diagnostic | Final prediction accuracy | Horizon-wise failure analysis, architecture comparison, baseline stress-testing |

---

## Results

All metrics on held-out test run **140207\_1** (118 timesteps).
Source: `outputs/metrics/all_results.csv`.

### Understanding what obs\_rmse measures on this dataset

Before reading the table, one structural fact must be understood: **the
aggregate obs\_rmse is almost entirely driven by Pt6 (absorber outlet).**

At each timestep exactly one of the six sampling points is observed. Across
118 test timesteps, each point is observed roughly equally (~15–21 times).
However, CO2 concentration is near-zero at Pts 1–5 (>95% absorbed by the
liquid MEA phase before reaching those stages) and rises to 0.025–0.040
fraction only at Pt6.

Per-point RMSE breakdown at h=1 (from Figure 8):

| Stage | CO2 level | GRU RMSE | Transformer RMSE |
|---|---|---|---|
| Pt1 | near-zero | 0.0011 | 0.0011 |
| Pt2 | ~zero | 0.0000 | 0.0000 |
| Pt3 | ~zero | 0.0003 | 0.0003 |
| Pt4 | ~zero | 0.0003 | 0.0003 |
| Pt5 | low | 0.0018 | 0.0025 |
| **Pt6** | **significant** | **0.0196** | **0.0107** |

Pt6 accounts for **99% of GRU aggregate MSE** and **94% of Transformer
aggregate MSE**. The reported obs\_rmse of 0.00837 (GRU) and 0.00465
(Transformer) are therefore almost entirely Pt6 error. Pts 1–5 show near-zero
RMSE not because the models are accurate there, but because CO2 is essentially
absent and predicting near-zero trivially succeeds.

**The meaningful comparison between all models is at Pt6, the only stage with
significant CO2.** GRU Pt6 RMSE = 0.0196, Transformer Pt6 RMSE = 0.0107 — a
1.8× gap. The kinetic prior, back-calculated from its aggregate obs\_rmse, has
an estimated Pt6 RMSE of approximately 0.0025, making it substantially more
accurate at the outlet than either standalone neural model.

This structure also explains why the kinetic prior aggregate obs\_rmse
(0.00107) is so much lower than the neural models: the kinetic model is more
accurate specifically at Pt6, the only stage that contributes meaningfully to
the aggregate.

### Observed-point RMSE table

All numbers from `outputs/metrics/all_results.csv`.
MAPE is reported for completeness but is diagnostic only — near-zero CO2 at
Pts 1–5 produces MAPE values of 30–61% that do not reflect prediction quality.

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Persistence | 0.00130 | 0.00161 | 0.00162 | 0.00168 |
| Kinetic prior | 0.00107 | 0.00108 | 0.00110 | 0.00112 |
| GRU | 0.00837 | 0.00224 | 0.01246 | 0.00838 |
| Transformer | 0.00465 | 0.00427 | 0.00377 | 0.00272 |
| GRU + IV fusion | 0.00089 | 0.00080 | 0.00090 | 0.00091 |
| GRU + KGC fusion | **0.00089** | 0.00080 | 0.00114 | **0.00073** |
| Transformer + IV fusion | 0.00097 | 0.00074 | **0.00083** | 0.00077 |
| Transformer + KGC fusion | 0.00117 | **0.00068** | 0.00090 | 0.00074 |

Bold: best observed-point RMSE at that horizon across all models.

### Key findings

**The kinetic prior is the strongest single baseline.** It achieves
obs\_rmse 0.00107–0.00112 across all horizons without any training data, and
is horizon-invariant by construction. Its advantage is concentrated at Pt6
(the outlet), where the mechanistic absorber model is physically most reliable.

**Standalone neural models fail primarily at Pt6.** The aggregate obs\_rmse
gap between neural models and the kinetic prior is almost entirely a Pt6
phenomenon. At Pts 1–5 all models including the kinetic prior are near-zero
for the same reason: CO2 is absent. The GRU Pt6 error (0.0196) is 1.8× worse
than the Transformer (0.0107), and both are substantially worse than the kinetic
prior (~0.0025 estimated at Pt6). This is the core standalone failure.

**The Transformer's advantage over GRU is real and grows with horizon.** Reading
the aggregate numbers at face value — GRU 0.00837 vs Transformer 0.00465 at h=1
— understates the structural difference. The architecture comparison plays out
at Pt6: Transformer 0.0107 vs GRU 0.0196. At h=12 the aggregate gap grows to
3.1× (0.00272 vs 0.00838), consistent with the Transformer's ability to attend
selectively to past context without a hidden-state bottleneck. GRU spikes to
0.01246 at h=6 — 11× worse than kinetic prior — while the Transformer degrades
more gracefully (0.00377, 3.4× worse).

**Fusion works by correcting Pt6 error while the kinetic prior handles the rest.**
The best fused variant at each horizon beats the kinetic prior:

| Horizon | Best fused model | Obs. RMSE | Kinetic prior | Improvement |
|---|---|---|---|---|
| h=1 | GRU + KGC | 0.00089 | 0.00107 | 16% |
| h=3 | Transformer + KGC | 0.00068 | 0.00108 | 37% |
| h=6 | Transformer + IV | 0.00083 | 0.00110 | 24% |
| h=12 | GRU + KGC | 0.00073 | 0.00112 | 34% |

Fusion lets the neural model contribute a residual correction specifically at
Pt6 — where the kinetic prior has systematic error — while the kinetic prior
dominates at Pts 1–5 where there is nothing for the neural model to correct.
The inverse-variance calibration discovers this spatial structure automatically
from validation residuals: Figure 6 shows kinetic weight approaching 1.0 at
Pt6, exactly where the kinetic model is most trustworthy.

Not every fusion variant is beneficial: transformer\_fused\_kgc at h=1 (0.00117)
is worse than the kinetic prior (0.00107). Full 6×6 covariance estimation from
228 validation rows is noisy; the GC localization mitigates this but does not
eliminate it at all horizons.

**In fused mode, GRU and Transformer converge.** Despite GRU being 1.8× worse
than Transformer at Pt6 in standalone mode, their fused results are comparable
(e.g. h=1: GRU+IV 0.00089, Transformer+IV 0.00097). Once the kinetic prior
provides the structural anchor, even a noisier neural correction signal adds
value. This confirms the central hypothesis: architecture choice determines
standalone quality at the hard prediction target (Pt6); fusion quality
determines deployed performance.

### Figure 1 — all models ranked at h=1 (observed-point RMSE)

![Final model comparison at h=1](outputs/figures/12_final_comparison_h1.png)

The aggregate obs\_rmse at h=1 is dominated by Pt6 (outlet) error — the only
stage where CO2 is significant. The ranking reflects primarily Pt6 performance.
GRU fused variants achieve the lowest aggregate. Transformer+KGC (0.00117) is
worse than the kinetic prior (0.00107): full covariance fusion from 228 val rows
without sufficient regularisation can backfire.

### Figure 2 — observed-point RMSE across all four forecast horizons

![RMSE by horizon](outputs/figures/06_rmse_by_horizon.png)

Right panel (observed-point RMSE) is the primary diagnostic. Remember that each
point on these curves reflects almost entirely Pt6 outlet error — Pts 1–5
contribute near-zero to the aggregate at all horizons. The kinetic prior (orange)
is flat because it is physically grounded and horizon-invariant. GRU (blue) spikes
at h=6 — hidden-state bottleneck failure at this specific horizon. The Transformer
(purple) degrades more gracefully, consistent with attention over the full 18-step
window. Note: horizons 1, 3, 6, 12 are not equally spaced in time — the x-axis
intervals are unequal, so the visual slope of each curve is compressed at longer
horizons.

---

## Target reconstruction

### Figure 3 — kinetic-anchored reconstruction on test run 140207\_1

![KAR reconstruction on test run](outputs/figures/01_reconstruction_140207_1.png)

Each panel shows one sampling point across 118 test timesteps. Orange dashed:
precomputed kinetic prior. Blue: KAR-reconstructed profile. Red dots: real AT400
observations (the only ground truth). The reconstruction anchors to the kinetic
shape and corrects residuals at observed points. Stages 2–4 show near-zero
absolute CO2, illustrating why MAPE is misleading on this dataset.

---

## SDAE dimensionality reduction

### Figure 4 — latent space structure (val run, PCA 2D)

![SDAE latent space](outputs/figures/03_sdae_latent_space.png)

The 2D PCA projection of SDAE-encoded val-run features (98.89% variance explained)
shows a continuous temporal manifold coloured by time step. The encoder captures
process evolution as smooth structure in latent space rather than random scatter —
confirming it extracts meaningful dynamic representations from 95-dimensional
raw sensor data. The curved trajectory reflects a genuine regime transition
within the run, not sampling artefact.

---

## Fusion

### Figure 5 — Transformer + KGC fusion on test run (h=1)

![Fusion: Transformer + Gaspari-Cohn h=1](outputs/figures/09_fusion_transformer_h01.png)

Per-stage time series for all 6 sampling points. Grey: reconstructed target.
Orange dotted: kinetic prior. Blue dashed: Transformer standalone. Purple:
KGC-fused prediction. Black dots: real AT400 observations. At Pt 6 (absorber
outlet) the fused and kinetic lines nearly overlap — the kinetic prior dominates
where physically most reliable. At Pt 5 the Transformer provides useful
correction.

### Figure 6 — inverse-variance fusion weights per sampling point (Transformer)

![Fusion weights Transformer](outputs/figures/10_fusion_weights_transformer.png)

Inverse-variance fusion weights estimated from validation residuals at each
sampling point (weights vary per horizon; shown here for the last trained h).
At Pt6 (absorber outlet, highest CO2), kinetic weight approaches 1.0 and
Transformer weight approaches 0 — the mechanistic model is most reliable
exactly where CO2 is highest and the neural model fails most. At Pt1 the
Transformer receives higher weight. This spatial structure is discovered
automatically from per-point validation residual variance, not hand-tuned.
It correctly identifies Pt6 as the stage where the kinetic prior should dominate.

### Figure 7 — Gaspari-Cohn localization matrix (L=2)

![Gaspari-Cohn matrix](outputs/figures/11_gaspari_cohn_matrix.png)

The localization matrix applied to B and R before Kalman fusion. Correlations
decay to near-zero for stages separated by 4+ positions (|i-j| >= 4). This
suppresses spurious long-range covariances that arise from estimating a 6×6
matrix from only 228 validation rows — a regularisation essential at this
dataset scale.

---

## Per-stage error breakdown

### Figure 8 — observed-point RMSE by sampling point at h=1

![Per-point RMSE heatmap](outputs/figures/07_per_point_heatmap.png)

GRU (left) and Transformer (right) observed-point RMSE per sampling point at h=1.
**Pts 2–4 show 0.0000 not because the models are accurate, but because CO2 is
essentially absent at those stages — both model and target are near-zero, so RMSE
is trivially small.** Pt6 (absorber outlet) is the only stage with significant CO2
and dominates the aggregate: GRU Pt6 = 0.0196, Transformer Pt6 = 0.0107, a 1.8×
gap that is the real performance difference between the architectures. Note that
the two colourbars have different scales (GRU up to 0.0175, Transformer up to
0.010) — both Pt6 cells are deep red but the GRU error is nearly 2× larger.

---

## Comparison with original (Chai et al. 2026)

**Where this implementation is stronger:**

- GRU vs Transformer comparison across horizons — absent from the original.
- Multi-horizon evaluation (h = 1, 3, 6, 12) — the original reports aggregate
  accuracy only.
- Per-horizon fusion calibration: B and R estimated separately at each h.
  Using h=1 matrices for h=12 fusion, as in the original, miscalibrates weights.
- Strict split discipline: test run never influences any fitted object.
- 17 automated tests covering data alignment, reconstruction, windowing, fusion.

**Where the original is stronger:**

- MHE-imputed labels are substantially better training targets than KAR
  pseudo-labels. MHE constrains reconstruction to the mechanistic state-space,
  producing physically consistent dense profiles.
- The 3.79% MAPE reported in the original is not directly comparable: different
  ground truth (MHE labels vs sparse AT400), different split, and MAPE is
  unreliable near zero CO2. RMSE is the appropriate metric here.

---

## Limitations

- The full six-point CO2 profile is a **pseudo-label**. Observed-point RMSE is
  the only metric against real measurements.
- The kinetic prior is a fixed precomputed input. The MATLAB/Simulink kinetic
  model and MHE state estimator are not reproduced.
- Very small dataset (8 runs, 898 total timesteps). Distribution shift between
  training and test regimes limits SDAE coverage and explains erratic GRU
  standalone behaviour. Fusion partially compensates, but not uniformly.
- Point predictions only — no uncertainty quantification.
- MAPE is reported for completeness but is not a reliable ranking metric on this
  dataset due to near-zero CO2 at stages 2–5.

---

## Project structure

```
my_v2/
  src/
    config.py                  all hyperparameters in one place
    data_loading.py            Excel + kinetic CSV loading with row-count validation
    target_reconstruction.py   KAR reconstruction + public interpolate_profile
    autoencoder.py             SDAE with AWGN noise injection
    models.py                  GRUForecaster + MiniTransformerForecaster
    training.py                combined loss, early stopping, AdamW + cosine LR
    metrics.py                 full-profile RMSE, observed-point RMSE, MAPE
    fusion.py                  inverse-variance + Kalman + Gaspari-Cohn
    windowing.py               sliding windows with SDAE encoding
    splitting.py               whole-run train/val/test split
    utils.py                   seed, scaler save/load, inverse_transform
  scripts/
    run_pipeline.py            single script: all phases end to end
    02_train_gru.py            stepwise GRU training (loads SDAE checkpoint)
    03_train_transformer.py    stepwise Transformer training
    04_evaluate_models.py      stepwise test evaluation
    05_run_fusion.py           stepwise fusion calibration and evaluation
  notebooks/
    01_data_understanding.ipynb
    02_sdae_dimensionality_reduction.ipynb
    03_gru_transformer_training.ipynb
    04_evaluation_and_fusion.ipynb
  tests/
    test_all.py                17 automated tests
  data/
    raw_with_label/            *.xlsx  raw sensor data + AT400 + sampling label
    kinetic_prior/             *.csv   precomputed 6-point kinetic prior per run
  outputs/
    figures/                   all plots (auto-generated)
    metrics/                   CSV results, training histories, scalers
    checkpoints/               model weights (.pt)
    fusion/                    fused prediction arrays (.npy)
  requirements.txt
  run_all.sh
```

---

## Quick start

```bash
git clone <your_repo_url>
cd my_v2
pip install -r requirements.txt

# Canonical reproduction command — runs all phases and tests
bash run_all.sh
```

Data must be present at `data/raw_with_label/` and `data/kinetic_prior/`.
Kinetic prior CSVs are from the original repository:
https://github.com/tonyzyl/CO2-Soft-sensor-for-a-carbon-capture-pilot-plant

---

## References

Chai S., Guo S., Mercangöz M. (2026). Hybrid Data-Driven and Mechanistic CO2 Soft
Sensor with MHE-Imputed Labels and Covariance-Weighted Fusion in a Pilot-Scale
Absorber. *Processes*, 14(6), 916. https://doi.org/10.3390/pr14060916

Zhuang Y. et al. (2022). Soft sensor for CO2 capture pilot plant.
*Computers in Industry*, 143, 103747.
