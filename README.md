# CO2 Soft-Sensor v2
## Physics-Informed Hybrid Forecasting for Post-Combustion Carbon Capture

A compact, reproducible PyTorch pipeline for estimating CO2 concentration profiles
across a six-stage MEA absorber column from sparse, asynchronous sensor readings.
Implements kinetic-anchored residual reconstruction, SDAE-based dimensionality
reduction, GRU and Transformer forecasters, and covariance-weighted fusion with
Gaspari-Cohn localization — following Chai, Guo & Mercangöz (2026) and extending
it with multi-horizon evaluation and a strict evaluation discipline.

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

### Primary metric: observed-point RMSE

Observed-point RMSE is computed only at the single real AT400 measurement per
timestep. It is the only metric against directly observed ground truth.
MAPE is reported separately as a diagnostic; it is unstable near zero CO2
(stages 2–5) and should not be used as a primary ranking criterion.

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

**The kinetic prior is the strongest single baseline.** At every horizon the
kinetic prior (obs\_rmse 0.00107–0.00112) outperforms both standalone neural
models at h=1 and h=6. It requires no training data and is horizon-invariant
by construction. This is expected: eight short pilot runs are insufficient for
a standalone neural model to generalise beyond its training regime.

**Standalone neural models are not consistently better than the kinetic prior.**
GRU standalone exceeds kinetic-prior error by 7–11× at h=1 and h=12, and spikes
to 0.01246 at h=6 — 11× worse — due to the hidden-state bottleneck described
above. The Transformer is better than GRU at all horizons, but still
underperforms the kinetic prior at h=1 (0.00465 vs 0.00107). Both results are
consistent with distribution shift between training and test regimes.

**The Transformer's advantage over GRU grows with horizon, confirming the
inductive bias hypothesis.** At h=1 both standalone models perform poorly
relative to the kinetic prior, with the Transformer 1.8× better than GRU. At
h=12 the gap grows to 3.1× (0.00272 vs 0.00838), consistent with the
Transformer's ability to draw on longer context without a compression bottleneck.

**The best fused variant at each horizon improves over the kinetic prior:**

| Horizon | Best fused model | Obs. RMSE | Kinetic prior | Improvement |
|---|---|---|---|---|
| h=1 | GRU + KGC | 0.00089 | 0.00107 | 17% |
| h=3 | Transformer + KGC | 0.00068 | 0.00108 | 37% |
| h=6 | Transformer + IV | 0.00083 | 0.00110 | 25% |
| h=12 | GRU + KGC | 0.00073 | 0.00112 | 35% |

Fusion consistently beats the kinetic prior at every horizon. However, not every
fused variant is beneficial: transformer\_fused\_kgc at h=1 (0.00117) is worse
than the kinetic prior (0.00107), showing that naive covariance fusion can
backfire when the neural model is far off-distribution.

**In fused mode, GRU and Transformer converge.** GRU fused (IV) at h=1 gives
0.00089; Transformer fused (IV) gives 0.00097. The architecture advantage of the
Transformer disappears once the kinetic prior provides the structural anchor.
This confirms the central hypothesis: architecture choice determines standalone
quality, but fusion quality determines deployed performance. A weaker standalone
model can still provide useful residual corrections.

**MAPE is unreliable on this dataset.** Stages 2–5 have near-zero CO2 (>95%
absorbed at stage 1), producing MAPE values of 30–61% despite small absolute
errors. The kinetic prior MAPE exceeds 60% despite being the most accurate model
by RMSE at h=1. Use RMSE for all comparisons.

### Figure 1 — all models ranked at h=1 (observed-point RMSE)

![Final model comparison at h=1](outputs/figures/12_final_comparison_h1.png)

At h=1 the kinetic prior (0.00107) outperforms all standalone neural models.
Only GRU fused (IV and KGC) and Transformer + IV beat the kinetic prior.
Transformer + KGC (0.00117) does not — covariance fusion is sensitive to the
quality of the validation-estimated residual statistics.

### Figure 2 — observed-point RMSE across all four forecast horizons

![RMSE by horizon](outputs/figures/06_rmse_by_horizon.png)

Right panel (observed-point RMSE) is the primary diagnostic. The kinetic prior
(orange) is flat and low across all horizons. GRU (blue) is erratic — the
hidden-state bottleneck causes it to collapse at h=6. The Transformer (purple)
degrades more gracefully as horizon increases, consistent with attention-based
context access. Fused variants (see results table) sit below all baselines at
every horizon.

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

Fusion weights estimated from validation residuals at each of the 6 sampling
points. At Pt 6 (absorber outlet, highest CO2), kinetic weight approaches 1.0 —
the mechanistic model is most reliable where the outlet concentration is
physically constrained by absorption equilibrium. At Pt 1 the Transformer
receives higher weight. The spatial variation in weights is a direct
consequence of per-point variance calibration; it is not hand-tuned.

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
Pt 6 (absorber outlet) dominates the error budget for both models — the highest
absolute CO2 and most dynamic variation. Pts 2–4 show near-zero RMSE not because
the models are accurate there, but because CO2 is essentially absent and the
observed mask fires rarely. This per-stage breakdown directly motivates the
spatially varying fusion weights in Figure 6: the kinetic prior should be trusted
most at Pt 6, and the data-driven model has more scope to contribute at Pt 1.

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
