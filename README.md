# CO2 Soft-Sensor v2
## Physics-Informed Hybrid Forecasting for Post-Combustion Carbon Capture

Inspired by and adapted from Chai et al. 2026: SDAE-style dimensionality reduction, GRU forecasting, covariance-weighted fusion, Gaspari-Cohn localization, and a Kalman-style fusion formula. Several hyperparameters are taken from Chai 2026 Tables 1–2, including GRU hidden size, dropout, batch size, learning rate, and training epochs. The implementation differs in important ways: it uses 6 sampling points, 95 input features, a 16-dimensional latent space, frozen SDAE pretraining, and kinetic-anchored residual reconstruction instead of full MHE-generated labels.

**Reproduced from the paper:** SDAE-based dimensionality reduction
(95 -> 16 latent features), GRU forecaster, covariance-weighted fusion with
Gaspari-Cohn localization (L=2), and the Kalman fusion formula. Most
hyperparameters (SDAE latent dim, learning rate, batch size, GRU hidden=44,
dropout) come directly from Chai 2026 Tables 1-2.

**Extensions added in this implementation:**

1. **Task reframing**: from single-step state estimation (paper) to explicit
   multi-horizon forecasting at h in {1, 3, 6, 12} steps = {43s, 2.1min,
   4.3min, 8.6min} of physical time.
2. **Architecture comparison**: Mini-Transformer (d_model=64, 1 layer, 4
   heads, 36,838 params) added alongside the GRU.
3. **Fusion ablation**: inverse-variance (point-wise) and Kalman-Gaspari-Cohn
   (full 6x6) variants both evaluated. Paper uses Kalman-GC only.
4. **Per-horizon B/R calibration** on validation set.
5. **Persistence baseline** added to comparison table.

### Two key simplifications

1. **Moving Horizon Estimation (MHE) is not implemented.** The paper uses
   MHE with the mechanistic ODE model (IPOPT/CasADi) to produce physically
   consistent dense CO2 labels. This implementation uses Kinetic-Anchored
   Residual (KAR) reconstruction as a cheaper proxy: residuals between AT400
   and a precomputed kinetic prior are interpolated rather than estimated
   through a constrained dynamical optimization. KAR labels are less
   physically consistent than MHE labels.

2. **The mechanistic model is not reproduced.** Precomputed kinetic-prior
   CSVs from the original repository (Zhuang 2022) are used as a fixed
   input. The mechanistic model parameters, MEA reaction kinetics, and
   CSTR-stage formulation from Chai 2026 Section 3.1 are not regenerated.

The framework's main paper contribution producing physically grounded
training labels via state estimation and it is therefore not present in this
work. The SDAE-GRU architecture and the fusion machinery are reproduced
faithfully.

---

## Problem

A rotating gas analyzer (AT400) samples CO2 concentration at one absorber
sampling point per timestep across a 43-second sampling period (verified on
all 8 runs: median dt = 43.0s, jitter 0-1s). Only one of six sampling
points is observed at each timestep. The goal is to forecast the full
six-point CO2 concentration profile h steps ahead, under partial
observability, with only 8 short pilot runs (50-228 rows each) available.

---

## Methodology

### Data and split

| Split | Runs | Rows | Role |
|-------|------|------|------|
| Train | 140120\_1, 140206\_1, 140207\_2, 140214\_1, 140214\_2, 140227\_1 | 552 | Scaler fit, SDAE, model training |
| Val | 140313\_1 | 228 | Early stopping, fusion calibration only |
| **Test** | **140207\_1** | **118** | **Final evaluation** |

Split is by whole run only. Sliding windows never cross run boundaries.
The test run is held out strictly from scaler fitting, SDAE training, early
stopping, and fusion calibration.

### Sampling-point geometry

The dataset has **6 sampling points**, not 5 as in Chai 2026 Section 3.1.
Empirical analysis indicates:

- Point 1: clean gas outlet (top of column, after absorption)
- Points 2-5: internal packing stages
- Point 6: flue gas inlet (bottom of column, before absorption)

Point 6 carries the bulk of measurable CO2 (~0.03-0.05 fraction). Points
1-5 range from near-zero to ~0.01 because >95% of CO2 is absorbed by stage
1. This geometry explains why aggregate obs\_RMSE is dominated by Pt6
performance. Forward column-to-point ordering verified empirically (forward
mapping gives 21x lower error than reversed mapping).

### Target reconstruction: kinetic-anchored residuals (KAR)

Direct linear interpolation of sparse AT400 readings produces distorted
training targets. This pipeline instead computes residuals between AT400
and the precomputed kinetic prior at observed points, interpolates those
residuals (which are 5-19x smaller in standard deviation than the raw
signal across the 8 pilot runs), then adds them back:

```
r(t, pt) = AT400(t)/100 - kinetic(t, pt)      # at observed point only
r_profile = interpolate_per_channel(r_sparse)  # fill all 6 channels
y_profile = kinetic + r_profile                # full reconstructed profile
```

**Kinetic prior fit against observed AT400 (across all 8 runs):**

| Split | n | R² | Residual RMSE | Std ratio |
|-------|---|-----|---------------|-----------|
| Train pooled (6 runs) | 552 | 0.992 | 0.00208 | 0.088 |
| Val (140313\_1) | 228 | 0.972 | 0.00421 | 0.156 |
| Test (140207\_1) | 118 | 0.991 | 0.00110 | 0.087 |

Per-run R² ranges from 0.968 to 0.997. These values are computed at
observed points only and quantify how well the kinetic prior matches the
single observed reading at each timestep. They cannot be interpreted as a
global fit quality across the full six-point profile, since five of the
six points are unobserved at each timestep.

The six-point profile is a pseudo-label. **Observed-point RMSE** - computed
only at the single AT400 measurement per timestep - is the primary metric
throughout.

### Feature pipeline

```
89 numeric sensors + 6 one-hot sampling-point indicators = 95 features
  -> StandardScaler (fit on train only)
  -> SDAE encoder: 95 -> 16 latent dims  (AWGN noise injection, unsupervised)
  -> Sliding window length W = 18 steps (one full AT400 rotation = ~13 min)
  -> Target: y_profile[t + h],  h in {1, 3, 6, 12} = {43s, 2.1min, 4.3min, 8.6min}
```

**Why W=18.** Empirically, AT400 dwells median=3 samples per sampling point
(range 2-3 across all 8 runs); one full 6-point sweep is therefore 18
samples. Each input window contains at least one observation at every
sampling point.

### Architectures

| Architecture | Params | Design |
|---|---|---|
| GRU | 21,192 | hidden=44, layers=2, dropout=0.1 |
| Mini-Transformer | 36,838 | d\_model=64, 1 layer, 4 heads, FF=128 |

GRU hidden=44 matches Chai 2026 Table 2 exactly. Transformer dimensions
are chosen to be parameter-comparable for fair architectural comparison
(within 1.7x).

Loss = full\_profile\_MSE + 1.0 x observed\_mask\_MSE. AdamW, cosine
annealing, gradient clipping 1.0, early stopping patience 20 on val
combined loss.

**Note on loss weighting.** alpha=1.0 in `full_mse + alpha * obs_mse`
gives the full-profile term effectively 6x more gradient weight than the
observed-point term, because the former averages over all 6 sampling
points per timestep while the latter averages over the single observed
point. This means the model is effectively trained more against KAR
pseudo-labels than against real AT400 measurements. A targeted ablation
over alpha is an obvious extension not performed here.

### Fusion

Fusion weights calibrated on validation set only, applied to test
predictions. Covariance matrices B (kinetic residuals) and R (model
residuals) are estimated per forecast horizon h on the validation set.

**Inverse-variance (IV):** per-point weighting by reciprocal residual
variance.

**Kalman + Gaspari-Cohn (KGC):** full 6x6 covariance fusion with spatial
localization (L=2). The Gaspari-Cohn localization preserves positive
semidefiniteness of the covariance matrix via the Schur (Hadamard) product
theorem, since both the GC localization matrix and the residual covariance
B are PSD by construction.

**Estimation target differs from the paper.** In Chai 2026 Equation 24, B
and R are estimated against MHE state estimates as ground truth:
`R = Cov(y_GRU - y_MHE)`, `B = Cov(y_mec - y_MHE)`. In this implementation,
MHE is not available, so residuals are computed against KAR pseudo-labels:
`R = Cov(model_pred - y_KAR)`, `B = Cov(kinetic - y_KAR)`. Since
`y_KAR = kinetic + interp(residuals)`, the computed B is closer to the
variance of the interpolated residual term than to the true
kinetic-vs-truth error covariance. This is a known approximation; fusion
weights still recover sensible spatial structure empirically.

---

## Results

All metrics on held-out test run **140207\_1** (118 timesteps).
Source: `outputs/metrics/all_results.csv`.

### Metric choice

The primary metric is **obs_RMSE**: RMSE computed only at observed AT400
points. This is the only metric in the table that uses real measurements
as ground truth. Other metrics (full_RMSE, full_MAE, MAPE) include
KAR-reconstructed pseudo-labels in the denominator and are reported as
diagnostics only.

**Why not MAPE.** The paper reports MAPE 3.79% as its headline. On this
dataset MAPE is structurally unreliable: at sampling points 1-5, CO2
fraction is near-zero (often < 0.001), and dividing small absolute errors
by near-zero targets inflates MAPE to 30-60% even for visually accurate
predictions. The kinetic prior has MAPE 60.8% (looks catastrophic) but
obs_RMSE 0.00107 (best standalone). The disconnect is dataset structure,
not method failure. obs_RMSE in absolute units is the appropriate metric;
MAPE values are kept in the CSV for completeness only and should not be
used for ranking.

### Observed-point RMSE table

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Persistence | 0.00130 | 0.00161 | 0.00162 | 0.00168 |
| Kinetic prior | 0.00107 | 0.00108 | 0.00110 | 0.00112 |
| GRU standalone | 0.00957 | 0.00282 | 0.00220 | 0.00863 |
| Transformer standalone | 0.00496 | 0.00946 | 0.00296 | 0.00346 |
| GRU + IV fusion | 0.00084 | 0.00074 | 0.00080 | 0.00102 |
| GRU + KGC fusion | 0.00092 | **0.00069** | **0.00076** | 0.00151 |
| Transformer + IV fusion | **0.00080** | 0.00078 | 0.00080 | 0.00089 |
| Transformer + KGC fusion | 0.00102 | 0.00115 | 0.00078 | **0.00086** |

Bold: best obs_RMSE per horizon.

### Best fused vs kinetic prior

| Horizon | Best fused | obs_RMSE | Kinetic | Improvement |
|---------|-----------|----------|---------|-------------|
| h=1 | Transformer + IV | 0.00080 | 0.00107 | 25% |
| h=3 | GRU + KGC | 0.00069 | 0.00108 | 36% |
| h=6 | GRU + KGC | 0.00076 | 0.00110 | 30% |
| h=12 | Transformer + KGC | 0.00086 | 0.00112 | 23% |

**Headline result:** Fused models improve over the kinetic prior baseline
by 23-36% across all four forecast horizons. The best single result is
GRU + KGC at h=3 with obs_RMSE = 0.00069 (36% reduction).

### Key findings

**The kinetic prior is the strongest standalone baseline.** It achieves
obs_RMSE 0.00107-0.00112 across all horizons without any training data
and is horizon-invariant by construction. Persistence (0.00130-0.00168)
is weaker because it lacks physical dynamics.

**Standalone neural models are 2-9x worse than the kinetic prior at every
horizon.** Neither GRU nor Transformer beats the physical baseline:

| Horizon | GRU std | TF std | Kinetic | GRU/Kin | TF/Kin |
|---------|---------|--------|---------|---------|--------|
| h=1 | 0.00957 | 0.00496 | 0.00107 | 9.0x | 4.6x |
| h=3 | 0.00282 | 0.00946 | 0.00108 | 2.6x | 8.8x |
| h=6 | 0.00220 | 0.00296 | 0.00110 | 2.0x | 2.7x |
| h=12 | 0.00863 | 0.00346 | 0.00112 | 7.7x | 3.1x |

Standalone failure is partly explained by SDAE distribution shift (val MSE
~4.0 vs train ~0.15, a 27x gap), partly by the small training set (552
rows), and partly by the kinetic prior's structural advantage at Pt6 -
the only sampling point with non-trivial CO2 concentration.

**Architecture spike patterns are inconsistent across horizons.** GRU
spikes badly at h=1 and h=12; Transformer spikes badly at h=3. Neither
architecture dominates the other consistently. With only one test run and
no multi-seed evaluation, these patterns are best understood as
small-dataset variance rather than clean architectural signals.

**Fusion works through automatic shrinkage to the kinetic prior, not
through complementary signal.** Inverse-variance weighting assigns
near-zero weight to neural predictions at points where they are noisy
(particularly Pt6, where the kinetic prior is most accurate). The fused
prediction is dominated by the kinetic prior with small corrections from
the neural model at points where it has useful signal. This is the
standard inverse-variance behavior; it matches the paper's mechanism but
operates with a different residual reference (KAR vs MHE).

**IV fusion is competitive with KGC fusion.** Across all horizons, IV and
KGC fusion are within 20% of each other:

- h=1: TF+IV (0.00080) beats TF+KGC (0.00102)
- h=3: GRU+KGC (0.00069) marginally beats GRU+IV (0.00074)
- h=6: GRU+KGC (0.00076) marginally beats GRU+IV (0.00080)
- h=12: TF+KGC (0.00086) marginally beats TF+IV (0.00089)

This is a useful negative finding: the additional 6x6 covariance
estimation in KGC does not decisively outperform the simpler per-point
weighting on this dataset size. With 228 validation rows, full covariance
matrices are noisy enough that the GC localization regularization barely
recovers, except at specific horizons.

**In fused mode, GRU and Transformer perform comparably.** Despite
standalone GRU being 2-9x off from the kinetic prior, fused GRU achieves
the best result overall (h=3, 36% improvement). The data-driven
architecture matters less when fusion can absorb noise; the kinetic prior
sets the floor and the neural component contributes residual correction.

### Caveats on the results

1. **Single test run.** All headline numbers are point estimates on
   one held-out run (140207_1, 118 timesteps, 84 minutes of plant
   operation). The original paper uses two blind test runs (datasets 6
   and 7). Across-run variance is unquantified here. Whether the
   23-36% improvements generalize to other operating regimes is not
   verified.

2. **No uncertainty quantification.** No bootstrap confidence intervals,
   no multi-seed runs. Each obs_RMSE in the table is a single number
   with no error bar. Single-seed variance and small-dataset
   sensitivity could be substantial.

3. **Comparison with the paper's 3.79% MAPE is not meaningful.**
   Different ground truth (MHE estimates vs sparse AT400), different
   split, different metric. Direct numerical comparison would mislead
   in either direction.

### Figure 1 - all models ranked at h=1

![Final model comparison at h=1](outputs/figures/12_final_comparison_h1.png)

### Figure 2 - obs_RMSE across all four forecast horizons

![RMSE by horizon](outputs/figures/06_rmse_by_horizon.png)

Right panel (obs_RMSE) is the primary diagnostic. The kinetic prior
(orange) is flat by construction. Both neural standalone curves are
noisy across horizons. All fused variants stay near or below the
kinetic baseline.

---

## Target reconstruction

### Figure 3 - KAR reconstruction on test run 140207\_1

![KAR reconstruction on test run](outputs/figures/01_reconstruction_140207_1.png)

Each panel shows one sampling point across 118 test timesteps. Orange
dashed: precomputed kinetic prior. Blue: KAR-reconstructed profile. Red
dots: real AT400 observations (the only ground truth). The reconstruction
anchors to the kinetic shape and corrects residuals at observed points.
Stages 2-4 show near-zero absolute CO2, illustrating why MAPE is
misleading on this dataset.

---

## SDAE dimensionality reduction

### Figure 4 - latent space structure (val run, PCA 2D)

![SDAE latent space](outputs/figures/03_sdae_latent_space.png)

The 2D PCA projection of SDAE-encoded val-run features (98.89% variance
explained) shows continuous temporal structure coloured by time step.
Process variables (temperatures, flows) have smooth temporal evolution by
construction, and this structure is preserved through the encoder.

The SDAE shows substantial distribution shift on val/test (val MSE ~4.0
vs train ~0.15, a 27x gap). This indicates the encoder generalizes poorly
beyond the 6 training runs - a fundamental limitation given dataset size
rather than a bottleneck-capacity issue. Latent dim=16 is taken from
Chai 2026 Table 1; ablation over {8, 16, 32} not performed.

---

## Fusion

### Figure 5 - Transformer + KGC fusion on test run (h=1)

![Fusion: Transformer + Gaspari-Cohn h=1](outputs/figures/09_fusion_transformer_h01.png)

Per-stage time series for all 6 sampling points. Grey: reconstructed
target. Orange dotted: kinetic prior. Blue dashed: Transformer standalone.
Purple: KGC-fused prediction. Black dots: real AT400 observations. At
Pt6 (flue gas inlet) the fused and kinetic lines nearly overlap - the
kinetic prior dominates where physically most reliable.

### Figure 6 - inverse-variance fusion weights per sampling point (Transformer)

![Fusion weights Transformer](outputs/figures/10_fusion_weights_transformer.png)

Inverse-variance fusion weights estimated from validation residuals at
each sampling point. At Pt6 (flue gas inlet, highest CO2), kinetic weight
approaches 1.0 and Transformer weight approaches 0. Mechanistically: Pt6
is the gas inlet where CO2 concentration is essentially set by feed
conditions which the kinetic model knows explicitly, so kinetic residual
variance is low there. The neural model has high residual variance and
gets correspondingly down-weighted. This spatial structure is discovered
automatically from per-point validation residual variance, not hand-tuned.

### Figure 7 - Gaspari-Cohn localization matrix (L=2)

![Gaspari-Cohn matrix](outputs/figures/11_gaspari_cohn_matrix.png)

The localization matrix applied to B and R before Kalman fusion.
Correlations decay to near-zero for stages separated by 4+ positions.
This suppresses spurious long-range covariances that arise from estimating
a 6x6 matrix from only 228 validation rows.

---

## Per-stage error breakdown

### Figure 8 - obs_RMSE by sampling point at h=1

![Per-point RMSE heatmap](outputs/figures/07_per_point_heatmap.png)

GRU (left) and Transformer (right) obs_RMSE per sampling point at h=1.
Pts 2-4 show near-zero values not because the models are accurate, but
because CO2 is essentially absent at those stages - both model and target
are near-zero, so RMSE is trivially small. Pt6 (flue gas inlet) is the
only stage with significant CO2 and dominates the aggregate.

---

## Comparison with original (Chai et al. 2026)

| Dimension | Chai 2026 | This implementation |
|---|---|---|
| Dense label construction | MHE with mechanistic ODE (IPOPT/CasADi) | KAR: residual interpolation |
| Mechanistic model | Reproduced (5-stage CSTR, Section 3.1) | Not reproduced (precomputed CSV used) |
| Neural forecaster | GRU only | GRU vs Mini-Transformer compared |
| Sampling points | 5 stages | 6 columns (inlet + 5 stages) |
| Task framing | Single-step state estimation | Multi-horizon forecasting h={1,3,6,12} |
| Fusion variants | Kalman-GC only | Inverse-variance + Kalman-GC |
| Covariance estimation | Sliding-window batch updates | Per-horizon val-only calibration |
| B, R reference | MHE estimates (Eq. 24) | KAR pseudo-labels (approximation) |
| Test sets | 2 blind runs (datasets 6, 7) | 1 held-out run (140207_1) |
| Primary metric | MAPE (3.79% reported) | obs_RMSE = 0.00069 best (36% improvement) |

**Where the paper is fundamentally stronger:** MHE-imputed labels are
substantially better training targets than KAR pseudo-labels; the
mechanistic model itself is implemented and used both for MHE and as
deployment-time predictor; two blind test sets give across-run robustness.

**Where this implementation adds value:** task reframing to multi-horizon
forecasting characterizes how each architecture degrades with prediction
distance; Transformer-vs-GRU benchmark in the same fusion framework;
inverse-variance fusion baseline shows that full 6x6 covariance is not
strictly necessary on small datasets.

---

## Limitations

- **Single test run.** Headline numbers are point estimates on one
  held-out run. No across-run variance estimation. LOO-CV across the 7
  non-test runs would be the obvious next step.
- **No uncertainty quantification.** No confidence intervals on obs_RMSE.
  Bootstrap on test predictions or multi-seed training would resolve
  this.
- **Pseudo-label targets.** Full six-point CO2 profile is a KAR
  reconstruction, not a measurement. obs_RMSE on observed points is the
  only metric against real measurements.
- **Mechanistic model not reproduced.** Kinetic prior is a fixed
  precomputed input. The MATLAB/Simulink kinetic model and MHE state
  estimator from Chai 2026 are not implemented here.
- **Small dataset.** 8 runs total, 898 timesteps. After windowing,
  effective training set drops to 150-200 windows at h=12. SDAE
  distribution shift (val MSE 27x train MSE) reflects this limit.
- **No hyperparameter ablation.** SDAE latent_dim, window length, alpha
  loss weight all taken from paper or chosen heuristically. Reported
  results are sensitive to these choices.
- **MAPE structurally unreliable on this dataset.** Near-zero CO2 at
  Pts 1-5 produces MAPE values that do not reflect prediction quality;
  reported in CSV for completeness but not used for ranking.
- **B and R estimated against KAR pseudo-labels, not MHE estimates** as
  in Chai 2026 Equation 24. Fusion weights still recover sensible
  structure empirically, but the theoretical correspondence with the
  paper's covariance derivation is partial.
- **Point predictions only** - no probabilistic forecasting.

---

## Project structure

```
my_v2/
  src/
    config.py                  all hyperparameters in one place
    data_loading.py            Excel + kinetic CSV loading with row-count validation
    target_reconstruction.py   KAR reconstruction
    autoencoder.py             SDAE with AWGN noise injection
    models.py                  GRUForecaster + MiniTransformerForecaster
    training.py                combined loss, early stopping, AdamW + cosine LR
    metrics.py                 full-profile RMSE, observed-point RMSE, MAPE
    fusion.py                  inverse-variance + Kalman + Gaspari-Cohn
    windowing.py               sliding windows with SDAE encoding
    splitting.py               whole-run train/val/test split
    utils.py                   seed, scaler save/load
  scripts/
    run_pipeline.py            single script: all phases end to end (canonical)
    02_train_gru.py            stepwise GRU training
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
git clone https://github.com/Mioulin/co2-soft-sensor-hybrid-forecasting
pip install -r requirements.txt

# Canonical reproduction command - runs all phases and tests
bash run_all.sh
```

Data must be present at `data/raw_with_label/` and `data/kinetic_prior/`.
Kinetic prior CSVs are from the original repository:
https://github.com/tonyzyl/CO2-Soft-sensor-for-a-carbon-capture-pilot-plant

---

## References

Chai S., Guo S., Mercangoz M. (2026). Hybrid Data-Driven and Mechanistic
CO2 Soft Sensor with MHE-Imputed Labels and Covariance-Weighted Fusion in
a Pilot-Scale Absorber. *Processes*, 14(6), 916.
https://doi.org/10.3390/pr14060916

Zhuang Y. et al. (2022). A hybrid data-driven and mechanistic model soft
sensor for estimating CO2 concentrations for a carbon capture pilot plant.
*Computers in Industry*, 143, 103747.

Rao C. V., Rawlings J. B., Mayne D. Q. (2003). Constrained state
estimation for nonlinear discrete-time systems: Stability and moving
horizon approximations. *IEEE Transactions on Automatic Control*, 48(2),
246-258.
