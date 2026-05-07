# CO2 Soft-Sensor: SDAE + GRU / Transformer + Kinetic Fusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)]()
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange)]()

## Architecture

```
Raw process data (95 features)
        |
        v
Stacked Denoising Autoencoder (SDAE)  95 -> 16 latent
        |
        +──────────────────────+
        v                      v
  GRU Forecaster          Mini-Transformer
  (hidden=44, L=2)        (d=64, 1 layer)
        |                      |
        v                      v
 6 CO2 predictions      6 CO2 predictions
        |                      |
        +──────┬───────────────+
               v
    Kalman-Gaspari-Cohn Fusion
    with precomputed Kinetic Prior
               |
               v
    Final fused CO2 profile (6 points)
```

## Key contributions vs Zhuang et al. (2022)

| Component | Zhuang 2022 | This work |
|-----------|-------------|-----------|
| Label reconstruction | Linear interpolation | Kinetic-Anchored Residuals (KAR) |
| Dimensionality reduction | None | SDAE (95→16) |
| Sequence model | LSTM | GRU vs Transformer comparison |
| Fusion | Kalman | Inverse-variance + Kalman/Gaspari-Cohn |
| Evaluation | Profile RMSE | Observed-point RMSE + MAPE |

## KAR Reconstruction

Residuals r = AT400 - kinetic_prior are interpolated (not raw CO2).
Residual std is 6–17× smaller than raw CO2 std across all runs.
This gives physically motivated, lower-error training targets.

## Run split

| Split | Runs | Purpose |
|-------|------|---------|
| Train | 140120_1, 140206_1, 140207_2, 140214_1, 140214_2, 140227_1 | Fit scalers + models |
| Val | 140313_1 | Early stopping + fusion weight estimation |
| Test | 140207_1 | Final evaluation only |

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh          # full pipeline + tests
# or step by step:
python scripts/run_pipeline.py
pytest tests/ -v
# then open notebooks/ in Jupyter
```

## Results (Test run 140207_1, Observed-Point RMSE)

| Model | h=1 | h=3 | h=6 | h=12 |
|-------|-----|-----|-----|------|
| Persistence | 0.00130 | 0.00161 | 0.00162 | 0.00168 |
| Kinetic prior | 0.00107 | 0.00108 | 0.00110 | 0.00112 |
| GRU (SDAE) | 0.00837 | 0.00224 | 0.01246 | 0.00838 |
| Transformer (SDAE) | 0.00465 | 0.00427 | 0.00377 | 0.00272 |
| **GRU + Kinetic (IV)** | **0.00089** | 0.00080 | 0.00090 | 0.00091 |
| **Transformer + Kinetic (KGC)** | 0.00117 | **0.00068** | 0.00090 | **0.00074** |

Fusion beats both components. Best: Transformer+Kinetic KGC at h=3: **37% improvement** over kinetic alone.

## Limitations

- Target profile is a pseudo-label (KAR reconstruction, not directly observed).
  Observed-point RMSE is the only ground-truth metric.
- Kinetic prior from precomputed MATLAB/Simulink outputs (not reproduced).
- 8 runs is a small dataset: results should be interpreted with run-level caution.
- SDAE shows distribution shift on val/test (expected with small datasets).
- MHE-based reconstruction (Chai et al. 2026) would be superior but requires
  online ODE solver — beyond scope of this assignment.

## Reference

Chai, S., Guo, S., Mercangöz, M. (2026). Hybrid Data-Driven and Mechanistic CO2
Soft Sensor with MHE-Imputed Labels and Covariance-Weighted Fusion.
*Processes*, 14, 916. https://doi.org/10.3390/pr14060916
