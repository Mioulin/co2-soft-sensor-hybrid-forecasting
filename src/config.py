"""Central configuration for CO2 soft-sensor forecasting project."""
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_RAW        = PROJECT_ROOT / "data" / "raw_with_label"
DATA_KINETIC    = PROJECT_ROOT / "data" / "kinetic_prior"
OUT_CHECKPOINTS = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions"
OUT_METRICS     = PROJECT_ROOT / "outputs" / "metrics"
OUT_FIGURES     = PROJECT_ROOT / "outputs" / "figures"
OUT_FUSION      = PROJECT_ROOT / "outputs" / "fusion"

# ── Run split ─────────────────────────────────────────────────────────────────
TRAIN_RUNS = ["140120_1","140206_1","140207_2","140214_1","140214_2","140227_1"]
VAL_RUNS   = ["140313_1"]   # early stopping + fusion weight estimation
TEST_RUNS  = ["140207_1"]   # held-out
ALL_RUNS   = TRAIN_RUNS + VAL_RUNS + TEST_RUNS

# ── Column indices ────────────────────────────────────────────────────────────
COL_AT400 = 3    # CO2 % rotating analyser
COL_LABEL = -1   # active sampling point 1..6
N_SAMPLING_POINTS = 6
AT400_SCALE  = 100.0
KINETIC_HEADER = None   # kinetic CSVs have no header row

# ── Reconstruction ────────────────────────────────────────────────────────────
# Kinetic-anchored residual reconstruction (replaces naive linear interpolation)
# Residuals r = AT400 - kinetic_prior are interpolated (12x smaller variance)
# y_profile = kinetic_prior + interpolated_residuals
RECONSTRUCTION = "kinetic_residual"   # "kinetic_residual" | "linear"

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_LENGTH = 18
HORIZONS      = [1, 3, 6, 12]
USE_LABEL_ONEHOT = True

# ── Scaling ───────────────────────────────────────────────────────────────────
SCALER_TYPE = "standard"

# ── SDAE (Stacked Denoising Autoencoder) ─────────────────────────────────────
SDAE_HIDDEN_DIMS  = [64, 32]   # encoder hidden layers before latent
SDAE_LATENT_DIM   = 16         # bottleneck dimension
SDAE_NOISE_STD    = 0.1        # AWGN noise std during training
SDAE_DROPOUT      = 0.1
SDAE_LR           = 6.785e-4
SDAE_BATCH        = 32
SDAE_EPOCHS       = 150

# ── GRU ───────────────────────────────────────────────────────────────────────
GRU_HIDDEN   = 44
GRU_LAYERS   = 2
GRU_DROPOUT  = 0.1

# ── Mini-Transformer ──────────────────────────────────────────────────────────
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD   = 4
TRANSFORMER_FF_DIM  = 128
TRANSFORMER_LAYERS  = 1
TRANSFORMER_DROPOUT = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 32
MAX_EPOCHS   = 200
LR           = 6.785e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
ES_PATIENCE  = 20
ES_MIN_DELTA = 1e-5
LOSS_ALPHA   = 1.0   # combined_loss = full_mse + alpha * obs_mse

# ── Fusion ────────────────────────────────────────────────────────────────────
FUSION_EPSILON    = 1e-8
KALMAN_REG_EPS    = 1e-6
GC_CORR_LENGTH    = 2.0   # Gaspari-Cohn correlation length (in sampling-point units)
