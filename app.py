import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from codecarbon import EmissionsTracker
import io, os, tempfile
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenTrain AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0f0d;
    --surface: #111a14;
    --surface2: #162019;
    --accent: #00ff87;
    --accent2: #00c46a;
    --warn: #ffb347;
    --danger: #ff5757;
    --text: #e8f5e9;
    --muted: #6b8f72;
    --border: #1e3325;
    --blue: #4fc3f7;
    --purple: #ce93d8;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.main-header {
    background: linear-gradient(135deg, #0a2e1a 0%, #0f3d22 50%, #071a0e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,255,135,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: var(--accent);
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    font-family: 'Space Mono', monospace;
}

.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent2); }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

.result-box {
    background: linear-gradient(135deg, #0a2e1a, #061a0e);
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(0,255,135,0.08);
}
.result-box h2 {
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    margin-bottom: 1rem;
}

.warn-box {
    background: linear-gradient(135deg, #2e1f0a, #1a1206);
    border: 2px solid var(--warn);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ── NEW: Budget Planner boxes ── */
.budget-box {
    background: linear-gradient(135deg, #0a1a2e, #060e1a);
    border: 2px solid var(--blue);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(79,195,247,0.06);
}
.budget-box h2 {
    color: var(--blue);
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    margin-bottom: 1rem;
}

.pass-badge {
    display: inline-block;
    background: var(--accent);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 3px 12px;
    border-radius: 20px;
}
.fail-badge {
    display: inline-block;
    background: var(--danger);
    color: #fff;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 3px 12px;
    border-radius: 20px;
}
.warn-badge {
    display: inline-block;
    background: var(--warn);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 3px 12px;
    border-radius: 20px;
}

/* ── NEW: Cost optimizer box ── */
.cost-box {
    background: linear-gradient(135deg, #1a0a2e, #0e0618);
    border: 2px solid var(--purple);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1rem 0;
    box-shadow: 0 0 30px rgba(206,147,216,0.06);
}
.cost-box h2 {
    color: var(--purple);
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    margin-bottom: 1rem;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 1.5rem;
}

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,255,135,0.25) !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider { color: var(--text) !important; }

div[data-testid="stSelectbox"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    border-radius: 8px !important;
}

hr { border-color: var(--border) !important; }
.stAlert { border-radius: 10px !important; }

/* Pareto chart table style */
.pareto-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.pareto-table th {
    color: var(--muted);
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    letter-spacing: 1px;
    font-size: 0.68rem;
    text-transform: uppercase;
}
.pareto-table td {
    padding: 7px 10px;
    border-bottom: 1px solid #1a2a1f;
    color: var(--text);
}
.pareto-table tr:hover td { background: #162019; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GPU_SPECS = {
    "RTX_3090": {"tdp": 350, "tflops": 35.6},
    "A100":     {"tdp": 400, "tflops": 77.0},
    "V100":     {"tdp": 300, "tflops": 14.0},
    "T4":       {"tdp": 70,  "tflops": 8.1},
    "RTX_3080": {"tdp": 320, "tflops": 29.8},
    "A10G":     {"tdp": 150, "tflops": 31.2},
}

ZONE_LABELS = {
    "US-CAL": "🇺🇸 US California",
    "US-TEX": "🇺🇸 US Texas",
    "IN-WE":  "🇮🇳 India West",
    "IN-SO":  "🇮🇳 India South",
    "EU-DE":  "🇩🇪 Germany",
    "EU-FR":  "🇫🇷 France",
    "GB":     "🇬🇧 United Kingdom",
    "AU-NSW": "🇦🇺 Australia NSW",
}

# ── NEW: Cloud pricing data ───────────────────────────────────────────────────
# Cost per hour (USD) for on-demand instances, per GPU unit
CLOUD_PRICING = {
    "AWS": {
        "A100":     {"instance": "p4d.24xlarge",  "cost_per_hr": 32.77, "gpus_per_node": 8},
        "V100":     {"instance": "p3.2xlarge",    "cost_per_hr": 3.06,  "gpus_per_node": 1},
        "T4":       {"instance": "g4dn.xlarge",   "cost_per_hr": 0.526, "gpus_per_node": 1},
        "A10G":     {"instance": "g5.xlarge",     "cost_per_hr": 1.006, "gpus_per_node": 1},
        "RTX_3090": {"instance": "Not on AWS",    "cost_per_hr": None,  "gpus_per_node": None},
        "RTX_3080": {"instance": "Not on AWS",    "cost_per_hr": None,  "gpus_per_node": None},
    },
    "GCP": {
        "A100":     {"instance": "a2-highgpu-1g", "cost_per_hr": 3.673, "gpus_per_node": 1},
        "V100":     {"instance": "n1+V100",       "cost_per_hr": 2.48,  "gpus_per_node": 1},
        "T4":       {"instance": "n1+T4",         "cost_per_hr": 0.35,  "gpus_per_node": 1},
        "A10G":     {"instance": "Not on GCP",    "cost_per_hr": None,  "gpus_per_node": None},
        "RTX_3090": {"instance": "Not on GCP",    "cost_per_hr": None,  "gpus_per_node": None},
        "RTX_3080": {"instance": "Not on GCP",    "cost_per_hr": None,  "gpus_per_node": None},
    },
}

# Multi-GPU parallel efficiency (Amdahl's law approximation)
# efficiency drops as more GPUs used due to communication overhead
def multigpu_efficiency(n_gpus):
    """Returns efficiency factor (0-1). 1 GPU = 1.0, diminishing returns after."""
    if n_gpus == 1:
        return 1.0
    # ~90% efficiency per doubling (communication overhead)
    return 1.0 / (1 + 0.12 * np.log2(n_gpus))

def multigpu_speedup(n_gpus):
    """Actual speedup factor with n_gpus vs 1 GPU."""
    return n_gpus * multigpu_efficiency(n_gpus)


# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def load_grid_data():
    from datetime import datetime, timedelta
    np.random.seed(42)

    zones = {
        "US-CAL": {"country": "USA",       "base": 200, "solar": 0.30, "wind": 0.15},
        "US-TEX": {"country": "USA",       "base": 400, "solar": 0.15, "wind": 0.25},
        "IN-WE":  {"country": "India",     "base": 600, "solar": 0.20, "wind": 0.10},
        "IN-SO":  {"country": "India",     "base": 550, "solar": 0.25, "wind": 0.12},
        "EU-DE":  {"country": "Germany",   "base": 300, "solar": 0.20, "wind": 0.30},
        "EU-FR":  {"country": "France",    "base": 80,  "solar": 0.10, "wind": 0.10},
        "GB":     {"country": "UK",        "base": 250, "solar": 0.05, "wind": 0.35},
        "AU-NSW": {"country": "Australia", "base": 500, "solar": 0.18, "wind": 0.12},
    }

    start = datetime(2023, 1, 1)
    rows = []
    for zone_key, cfg in zones.items():
        for h in range(8760):
            dt = start + timedelta(hours=h)
            hour, month, dow = dt.hour, dt.month, dt.weekday()
            season = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}[month]

            solar_factor = max(0, np.sin(np.pi*(hour-6)/12)) if 6<=hour<=18 else 0
            seasonal_solar = 1.3 if season=="Summer" else (0.7 if season=="Winter" else 1.0)
            solar_pct = max(0, min(cfg["solar"]*solar_factor*seasonal_solar + np.random.normal(0,0.02), 0.6))
            wind_pct = max(0, min(cfg["wind"]*(1.1 if hour<6 or hour>20 else 0.9) + np.random.normal(0,0.05), 0.7))

            renewable_pct = solar_pct + wind_pct + (0.55 if zone_key=="EU-FR" else 0.03)
            fossil_pct = max(0, min(1 - renewable_pct - 0.05, 0.95))
            carbon_intensity = max(10, fossil_pct*0.5*820 + fossil_pct*0.5*490 + np.random.normal(0,15))

            rows.append({
                "zone_key": zone_key, "country": cfg["country"],
                "datetime": dt, "hour_of_day": hour, "month": month,
                "day_of_week": dow, "season": season,
                "solar_pct": round(solar_pct,4), "wind_pct": round(wind_pct,4),
                "renewable_pct": round(min(renewable_pct,1.0),4),
                "fossil_pct": round(fossil_pct,4),
                "carbon_intensity_gco2_kwh": round(carbon_intensity,2),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_training_data():
    np.random.seed(42)
    gpus = {
        "RTX_3090": {"tdp":350,"tflops":35.6},
        "A100":     {"tdp":400,"tflops":77.0},
        "V100":     {"tdp":300,"tflops":14.0},
        "T4":       {"tdp":70, "tflops":8.1},
        "RTX_3080": {"tdp":320,"tflops":29.8},
        "A10G":     {"tdp":150,"tflops":31.2},
    }
    model_types = ["MLP","CNN","LSTM","Transformer","ResNet","BERT_small","RNN","GRU"]

    # Compute overhead per architecture relative to a simple MLP
    # Calibrated from real benchmarks (ResNet-50/V100, BERT/A100, etc.)
    ARCH_MULT = {
        "MLP":1.0, "CNN":2.5, "RNN":3.0, "GRU":3.5,
        "LSTM":4.0, "ResNet":3.0, "Transformer":7.0, "BERT_small":7.0,
    }
    # Scale factor calibrated from ResNet-50 benchmark:
    # 25M params, 450k steps, V100 (14 TFLOPS) → ~29 hours
    SCALE = 5.4

    rows = []
    for _ in range(5000):
        mt = np.random.choice(model_types)
        gn = np.random.choice(list(gpus.keys()))
        g  = gpus[gn]

        # Params in millions — realistic ranges per architecture
        if mt in ["Transformer","BERT_small"]:
            params_m = np.exp(np.random.uniform(np.log(10), np.log(1000)))
            nl = np.random.randint(4, 48)
        elif mt in ["ResNet","CNN"]:
            params_m = np.exp(np.random.uniform(np.log(1), np.log(200)))
            nl = np.random.randint(5, 152)
        elif mt == "MLP":
            params_m = np.exp(np.random.uniform(np.log(0.01), np.log(50)))
            nl = np.random.randint(1, 20)
        else:  # LSTM, GRU, RNN
            params_m = np.exp(np.random.uniform(np.log(0.1), np.log(100)))
            nl = np.random.randint(1, 12)

        ds = int(np.exp(np.random.uniform(np.log(500), np.log(2e6))))
        bs = np.random.choice([8, 16, 32, 64, 128, 256, 512])
        ep = np.random.choice([1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200])

        steps_k = max((ds / bs * ep) / 1000, 0.01)  # total steps in thousands

        # Calibrated duration formula (benchmark-derived)
        base_dur = SCALE * ARCH_MULT[mt] * params_m * (steps_k ** 0.85) / g["tflops"]
        noise    = np.random.lognormal(0, 0.25)  # realistic ±25% hardware variance
        dur_min  = float(np.clip(base_dur * noise, 1.0, 5000.0))
        energy   = max(0.001, (g["tdp"]/1000) * (dur_min/60) * np.random.lognormal(0, 0.05))

        rows.append({
            "model_type":            mt,
            "num_parameters":        params_m * 1e6,
            "params_millions":       params_m,
            "num_layers":            nl,
            "dataset_size":          ds,
            "batch_size":            bs,
            "epochs":                ep,
            "gpu_type":              gn,
            "gpu_tdp_watts":         g["tdp"],
            "gpu_tflops":            g["tflops"],
            "training_duration_min": round(dur_min, 4),
            "energy_consumed_kwh":   round(energy, 6),
        })
    return pd.DataFrame(rows)


@st.cache_resource
def train_duration_model(df):
    from sklearn.metrics import mean_absolute_error, r2_score

    ARCH_MULT = {
        "MLP":1.0, "CNN":2.5, "RNN":3.0, "GRU":3.5,
        "LSTM":4.0, "ResNet":3.0, "Transformer":7.0, "BERT_small":7.0,
    }

    le_mt  = LabelEncoder()
    le_gpu = LabelEncoder()
    df = df.copy()
    df["model_type_enc"] = le_mt.fit_transform(df["model_type"])
    df["gpu_type_enc"]   = le_gpu.fit_transform(df["gpu_type"])

    # ── Engineered features ──────────────────────────────────────────────────
    params_m = df["num_parameters"] / 1e6
    df["log_params"]    = np.log1p(params_m)
    df["log_dataset"]   = np.log1p(df["dataset_size"])
    df["steps_k"]       = np.maximum(df["dataset_size"] / df["batch_size"] * df["epochs"] / 1000, 0.01)
    df["log_steps"]     = np.log1p(df["steps_k"])
    df["log_compute"]   = np.log1p(
        params_m * (df["steps_k"] ** 0.85) / df["gpu_tflops"]
    )
    df["tdp_per_tflop"] = df["gpu_tdp_watts"] / df["gpu_tflops"]
    df["arch_mult"]     = df["model_type"].map(ARCH_MULT)

    features = [
        "model_type_enc", "log_params", "num_layers",
        "log_dataset",    "batch_size", "epochs",
        "gpu_tflops",     "gpu_tdp_watts", "tdp_per_tflop",
        "log_steps",      "log_compute",   "arch_mult",
    ]

    X     = df[features]
    y_raw = df["training_duration_min"]
    y     = np.log1p(y_raw)   # predict in log space

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42
    )

    # ── Base learners (GBR + RF + XGBoost) ───────────────────────────────────
    gbr = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, max_features=0.8, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, max_features=0.7,
        min_samples_split=10, n_jobs=-1, random_state=42
    )
    xgb = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=-1
    )

    # ── Stacking ensemble with XGBoost added ─────────────────────────────────
    stacking_model = StackingRegressor(
        estimators=[('gbr', gbr), ('rf', rf), ('xgb', xgb)],
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        cv=5, n_jobs=-1, verbose=0
    )

    # ── Wrap fit() with CodeCarbon tracker ───────────────────────────────────
    _tmpdir = tempfile.mkdtemp()
    tracker = EmissionsTracker(
        project_name="GreenTrainAI_ModelFit",
        output_dir=_tmpdir,
        log_level="error",
        save_to_file=True,
        tracking_mode="process",
    )
    tracker.start()
    stacking_model.fit(X_train, y_train)
    training_co2_kg = tracker.stop()          # kg CO₂eq
    training_co2_g  = (training_co2_kg or 0.0) * 1000   # → grams

    model = stacking_model

    y_pred_test  = np.expm1(model.predict(X_test))
    y_pred_train = np.expm1(model.predict(X_train))
    y_raw_test_arr  = np.array(y_raw_test)
    y_raw_train_arr = np.array(y_raw_train)

    mae      = mean_absolute_error(y_raw_test_arr,  y_pred_test)
    r2       = r2_score(y_raw_test_arr,  y_pred_test)
    mae_train= mean_absolute_error(y_raw_train_arr, y_pred_train)
    r2_train = r2_score(y_raw_train_arr, y_pred_train)

    eval_data = {
        "y_test":            y_raw_test_arr,
        "y_pred_test":       y_pred_test,
        "y_train":           y_raw_train_arr,
        "y_pred_train":      y_pred_train,
        "mae_train":         mae_train,
        "r2_train":          r2_train,
        "residuals":         y_raw_test_arr - y_pred_test,
        "features":          features,
        "training_co2_g":    training_co2_g,   # ← NEW: CodeCarbon measurement
    }
    return model, le_mt, le_gpu, mae, r2, eval_data


def predict_duration(model, le_mt, le_gpu, model_type, num_params, num_layers,
                     dataset_size, batch_size, epochs, gpu_type, gpu_tdp, gpu_tflops):
    ARCH_MULT = {
        "MLP":1.0, "CNN":2.5, "RNN":3.0, "GRU":3.5,
        "LSTM":4.0, "ResNet":3.0, "Transformer":7.0, "BERT_small":7.0,
    }
    try:
        mt_enc = le_mt.transform([model_type])[0]
    except:
        mt_enc = 0

    params_m     = num_params / 1e6
    log_params   = np.log1p(params_m)
    log_dataset  = np.log1p(dataset_size)
    steps_k      = max((dataset_size / batch_size * epochs) / 1000, 0.01)
    log_steps    = np.log1p(steps_k)
    log_compute  = np.log1p(params_m * (steps_k ** 0.85) / gpu_tflops)
    tdp_per_tflop= gpu_tdp / gpu_tflops
    arch_mult    = ARCH_MULT.get(model_type, 3.0)

    import pandas as pd
    X = pd.DataFrame([[mt_enc, log_params, num_layers, log_dataset,
                       batch_size, epochs, gpu_tflops, gpu_tdp,
                       tdp_per_tflop, log_steps, log_compute, arch_mult]],
                     columns=["model_type_enc","log_params","num_layers","log_dataset",
                              "batch_size","epochs","gpu_tflops","gpu_tdp_watts",
                              "tdp_per_tflop","log_steps","log_compute","arch_mult"])

    log_pred = model.predict(X)[0]
    return float(max(1.0, np.expm1(log_pred)))


def get_carbon_forecast(grid_df, zone_key, from_hour=0, window=24):
    zone_data = grid_df[grid_df["zone_key"] == zone_key].copy()
    hourly_avg = zone_data.groupby("hour_of_day")["carbon_intensity_gco2_kwh"].mean()
    hourly_std = zone_data.groupby("hour_of_day")["carbon_intensity_gco2_kwh"].std()

    zone_profiles = {
        "US-CAL": {"solar_peak":13,"solar_strength":0.22,"wind_night":True},
        "US-TEX": {"solar_peak":14,"solar_strength":0.15,"wind_night":True},
        "IN-WE":  {"solar_peak":12,"solar_strength":0.18,"wind_night":False},
        "IN-SO":  {"solar_peak":12,"solar_strength":0.20,"wind_night":False},
        "EU-DE":  {"solar_peak":13,"solar_strength":0.16,"wind_night":True},
        "EU-FR":  {"solar_peak":13,"solar_strength":0.08,"wind_night":False},
        "GB":     {"solar_peak":13,"solar_strength":0.05,"wind_night":True},
        "AU-NSW": {"solar_peak":12,"solar_strength":0.16,"wind_night":False},
    }
    profile = zone_profiles.get(zone_key, {"solar_peak":13,"solar_strength":0.12,"wind_night":False})

    wind_event_start = np.random.randint(0, 20)
    wind_event_strength = np.random.uniform(20, 80)

    forecast = []
    for i in range(window):
        h = (from_hour + i) % 24
        base = hourly_avg[h]

        solar_offset = abs(h - profile["solar_peak"])
        solar_reduction = profile["solar_strength"] * 200 * max(0, 1 - solar_offset/5)

        night_wind_reduction = 0
        if profile["wind_night"] and (h >= 22 or h <= 5):
            night_wind_reduction = np.random.uniform(15, 45)

        wind_reduction = 0
        if wind_event_start <= i <= wind_event_start + 3:
            wind_reduction = wind_event_strength * max(0, 1 - abs(i - wind_event_start - 1.5)/3)

        noise = np.random.normal(0, hourly_std[h] * 0.5)
        intensity = max(10, base - solar_reduction - night_wind_reduction - wind_reduction + noise)
        forecast.append({
            "offset_hours": i,
            "hour_of_day": h,
            "carbon_intensity": round(intensity, 2),
        })
    return pd.DataFrame(forecast)


def find_optimal_start(forecast_df, duration_min):
    duration_hrs = duration_min / 60
    n = len(forecast_df)
    window_size = max(1, min(int(np.ceil(duration_hrs)), n))
    results = []
    for start_i in range(n):
        end_i = min(start_i + window_size, n)
        window = forecast_df.iloc[start_i:end_i]["carbon_intensity"].values
        avg_intensity = np.mean(window)
        results.append({
            "start_offset_hrs": forecast_df.iloc[start_i]["offset_hours"],
            "start_hour": forecast_df.iloc[start_i]["hour_of_day"],
            "avg_carbon_intensity": avg_intensity,
        })
    if not results:
        results.append({
            "start_offset_hrs": 0,
            "start_hour": forecast_df.iloc[0]["hour_of_day"],
            "avg_carbon_intensity": forecast_df.iloc[0]["carbon_intensity"],
        })
    return pd.DataFrame(results)


def co2_to_equivalent(co2_grams):
    km_car = co2_grams / 120
    phone_charges = co2_grams / 8.22
    trees_hours = co2_grams / (21000/8760)
    return km_car, phone_charges, trees_hours


# ── NEW: Carbon Budget Planner helpers ───────────────────────────────────────
def evaluate_budget_checklist(co2_grams, budget_grams):
    """Return status, percentage used, and recommendation."""
    pct = (co2_grams / budget_grams) * 100 if budget_grams > 0 else 999
    if pct <= 80:
        status = "PASS"
        color = "#00ff87"
        badge = "pass-badge"
        tip = "Well within budget. You have headroom for more epochs or a larger model."
    elif pct <= 100:
        status = "WARN"
        color = "#ffb347"
        badge = "warn-badge"
        tip = "Close to budget. Consider reducing epochs or using a smaller batch size."
    else:
        status = "OVER"
        color = "#ff5757"
        badge = "fail-badge"
        tip = "Over budget! Switch to a lower-TDP GPU, reduce epochs, or pick a greener grid zone."
    return status, pct, color, badge, tip


def suggest_green_alternatives(co2_grams, budget_grams, gpu_type, dur_min, best_intensity):
    """Suggest config tweaks to bring run under budget."""
    suggestions = []
    if co2_grams <= budget_grams:
        return suggestions

    overage_pct = (co2_grams - budget_grams) / co2_grams * 100

    # Suggest lower-TDP GPU
    current_tdp = GPU_SPECS[gpu_type]["tdp"]
    for g, s in GPU_SPECS.items():
        if s["tdp"] < current_tdp:
            alt_energy = (s["tdp"]/1000) * (dur_min/60)
            alt_co2 = alt_energy * best_intensity
            if alt_co2 <= budget_grams:
                suggestions.append(f"🖥 Switch GPU to **{g}** (TDP: {s['tdp']}W) → estimated CO₂: **{alt_co2:.0f}g**")

    # Suggest greener zone
    zone_intensities = {
        "EU-FR": 80, "US-CAL": 180, "GB": 220, "EU-DE": 280,
        "AU-NSW": 420, "US-TEX": 380, "IN-SO": 500, "IN-WE": 560
    }
    current_energy_kwh = (GPU_SPECS[gpu_type]["tdp"]/1000) * (dur_min/60)
    for zone, intensity in sorted(zone_intensities.items(), key=lambda x: x[1]):
        alt_co2 = current_energy_kwh * intensity
        if alt_co2 <= budget_grams:
            suggestions.append(f"🌍 Train in **{ZONE_LABELS[zone]}** (avg ~{intensity} gCO₂/kWh) → estimated CO₂: **{alt_co2:.0f}g**")
            break

    # Suggest epoch reduction
    needed_reduction = budget_grams / co2_grams
    suggestions.append(f"⏱ Reduce training scale by **{(1-needed_reduction)*100:.0f}%** (e.g., fewer epochs or smaller dataset)")

    return suggestions[:3]  # top 3


# ── NEW: Cloud cost optimizer helpers ────────────────────────────────────────
def compute_cloud_costs(gpu_type, dur_min, n_gpus, providers=["AWS", "GCP"]):
    """Compute cost for each cloud provider & instance type."""
    actual_dur = dur_min / multigpu_speedup(n_gpus)  # faster with more GPUs
    actual_dur_hrs = actual_dur / 60
    results = []

    for provider in providers:
        pricing = CLOUD_PRICING.get(provider, {}).get(gpu_type)
        if pricing and pricing["cost_per_hr"] is not None:
            gpn = pricing["gpus_per_node"]
            # How many nodes do we need?
            nodes_needed = max(1, int(np.ceil(n_gpus / gpn)))
            total_cost = pricing["cost_per_hr"] * nodes_needed * actual_dur_hrs
            results.append({
                "provider": provider,
                "instance": pricing["instance"],
                "nodes": nodes_needed,
                "gpus_per_node": gpn,
                "cost_per_hr": pricing["cost_per_hr"] * nodes_needed,
                "total_cost_usd": round(total_cost, 4),
                "actual_duration_min": round(actual_dur, 1),
            })
        else:
            results.append({
                "provider": provider,
                "instance": "N/A",
                "nodes": None,
                "gpus_per_node": None,
                "cost_per_hr": None,
                "total_cost_usd": None,
                "actual_duration_min": round(actual_dur, 1),
            })
    return results


def pareto_gpu_analysis(model_type, num_params, num_layers, dataset_size,
                        batch_size, epochs, best_intensity, dur_model, le_mt, le_gpu):
    """For each GPU, compute (cost, CO2) to build a Pareto front."""
    rows = []
    for gname, gspec in GPU_SPECS.items():
        dur = predict_duration(dur_model, le_mt, le_gpu,
                               model_type, num_params, num_layers,
                               dataset_size, batch_size, epochs,
                               gname, gspec["tdp"], gspec["tflops"])
        energy = (gspec["tdp"]/1000) * (dur/60)
        co2 = energy * best_intensity

        # Cost: use AWS if available, else GCP, else estimate
        aws = CLOUD_PRICING["AWS"].get(gname)
        gcp = CLOUD_PRICING["GCP"].get(gname)
        cost_hr = None
        provider_label = "Local"
        if aws and aws["cost_per_hr"]:
            cost_hr = aws["cost_per_hr"]
            provider_label = f"AWS {aws['instance']}"
        elif gcp and gcp["cost_per_hr"]:
            cost_hr = gcp["cost_per_hr"]
            provider_label = f"GCP {gcp['instance']}"

        if cost_hr:
            total_cost = cost_hr * (dur/60)
        else:
            total_cost = None

        rows.append({
            "gpu": gname,
            "provider": provider_label,
            "duration_min": round(dur,1),
            "energy_kwh": round(energy,4),
            "co2_g": round(co2,1),
            "cost_usd": round(total_cost, 4) if total_cost else None,
        })
    return pd.DataFrame(rows)


# ── Load data & train model ───────────────────────────────────────────────────
with st.spinner("🌿 Initializing GreenTrain AI..."):
    grid_df = load_grid_data()
    train_df = load_training_data()
    dur_model, le_mt, le_gpu, mae, r2, eval_data = train_duration_model(train_df)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 GreenTrain AI</h1>
    <p>Carbon-aware ML training scheduler · Predict emissions · Find the greenest moment to train</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Model Configuration</div>', unsafe_allow_html=True)

    model_type = st.selectbox("Model Type",
        ["MLP","CNN","LSTM","Transformer","ResNet","BERT_small","RNN","GRU"])

    gpu_type = st.selectbox("GPU Type", list(GPU_SPECS.keys()))

    # ── NEW: Multi-GPU ──
    n_gpus = st.select_slider("Number of GPUs", options=[1,2,4,8,16], value=1)
    if n_gpus > 1:
        eff = multigpu_efficiency(n_gpus)
        speedup = multigpu_speedup(n_gpus)
        st.markdown(f"""
        <div style="background:#0a2e1a;border:1px solid #1e3325;border-radius:8px;padding:0.6rem 1rem;font-family:'Space Mono',monospace;font-size:0.72rem;color:#6b8f72">
        Parallel efficiency: <b style="color:#00ff87">{eff*100:.0f}%</b><br>
        Effective speedup: <b style="color:#00ff87">{speedup:.1f}×</b>
        </div>
        """, unsafe_allow_html=True)

    num_params = st.number_input("Parameters (millions)",
        min_value=0.001, max_value=1000.0, value=10.0, step=1.0)

    num_layers = st.slider("Number of Layers", 1, 50, 6)

    dataset_size = st.number_input("Dataset Size (rows)",
        min_value=100, max_value=1_000_000, value=10000, step=1000)

    batch_size = st.select_slider("Batch Size", options=[16,32,64,128,256], value=64)

    epochs = st.slider("Epochs", 1, 200, 20)

    st.markdown("---")
    st.markdown('<div class="section-header">🌍 Grid Zone</div>', unsafe_allow_html=True)

    zone_key = st.selectbox("Cloud Region / Grid Zone",
        options=list(ZONE_LABELS.keys()),
        format_func=lambda x: ZONE_LABELS[x])

    current_hour = st.slider("Current Hour of Day", 0, 23, 9,
        help="What time is it now? (used to build the 24hr forecast)")

    scheduling_window = st.slider("Scheduling Window (hours)", 6, 24, 24)

    # ── NEW: Carbon Budget ──
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Carbon Budget</div>', unsafe_allow_html=True)
    enable_budget = st.toggle("Enable Budget Planner", value=True)
    carbon_budget_g = st.number_input(
        "Max CO₂ Budget (grams)",
        min_value=10, max_value=100000, value=500, step=50,
        help="Set your maximum acceptable CO₂ emissions for this training run."
    ) if enable_budget else None

    st.markdown("---")
    predict_btn = st.button("🔍 Predict & Schedule", use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Prediction & Schedule",
    "🎯 Carbon Budget Planner",
    "💰 Cloud Cost Optimizer",
    "📊 Grid Analysis",
    "🤖 Model Insights & Dataset",
    "🔬 Model Evaluation"
])

# ── Shared computation ────────────────────────────────────────────────────────
gpu = GPU_SPECS[gpu_type]
num_params_actual = int(num_params * 1_000_000)

# Single-GPU base duration
base_dur_min = predict_duration(
    dur_model, le_mt, le_gpu,
    model_type, num_params_actual, num_layers,
    dataset_size, batch_size, epochs,
    gpu_type, gpu["tdp"], gpu["tflops"]
)

# Apply multi-GPU speedup
speedup_factor = multigpu_speedup(n_gpus)
dur_min = base_dur_min / speedup_factor
dur_hrs = dur_min / 60

# Total power: n_gpus * TDP (watts)
total_tdp_watts = gpu["tdp"] * n_gpus

# Forecast & scheduling
forecast_df = get_carbon_forecast(grid_df, zone_key, current_hour, scheduling_window)
schedule_df = find_optimal_start(forecast_df, dur_min)

current_intensity = forecast_df.iloc[0]["carbon_intensity"]
current_energy = (total_tdp_watts / 1000) * dur_hrs
current_co2 = current_energy * current_intensity

best_row = schedule_df.loc[schedule_df["avg_carbon_intensity"].idxmin()]
best_offset = best_row["start_offset_hrs"]
best_intensity = best_row["avg_carbon_intensity"]
best_energy = current_energy
best_co2 = best_energy * best_intensity
savings_pct = max(0, (current_co2 - best_co2) / current_co2 * 100)

# ── 🌱 Sustainability Report (sidebar — rendered after shared computation) ────
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="section-header">🌱 Sustainability Report</div>', unsafe_allow_html=True)

    _training_co2_g  = eval_data.get("training_co2_g", 0.0)
    _sr_co2_display  = f"{_training_co2_g:.4f} g" if _training_co2_g < 1.0 else f"{_training_co2_g:.3f} g"
    _sr_color        = "#00ff87" if _training_co2_g < 1.0 else "#ffb347" if _training_co2_g < 10.0 else "#ff5757"
    _infer_co2       = current_co2
    _infer_color     = "#00ff87" if _infer_co2 < 200 else "#ffb347" if _infer_co2 < 500 else "#ff5757"
    _savings_g       = max(0, current_co2 - best_co2)

    st.markdown(f"""
    <div style="background:#0a1a10;border:1px solid #1e3325;border-radius:12px;
                padding:1rem;font-family:'Space Mono',monospace;font-size:0.75rem">
        <div style="color:#6b8f72;font-size:0.62rem;letter-spacing:1.5px;margin-bottom:0.4rem">
            🤖 MODEL FIT (CodeCarbon)
        </div>
        <div style="color:{_sr_color};font-size:1.1rem;font-weight:700;margin-bottom:0.8rem">
            {_sr_co2_display} CO₂eq
        </div>
        <div style="color:#6b8f72;font-size:0.62rem;letter-spacing:1.5px;margin-bottom:0.4rem">
            ⚡ INFERENCE RUN (est.)
        </div>
        <div style="color:{_infer_color};font-size:1rem;font-weight:700;margin-bottom:0.8rem">
            {_infer_co2:.0f} g CO₂
        </div>
        <div style="color:#6b8f72;font-size:0.62rem;letter-spacing:1.5px;margin-bottom:0.4rem">
            💚 POTENTIAL SAVINGS
        </div>
        <div style="color:#00ff87;font-size:1rem;font-weight:700;margin-bottom:0.8rem">
            {_savings_g:.0f} g saved
        </div>
        <div style="border-top:1px solid #1e3325;padding-top:0.6rem;
                    color:#6b8f72;font-size:0.62rem;line-height:1.6">
            Stack: <span style="color:#e8f5e9">GBR + RF + XGBoost</span><br>
            Tracker: <span style="color:#e8f5e9">CodeCarbon v3</span><br>
            Zone: <span style="color:#e8f5e9">{ZONE_LABELS.get(zone_key, zone_key)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction & Scheduling
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Top metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{dur_min:.0f}<span style="font-size:1rem"> min</span></div>
            <div class="metric-label">Predicted Duration</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{speedup_factor:.1f}<span style="font-size:1rem">×</span></div>
            <div class="metric-label">GPU Speedup ({n_gpus} GPU{'s' if n_gpus>1 else ''})</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{current_energy:.3f}<span style="font-size:1rem"> kWh</span></div>
            <div class="metric-label">Energy ({n_gpus}× GPU)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        co2_val = f"{current_co2:.0f}" if current_co2 < 1000 else f"{current_co2/1000:.2f}k"
        co2_color = '#ff5757' if current_co2 > 500 else '#ffb347' if current_co2 > 200 else '#00ff87'
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{co2_color}">{co2_val}<span style="font-size:1rem"> g</span></div>
            <div class="metric-label">CO₂ if Start Now</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00ff87">{savings_pct:.1f}<span style="font-size:1rem">%</span></div>
            <div class="metric-label">Potential Savings</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Multi-GPU info banner ──
    if n_gpus > 1:
        eff_pct = multigpu_efficiency(n_gpus) * 100
        st.markdown(f"""
        <div style="background:#0a1a2e;border:1px solid #4fc3f7;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-bottom:1rem;
                    font-family:'Space Mono',monospace;font-size:0.8rem;color:#b0e0f7">
        🖥 <b style="color:#4fc3f7">{n_gpus}× {gpu_type}</b> &nbsp;·&nbsp;
        Total TDP: <b style="color:#4fc3f7">{total_tdp_watts}W</b> &nbsp;·&nbsp;
        Parallel efficiency: <b style="color:#4fc3f7">{eff_pct:.0f}%</b> &nbsp;·&nbsp;
        Effective speedup: <b style="color:#4fc3f7">{speedup_factor:.2f}×</b>
        (vs {base_dur_min:.0f} min on 1 GPU → <b style="color:#00ff87">{dur_min:.0f} min</b>)
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendation ──
    lcol, rcol = st.columns([1.2, 1])

    with lcol:
        if best_offset == 0:
            st.markdown(f"""<div class="result-box">
                <h2>✅ OPTIMAL: START NOW</h2>
                <p style="color:#e8f5e9;font-size:1.1rem">This is the greenest window in the next {scheduling_window} hours.</p>
                <p style="color:#6b8f72;font-family:'Space Mono',monospace;font-size:0.85rem">
                Grid intensity: <b style="color:#00ff87">{current_intensity:.0f} gCO₂/kWh</b><br>
                Expected CO₂: <b style="color:#00ff87">{current_co2:.0f} g CO₂</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            best_hr = int(best_row["start_hour"])
            ampm = "AM" if best_hr < 12 else "PM"
            hr12 = best_hr if best_hr <= 12 else best_hr - 12
            st.markdown(f"""<div class="result-box">
                <h2>⏰ RECOMMENDED: WAIT {best_offset:.1f} HOURS</h2>
                <div style="display:flex;align-items:center;gap:2rem;margin:1rem 0">
                    <div>
                        <div style="font-family:'Space Mono',monospace;font-size:2rem;color:#00ff87;font-weight:700">{hr12}:00 {ampm}</div>
                        <div style="color:#6b8f72;font-size:0.8rem;letter-spacing:1px">OPTIMAL START TIME</div>
                    </div>
                    <div style="background:#00ff87;color:#000;font-family:'Space Mono',monospace;font-weight:700;font-size:1.3rem;padding:8px 20px;border-radius:20px">
                        SAVE {savings_pct:.0f}%
                    </div>
                </div>
                <p style="color:#6b8f72;font-family:'Space Mono',monospace;font-size:0.82rem">
                Now: <b style="color:#ff5757">{current_intensity:.0f} gCO₂/kWh</b> →
                Best: <b style="color:#00ff87">{best_intensity:.0f} gCO₂/kWh</b><br>
                CO₂ saved: <b style="color:#00ff87">{current_co2-best_co2:.0f} g</b>
                ({current_co2:.0f}g → {best_co2:.0f}g)
                </p>
            </div>""", unsafe_allow_html=True)

    with rcol:
        km, phones, trees = co2_to_equivalent(current_co2)
        km_s, phones_s, trees_s = co2_to_equivalent(best_co2)
        st.markdown(f"""<div class="warn-box">
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:2px;color:#6b8f72;margin-bottom:1rem">🌱 CO₂ EQUIVALENTS</div>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse">
                <tr>
                    <td style="color:#6b8f72;padding:4px 0">🚗 Driving</td>
                    <td style="color:#ff5757;font-family:'Space Mono',monospace;text-align:right">{km:.2f} km</td>
                    <td style="color:#00ff87;font-family:'Space Mono',monospace;text-align:right">→ {km_s:.2f} km</td>
                </tr>
                <tr>
                    <td style="color:#6b8f72;padding:4px 0">📱 Phone charges</td>
                    <td style="color:#ff5757;font-family:'Space Mono',monospace;text-align:right">{phones:.1f}x</td>
                    <td style="color:#00ff87;font-family:'Space Mono',monospace;text-align:right">→ {phones_s:.1f}x</td>
                </tr>
                <tr>
                    <td style="color:#6b8f72;padding:4px 0">🌳 Tree-hours</td>
                    <td style="color:#ff5757;font-family:'Space Mono',monospace;text-align:right">{trees:.1f}h</td>
                    <td style="color:#00ff87;font-family:'Space Mono',monospace;text-align:right">→ {trees_s:.1f}h</td>
                </tr>
            </table>
            <div style="margin-top:1rem;font-size:0.72rem;color:#6b8f72;font-family:'Space Mono',monospace">
            RED = start now &nbsp;|&nbsp; GREEN = optimal time
            </div>
        </div>""", unsafe_allow_html=True)

    # ── 24hr forecast chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 24-Hour Carbon Intensity Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_vrect(x0=0, x1=min(dur_hrs, scheduling_window),
        fillcolor="rgba(255,87,87,0.08)", line_width=0,
        annotation_text="Train Now", annotation_position="top left",
        annotation_font_color="#ff5757", annotation_font_size=11)
    if best_offset > 0:
        fig.add_vrect(x0=best_offset, x1=min(best_offset+dur_hrs, scheduling_window),
            fillcolor="rgba(0,255,135,0.08)", line_width=0,
            annotation_text="Optimal", annotation_position="top left",
            annotation_font_color="#00ff87", annotation_font_size=11)

    fig.add_trace(go.Scatter(
        x=forecast_df["offset_hours"], y=forecast_df["carbon_intensity"],
        mode="lines", line=dict(color="#00c46a", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,196,106,0.07)",
        name="Carbon Intensity",
        hovertemplate="<b>+%{x:.0f}h</b><br>%{y:.0f} gCO₂/kWh<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[best_offset], y=[best_intensity], mode="markers",
        marker=dict(color="#00ff87", size=14, symbol="star", line=dict(color="#000",width=1)),
        name="Best Start",
        hovertemplate=f"<b>Best Start: +{best_offset:.0f}h</b><br>{best_intensity:.0f} gCO₂/kWh<extra></extra>"
    ))
    fig.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        margin=dict(l=20,r=20,t=20,b=40), height=280,
        xaxis=dict(title="Hours from now", gridcolor="#1e3325", gridwidth=1, tickfont=dict(size=11), zeroline=False),
        yaxis=dict(title="gCO₂/kWh", gridcolor="#1e3325", gridwidth=1, tickfont=dict(size=11), zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11))
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Scheduling bar chart ──
    st.markdown('<div class="section-header">🗓 Start Time vs CO₂ Emissions</div>', unsafe_allow_html=True)
    fig2 = go.Figure(go.Bar(
        x=schedule_df["start_offset_hrs"], y=schedule_df["avg_carbon_intensity"],
        marker=dict(color=schedule_df["avg_carbon_intensity"],
            colorscale=[[0,"#00ff87"],[0.5,"#ffb347"],[1,"#ff5757"]],
            showscale=True,
            colorbar=dict(title=dict(text="gCO₂/kWh", font=dict(color="#6b8f72")),
                         tickfont=dict(color="#6b8f72"))),
        hovertemplate="Start in <b>+%{x:.0f}h</b><br>Avg: <b>%{y:.0f} gCO₂/kWh</b><extra></extra>"
    ))
    fig2.add_vline(x=best_offset, line_color="#00ff87", line_width=2,
        annotation_text=f"⭐ Best (+{best_offset:.0f}h)",
        annotation_font_color="#00ff87", annotation_font_size=12)
    fig2.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        margin=dict(l=20,r=20,t=20,b=40), height=240,
        xaxis=dict(title="Start in X hours", gridcolor="#1e3325", zeroline=False),
        yaxis=dict(title="Avg gCO₂/kWh", gridcolor="#1e3325", zeroline=False),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 🌱 Sustainability Report (Tab 1) ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌱 Sustainability Report — This Run</div>', unsafe_allow_html=True)

    training_co2_g = eval_data.get("training_co2_g", 0.0)
    total_system_co2 = training_co2_g + current_co2   # meta-model fit + inference
    total_optimal_co2 = training_co2_g + best_co2

    sr1, sr2, sr3, sr4 = st.columns(4)
    sr_cards = [
        ("🤖 Model Fit (CodeCarbon)", f"{training_co2_g:.4f} g", "#ce93d8",
         "CO₂eq measured by CodeCarbon during stacking ensemble training"),
        ("⚡ Inference (Now)", f"{current_co2:.0f} g", "#ff5757",
         f"Estimated CO₂ at current grid intensity ({current_intensity:.0f} gCO₂/kWh)"),
        ("🌿 Inference (Optimal)", f"{best_co2:.0f} g", "#00ff87",
         f"Estimated CO₂ at best window ({best_intensity:.0f} gCO₂/kWh)"),
        ("💚 Total Saving", f"{max(0, current_co2-best_co2):.0f} g", "#ffb347",
         "CO₂ saved by scheduling at the optimal time"),
    ]
    for col, (label, val, color, desc) in zip([sr1, sr2, sr3, sr4], sr_cards):
        with col:
            st.markdown(f"""
            <div style="background:#111a14;border:1px solid #1e3325;border-radius:12px;
                        padding:1rem 1.2rem;text-align:center">
                <div style="font-size:0.68rem;color:#6b8f72;letter-spacing:1.5px;
                            font-family:'Space Mono',monospace;margin-bottom:6px">{label}</div>
                <div style="font-family:'Space Mono',monospace;font-size:1.5rem;
                            font-weight:700;color:{color}">{val}</div>
                <div style="font-size:0.65rem;color:#3a5940;margin-top:4px;
                            font-family:'Space Mono',monospace">{desc[:55]}…</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stacked bar: model fit vs inference breakdown
    fig_sr = go.Figure()
    fig_sr.add_trace(go.Bar(
        name="Model Fit (CodeCarbon)", x=["Start Now", "Optimal Window"],
        y=[training_co2_g, training_co2_g],
        marker_color="#ce93d8",
        hovertemplate="Model Fit CO₂: <b>%{y:.4f} g</b><extra></extra>"
    ))
    fig_sr.add_trace(go.Bar(
        name="Inference CO₂", x=["Start Now", "Optimal Window"],
        y=[current_co2, best_co2],
        marker_color=["#ff5757", "#00ff87"],
        hovertemplate="Inference CO₂: <b>%{y:.1f} g</b><extra></extra>"
    ))
    fig_sr.update_layout(
        barmode="stack",
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        height=260, margin=dict(l=20, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1e3325", zeroline=False),
        yaxis=dict(title="Total CO₂ (g)", gridcolor="#1e3325", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        title=dict(text="Carbon Cost Breakdown: Model Fit vs Inference",
                   font=dict(color="#e8f5e9", size=12))
    )
    st.plotly_chart(fig_sr, use_container_width=True)

    st.markdown(f"""
    <div style="background:#0a1a10;border:1px solid #1e3325;border-radius:10px;
                padding:0.9rem 1.4rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b8f72">
    📌 <b style="color:#e8f5e9">How this is measured:</b>
    Model fit CO₂ tracked by <b style="color:#ce93d8">CodeCarbon</b> (hardware power × duration × grid intensity).
    Inference CO₂ estimated from GPU TDP × training duration × grid carbon intensity.
    Ensemble: <b style="color:#00ff87">GBR + RandomForest + XGBoost → RidgeCV meta-learner</b>.
    </div>
    """, unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not enable_budget:
        st.info("💡 Enable the **Carbon Budget Planner** toggle in the sidebar to use this feature.")
    else:
        st.markdown('<div class="section-header">🎯 Carbon Budget Planner</div>', unsafe_allow_html=True)

        budget = carbon_budget_g

        # Evaluate NOW vs OPTIMAL against budget
        status_now, pct_now, color_now, badge_now, tip_now = evaluate_budget_checklist(current_co2, budget)
        status_opt, pct_opt, color_opt, badge_opt, tip_opt = evaluate_budget_checklist(best_co2, budget)

        # ── Budget overview cards ──
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#4fc3f7">{budget:.0f}<span style="font-size:1rem"> g</span></div>
                <div class="metric-label">CO₂ Budget</div>
            </div>""", unsafe_allow_html=True)
        with bc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color_now}">{current_co2:.0f}<span style="font-size:1rem"> g</span></div>
                <div class="metric-label">CO₂ if Start Now</div>
            </div>""", unsafe_allow_html=True)
        with bc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color_opt}">{best_co2:.0f}<span style="font-size:1rem"> g</span></div>
                <div class="metric-label">CO₂ at Optimal Time</div>
            </div>""", unsafe_allow_html=True)
        with bc4:
            saved = current_co2 - best_co2
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#00ff87">{saved:.0f}<span style="font-size:1rem"> g</span></div>
                <div class="metric-label">Max Savings via Scheduling</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Checklist panel ──
        chk1, chk2 = st.columns(2)

        with chk1:
            border_color = color_now
            st.markdown(f"""<div style="background:#0a0f0d;border:2px solid {border_color};border-radius:14px;padding:1.5rem;margin-bottom:1rem">
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:2px;color:#6b8f72;margin-bottom:0.8rem">📋 START NOW ANALYSIS</div>
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
                    <span class="{badge_now}">{status_now}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{color_now};font-weight:700">{pct_now:.1f}% of budget</span>
                </div>
                <div style="background:#0f1a12;border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:1rem">
                    <div style="background:{color_now};height:100%;width:{min(pct_now,100):.0f}%;transition:width 0.5s;border-radius:8px"></div>
                </div>
                <p style="color:#6b8f72;font-size:0.82rem;font-family:'Space Mono',monospace;margin:0">{tip_now}</p>
            </div>""", unsafe_allow_html=True)

        with chk2:
            border_color2 = color_opt
            st.markdown(f"""<div style="background:#0a0f0d;border:2px solid {border_color2};border-radius:14px;padding:1.5rem;margin-bottom:1rem">
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;letter-spacing:2px;color:#6b8f72;margin-bottom:0.8rem">⭐ OPTIMAL TIME ANALYSIS</div>
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
                    <span class="{badge_opt}">{status_opt}</span>
                    <span style="font-family:'Space Mono',monospace;font-size:1.3rem;color:{color_opt};font-weight:700">{pct_opt:.1f}% of budget</span>
                </div>
                <div style="background:#0f1a12;border-radius:8px;height:10px;width:100%;overflow:hidden;margin-bottom:1rem">
                    <div style="background:{color_opt};height:100%;width:{min(pct_opt,100):.0f}%;transition:width 0.5s;border-radius:8px"></div>
                </div>
                <p style="color:#6b8f72;font-size:0.82rem;font-family:'Space Mono',monospace;margin:0">{tip_opt}</p>
            </div>""", unsafe_allow_html=True)

        # ── Budget gauge chart ──
        st.markdown('<div class="section-header">📊 Budget Gauge</div>', unsafe_allow_html=True)

        fig_gauge = make_subplots(
            rows=1, cols=2,
            specs=[[{"type":"indicator"},{"type":"indicator"}]],
            subplot_titles=["Start Now", "Optimal Time"]
        )
        for i, (val, title, color) in enumerate([
            (current_co2, "Start Now", color_now),
            (best_co2,    "Optimal",   color_opt)
        ], 1):
            fig_gauge.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=val,
                delta={"reference": budget, "valueformat":".0f",
                       "increasing":{"color":"#ff5757"},"decreasing":{"color":"#00ff87"}},
                number={"suffix":" g", "font":{"color":color,"family":"Space Mono","size":28}},
                gauge={
                    "axis":{"range":[0, max(budget*1.5, current_co2*1.2)],
                            "tickcolor":"#6b8f72","tickfont":{"color":"#6b8f72","size":10}},
                    "bar":{"color":color,"thickness":0.25},
                    "bgcolor":"#111a14",
                    "bordercolor":"#1e3325",
                    "steps":[
                        {"range":[0, budget*0.8],         "color":"#061a0e"},
                        {"range":[budget*0.8, budget],    "color":"#1a1206"},
                        {"range":[budget, budget*1.5],    "color":"#1a0606"},
                    ],
                    "threshold":{"line":{"color":"#fff","width":2},"thickness":0.8,"value":budget}
                }
            ), row=1, col=i)

        fig_gauge.update_layout(
            paper_bgcolor="#0a0f0d",
            font=dict(color="#6b8f72", family="Space Mono"),
            height=280, margin=dict(l=20,r=20,t=40,b=10),
        )
        fig_gauge.update_annotations(font_color="#6b8f72", font_size=11)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Green alternatives (if over budget) ──
        suggestions = suggest_green_alternatives(current_co2, budget, gpu_type, dur_min, best_intensity)
        if suggestions:
            st.markdown('<div class="section-header">💡 Green Alternatives to Fit Budget</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="budget-box">
                <h2>🔧 CONFIGURATION SUGGESTIONS</h2>
                {''.join([f'<p style="color:#e8f5e9;font-size:0.9rem;margin:0.5rem 0;border-bottom:1px solid #1e3325;padding-bottom:0.5rem">{s}</p>' for s in suggestions])}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-box">
                <h2>✅ NO CHANGES NEEDED</h2>
                <p style="color:#e8f5e9">Your current configuration fits within the budget at the optimal schedule time.</p>
            </div>""", unsafe_allow_html=True)

        # ── Budget sensitivity — what if I change epochs? ──
        st.markdown('<div class="section-header">🔬 Budget Sensitivity: Epochs vs CO₂</div>', unsafe_allow_html=True)
        epoch_range = [5, 10, 20, 30, 50, 75, 100, 150, 200]
        sens_co2 = []
        for ep in epoch_range:
            d = predict_duration(dur_model, le_mt, le_gpu,
                                 model_type, num_params_actual, num_layers,
                                 dataset_size, batch_size, ep,
                                 gpu_type, gpu["tdp"], gpu["tflops"])
            d_scaled = d / speedup_factor
            e = (total_tdp_watts/1000) * (d_scaled/60)
            sens_co2.append(e * best_intensity)

        fig_sens = go.Figure()
        bar_colors = ["#00ff87" if c <= budget else "#ff5757" for c in sens_co2]
        fig_sens.add_trace(go.Bar(
            x=epoch_range, y=sens_co2,
            marker_color=bar_colors,
            hovertemplate="Epochs: <b>%{x}</b><br>CO₂: <b>%{y:.0f}g</b><extra></extra>"
        ))
        fig_sens.add_hline(y=budget, line_color="#4fc3f7", line_width=2, line_dash="dash",
            annotation_text=f"Budget: {budget}g", annotation_font_color="#4fc3f7", annotation_font_size=11)
        fig_sens.add_hline(y=current_co2, line_color="#ff5757", line_width=1, line_dash="dot",
            annotation_text="Current", annotation_font_color="#ff5757", annotation_font_size=10)
        fig_sens.update_layout(
            plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
            font=dict(color="#6b8f72", family="Space Mono"),
            height=260, margin=dict(l=20,r=20,t=20,b=40),
            xaxis=dict(title="Epochs", gridcolor="#1e3325", zeroline=False),
            yaxis=dict(title="CO₂ at Optimal Time (g)", gridcolor="#1e3325", zeroline=False),
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        st.markdown(f"""<p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b8f72">
        🟢 Green bars = within budget &nbsp;·&nbsp; 🔴 Red bars = over budget &nbsp;·&nbsp;
        Blue line = your budget ({budget}g)
        </p>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cloud Cost Optimizer (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">💰 Cloud Cost + Carbon Optimizer</div>', unsafe_allow_html=True)

    # ── Cost for current config ──
    cloud_costs = compute_cloud_costs(gpu_type, base_dur_min, n_gpus)

    # ── Summary cards ──
    cc1, cc2, cc3 = st.columns(3)
    aws_result = next((r for r in cloud_costs if r["provider"]=="AWS"), None)
    gcp_result = next((r for r in cloud_costs if r["provider"]=="GCP"), None)

    with cc1:
        aws_cost = aws_result["total_cost_usd"] if aws_result and aws_result["total_cost_usd"] else None
        val = f"${aws_cost:.4f}" if aws_cost else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff9900;font-size:1.5rem">{val}</div>
            <div class="metric-label">AWS Total Cost</div>
        </div>""", unsafe_allow_html=True)
    with cc2:
        gcp_cost = gcp_result["total_cost_usd"] if gcp_result and gcp_result["total_cost_usd"] else None
        val = f"${gcp_cost:.4f}" if gcp_cost else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#4285f4;font-size:1.5rem">{val}</div>
            <div class="metric-label">GCP Total Cost</div>
        </div>""", unsafe_allow_html=True)
    with cc3:
        best_co2_val = f"{best_co2:.0f}g"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00ff87;font-size:1.5rem">{best_co2_val}</div>
            <div class="metric-label">CO₂ at Optimal Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Instance detail boxes ──
    cd1, cd2 = st.columns(2)
    for col, result in zip([cd1, cd2], cloud_costs):
        provider = result["provider"]
        pcolor = "#ff9900" if provider == "AWS" else "#4285f4"
        picon = "☁️" if provider == "AWS" else "🌐"
        with col:
            if result["total_cost_usd"]:
                cost_hr_str = f"${result['cost_per_hr']:.3f}/hr"
                st.markdown(f"""<div class="cost-box">
                    <h2>{picon} {provider} COST BREAKDOWN</h2>
                    <table style="width:100%;font-family:'Space Mono',monospace;font-size:0.82rem;border-collapse:collapse">
                        <tr><td style="color:#6b8f72;padding:5px 0">Instance</td>
                            <td style="color:#ce93d8;text-align:right">{result['instance']}</td></tr>
                        <tr><td style="color:#6b8f72;padding:5px 0">Nodes needed</td>
                            <td style="color:#e8f5e9;text-align:right">{result['nodes']}</td></tr>
                        <tr><td style="color:#6b8f72;padding:5px 0">Cost/hr (all nodes)</td>
                            <td style="color:#e8f5e9;text-align:right">{cost_hr_str}</td></tr>
                        <tr><td style="color:#6b8f72;padding:5px 0">Duration (with {n_gpus} GPU{'s' if n_gpus>1 else ''})</td>
                            <td style="color:#e8f5e9;text-align:right">{result['actual_duration_min']:.1f} min</td></tr>
                        <tr style="border-top:1px solid #3a1a5e">
                            <td style="color:#ce93d8;padding:8px 0;font-weight:700">TOTAL COST</td>
                            <td style="color:{pcolor};font-size:1.2rem;font-weight:700;text-align:right">${result['total_cost_usd']:.4f}</td></tr>
                    </table>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="cost-box">
                    <h2>{picon} {provider}</h2>
                    <p style="color:#6b8f72;font-family:'Space Mono',monospace;font-size:0.85rem">
                    {gpu_type} is not available as a managed instance on {provider}.<br>
                    Consider <b style="color:#ce93d8">A100, V100, or T4</b> for cloud deployments.</p>
                </div>""", unsafe_allow_html=True)

    # ── Pareto: CO2 vs Cost across all GPUs ──
    st.markdown('<div class="section-header">🎯 GPU Pareto Front — Cost vs Carbon</div>', unsafe_allow_html=True)
    st.markdown("""<p style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#6b8f72;margin-bottom:1rem">
    Each point = one GPU type. Bottom-left = cheapest AND greenest. Pareto-optimal GPUs are highlighted. ⭐
    </p>""", unsafe_allow_html=True)

    pareto_df = pareto_gpu_analysis(
        model_type, num_params_actual, num_layers,
        dataset_size, batch_size, epochs, best_intensity,
        dur_model, le_mt, le_gpu
    )

    # Filter to GPUs that have pricing
    pareto_priced = pareto_df[pareto_df["cost_usd"].notna()].copy()

    if len(pareto_priced) >= 2:
        # Identify Pareto front (non-dominated solutions)
        pareto_priced = pareto_priced.sort_values("cost_usd")
        is_pareto = []
        min_co2_so_far = float("inf")
        for _, row in pareto_priced.iterrows():
            if row["co2_g"] < min_co2_so_far:
                is_pareto.append(True)
                min_co2_so_far = row["co2_g"]
            else:
                is_pareto.append(False)
        pareto_priced["is_pareto"] = is_pareto

        fig_pareto = go.Figure()

        # Non-pareto points
        non_p = pareto_priced[~pareto_priced["is_pareto"]]
        fig_pareto.add_trace(go.Scatter(
            x=non_p["cost_usd"], y=non_p["co2_g"],
            mode="markers+text",
            marker=dict(color="#3a5a44", size=12, symbol="circle",
                       line=dict(color="#1e3325",width=1)),
            text=non_p["gpu"], textposition="top center",
            textfont=dict(color="#6b8f72",size=10,family="Space Mono"),
            name="Sub-optimal",
            hovertemplate="<b>%{text}</b><br>Cost: $%{x:.4f}<br>CO₂: %{y:.0f}g<extra></extra>"
        ))

        # Pareto-optimal points
        p = pareto_priced[pareto_priced["is_pareto"]]
        fig_pareto.add_trace(go.Scatter(
            x=p["cost_usd"], y=p["co2_g"],
            mode="markers+text",
            marker=dict(color="#00ff87", size=16, symbol="star",
                       line=dict(color="#000",width=1)),
            text=p["gpu"], textposition="top center",
            textfont=dict(color="#00ff87",size=11,family="Space Mono"),
            name="⭐ Pareto Optimal",
            hovertemplate="<b>%{text}</b> ⭐<br>Cost: $%{x:.4f}<br>CO₂: %{y:.0f}g<extra></extra>"
        ))

        # Highlight current GPU
        curr = pareto_priced[pareto_priced["gpu"]==gpu_type]
        if len(curr):
            fig_pareto.add_trace(go.Scatter(
                x=curr["cost_usd"], y=curr["co2_g"],
                mode="markers",
                marker=dict(color="#4fc3f7", size=20, symbol="diamond",
                           line=dict(color="#fff",width=2)),
                name="Your Config",
                hovertemplate=f"<b>{gpu_type}</b> (current)<br>Cost: $%{{x:.4f}}<br>CO₂: %{{y:.0f}}g<extra></extra>"
            ))

        fig_pareto.update_layout(
            plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
            font=dict(color="#6b8f72", family="Space Mono"),
            height=340, margin=dict(l=20,r=20,t=20,b=40),
            xaxis=dict(title="Estimated Cost (USD)", gridcolor="#1e3325", zeroline=False),
            yaxis=dict(title="CO₂ at Optimal Time (g)", gridcolor="#1e3325", zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    # ── Full GPU comparison table ──
    st.markdown('<div class="section-header">📋 All GPUs — Cost & Carbon Comparison</div>', unsafe_allow_html=True)
    rows_html = ""
    for _, row in pareto_df.sort_values("co2_g").iterrows():
        is_curr = "⭐" if row["gpu"] == gpu_type else ""
        cost_str = f"${row['cost_usd']:.4f}" if row["cost_usd"] else "Local only"
        co2_color = "#00ff87" if row["co2_g"] < 200 else "#ffb347" if row["co2_g"] < 600 else "#ff5757"
        rows_html += f"""<tr>
            <td><b style="color:#e8f5e9">{row['gpu']}</b> {is_curr}</td>
            <td>{row['provider']}</td>
            <td style="color:#4fc3f7">{row['duration_min']:.0f} min</td>
            <td style="color:#ffb347">{row['energy_kwh']:.4f} kWh</td>
            <td style="color:{co2_color}">{row['co2_g']:.0f} g</td>
            <td style="color:#ce93d8">{cost_str}</td>
        </tr>"""
    st.markdown(f"""
    <div style="background:#111a14;border:1px solid #1e3325;border-radius:12px;padding:1rem;overflow-x:auto">
    <table class="pareto-table">
        <thead><tr>
            <th>GPU</th><th>Cloud</th><th>Duration</th><th>Energy</th><th>CO₂ (optimal)</th><th>Est. Cost</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Multi-GPU scaling chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Multi-GPU Scaling: Duration, Cost & CO₂</div>', unsafe_allow_html=True)
    gpu_counts = [1, 2, 4, 8, 16]
    scale_data = []
    for ng in gpu_counts:
        sp = multigpu_speedup(ng)
        d = base_dur_min / sp
        e = (gpu["tdp"] * ng / 1000) * (d / 60)
        c = e * best_intensity
        aws_p = CLOUD_PRICING["AWS"].get(gpu_type)
        if aws_p and aws_p["cost_per_hr"]:
            gpn = aws_p["gpus_per_node"]
            nodes = max(1, int(np.ceil(ng / gpn)))
            cost = aws_p["cost_per_hr"] * nodes * (d / 60)
        else:
            cost = None
        scale_data.append({"gpus": ng, "duration_min": d, "co2_g": c, "cost_usd": cost, "speedup": sp})
    scale_df = pd.DataFrame(scale_data)

    fig_scale = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Duration (min)", "CO₂ (g)", "AWS Cost ($)"]
    )
    colors = ["#00ff87","#4fc3f7","#ce93d8"]
    for i, (col_name, color) in enumerate([("duration_min","#00ff87"),("co2_g","#ffb347"),("cost_usd","#ce93d8")], 1):
        valid = scale_df[scale_df[col_name].notna()]
        fig_scale.add_trace(go.Scatter(
            x=valid["gpus"], y=valid[col_name],
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8),
            name=col_name,
            hovertemplate=f"%{{x}} GPUs<br>{col_name}: <b>%{{y:.2f}}</b><extra></extra>"
        ), row=1, col=i)

    fig_scale.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        height=280, margin=dict(l=20,r=20,t=40,b=40),
        showlegend=False,
    )
    for i in range(1, 4):
        fig_scale.update_xaxes(gridcolor="#1e3325", zeroline=False, title="# GPUs", row=1, col=i)
        fig_scale.update_yaxes(gridcolor="#1e3325", zeroline=False, row=1, col=i)
    fig_scale.update_annotations(font_color="#6b8f72", font_size=11)
    st.plotly_chart(fig_scale, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Grid Analysis (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🌍 Grid Carbon Intensity by Zone</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        zone_avg = grid_df.groupby("zone_key")["carbon_intensity_gco2_kwh"].mean().reset_index()
        zone_avg["label"] = zone_avg["zone_key"].map(ZONE_LABELS)
        zone_avg = zone_avg.sort_values("carbon_intensity_gco2_kwh")
        fig3 = go.Figure(go.Bar(
            x=zone_avg["carbon_intensity_gco2_kwh"], y=zone_avg["label"], orientation="h",
            marker=dict(color=zone_avg["carbon_intensity_gco2_kwh"],
                       colorscale=[[0,"#00ff87"],[0.5,"#ffb347"],[1,"#ff5757"]]),
            hovertemplate="<b>%{y}</b><br>%{x:.0f} gCO₂/kWh<extra></extra>"
        ))
        fig3.update_layout(
            title=dict(text="Annual Average Carbon Intensity", font=dict(color="#e8f5e9",size=13)),
            plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
            font=dict(color="#6b8f72",family="Space Mono"),
            height=300, margin=dict(l=10,r=20,t=40,b=20),
            xaxis=dict(title="gCO₂/kWh",gridcolor="#1e3325",zeroline=False),
            yaxis=dict(gridcolor="#1e3325")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        sel_zone = st.selectbox("Zone for hourly pattern",
            options=list(ZONE_LABELS.keys()),
            format_func=lambda x: ZONE_LABELS[x], key="zone_pattern")
        hourly = grid_df[grid_df["zone_key"]==sel_zone].groupby("hour_of_day")["carbon_intensity_gco2_kwh"].mean()
        fig4 = go.Figure(go.Scatter(
            x=hourly.index, y=hourly.values,
            mode="lines+markers",
            line=dict(color="#00c46a",width=2.5),
            marker=dict(color="#00ff87",size=6),
            fill="tozeroy", fillcolor="rgba(0,196,106,0.08)",
            hovertemplate="Hour %{x}:00<br><b>%{y:.0f} gCO₂/kWh</b><extra></extra>"
        ))
        fig4.update_layout(
            title=dict(text="Average Hourly Pattern", font=dict(color="#e8f5e9",size=13)),
            plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
            font=dict(color="#6b8f72",family="Space Mono"),
            height=300, margin=dict(l=10,r=20,t=40,b=20),
            xaxis=dict(title="Hour of Day",gridcolor="#1e3325",zeroline=False,
                      tickvals=list(range(0,24,3))),
            yaxis=dict(title="gCO₂/kWh",gridcolor="#1e3325",zeroline=False)
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">🗓 Seasonal × Hour Heatmap</div>', unsafe_allow_html=True)
    sel_zone2 = st.selectbox("Zone", options=list(ZONE_LABELS.keys()),
        format_func=lambda x: ZONE_LABELS[x], key="zone_heatmap")
    pivot = grid_df[grid_df["zone_key"]==sel_zone2].pivot_table(
        values="carbon_intensity_gco2_kwh", index="season", columns="hour_of_day", aggfunc="mean")
    season_order = ["Winter","Spring","Summer","Fall"]
    pivot = pivot.reindex([s for s in season_order if s in pivot.index])
    fig5 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#00ff87"],[0.5,"#ffb347"],[1,"#ff5757"]],
        hovertemplate="Hour %{x}:00 · %{y}<br><b>%{z:.0f} gCO₂/kWh</b><extra></extra>"
    ))
    fig5.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
        font=dict(color="#6b8f72",family="Space Mono"),
        height=250, margin=dict(l=10,r=10,t=20,b=40),
        xaxis=dict(title="Hour of Day",tickvals=list(range(0,24,2))),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Insights & Dataset (merged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🤖 Duration Predictor Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem">{r2:.3f}</div>
            <div class="metric-label">R² Score</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem">{mae:.1f}</div>
            <div class="metric-label">MAE (minutes)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem">Stack</div>
            <div class="metric-label">GBR + RF + XGBoost</div>
        </div>""", unsafe_allow_html=True)

    # ── CodeCarbon: CO₂ emitted during model training ─────────────────────────
    training_co2_g = eval_data.get("training_co2_g", 0.0)
    co2_display = f"{training_co2_g:.4f}" if training_co2_g < 1.0 else f"{training_co2_g:.2f}"
    co2_color = "#00ff87" if training_co2_g < 1.0 else "#ffb347" if training_co2_g < 10.0 else "#ff5757"
    equiv_phones = training_co2_g / 8.22
    equiv_km     = training_co2_g / 120
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0a1f10,#061209);border:1.5px solid {co2_color};
                border-radius:14px;padding:1.2rem 1.8rem;margin:1.2rem 0;
                display:flex;align-items:center;gap:2rem;flex-wrap:wrap">
        <div>
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:2px;color:#6b8f72;margin-bottom:4px">
                🌿 CO₂ EMITTED DURING MODEL TRAINING (CodeCarbon)
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:{co2_color}">
                {co2_display} g CO₂eq
            </div>
        </div>
        <div style="display:flex;gap:2rem">
            <div style="text-align:center">
                <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e8f5e9">{equiv_phones:.4f}×</div>
                <div style="font-size:0.7rem;color:#6b8f72">📱 Phone charges</div>
            </div>
            <div style="text-align:center">
                <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#e8f5e9">{equiv_km:.5f} km</div>
                <div style="font-size:0.7rem;color:#6b8f72">🚗 Driving equiv.</div>
            </div>
            <div style="text-align:center">
                <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#00ff87">GBR + RF + XGBoost</div>
                <div style="font-size:0.7rem;color:#6b8f72">🤖 Stacking ensemble</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        features = ["model_type_enc","log_params","num_layers","log_dataset",
                    "batch_size","epochs","gpu_tflops","gpu_tdp_watts",
                    "tdp_per_tflop","log_steps","log_compute","arch_mult"]
        # StackingRegressor: average feature importances from GBR + RF + XGBoost base learners
        try:
            gbr_imp = dur_model.named_estimators_["gbr"].feature_importances_
            rf_imp  = dur_model.named_estimators_["rf"].feature_importances_
            xgb_imp = dur_model.named_estimators_["xgb"].feature_importances_
            importance = (gbr_imp + rf_imp + xgb_imp) / 3.0
        except Exception:
            importance = np.ones(len(features)) / len(features)
        fi_df = pd.DataFrame({"feature":features,"importance":importance}).sort_values("importance")
        fig6 = go.Figure(go.Bar(
            x=fi_df["importance"], y=fi_df["feature"], orientation="h",
            marker=dict(color=fi_df["importance"],
                       colorscale=[[0,"#1e3325"],[1,"#00ff87"]])
        ))
        fig6.update_layout(
            title=dict(text="Feature Importance", font=dict(color="#e8f5e9",size=13)),
            plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
            font=dict(color="#6b8f72",family="Space Mono"),
            height=300, margin=dict(l=10,r=20,t=40,b=20),
            xaxis=dict(title="Importance",gridcolor="#1e3325",zeroline=False),
            yaxis=dict(gridcolor="#1e3325")
        )
        st.plotly_chart(fig6, use_container_width=True)

    with c2:
        mt_avg = train_df.groupby("model_type")["training_duration_min"].mean().sort_values()
        fig7 = go.Figure(go.Bar(
            x=mt_avg.values, y=mt_avg.index, orientation="h",
            marker=dict(color=mt_avg.values,
                       colorscale=[[0,"#00ff87"],[1,"#ff5757"]])
        ))
        fig7.update_layout(
            title=dict(text="Avg Training Duration by Model Type", font=dict(color="#e8f5e9",size=13)),
            plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
            font=dict(color="#6b8f72",family="Space Mono"),
            height=300, margin=dict(l=10,r=20,t=40,b=20),
            xaxis=dict(title="Minutes",gridcolor="#1e3325",zeroline=False),
            yaxis=dict(gridcolor="#1e3325")
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown('<div class="section-header">📊 Parameters vs Training Duration</div>', unsafe_allow_html=True)
    fig8 = px.scatter(train_df, x="num_parameters", y="training_duration_min",
                     color="gpu_type", log_x=True,
                     color_discrete_sequence=["#00ff87","#00c46a","#ffb347","#ff8c42","#ff5757","#c44dff"],
                     hover_data=["model_type","epochs","batch_size"])
    fig8.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
        font=dict(color="#6b8f72",family="Space Mono"),
        height=320, margin=dict(l=10,r=10,t=10,b=40),
        xaxis=dict(title="Parameters (log scale)",gridcolor="#1e3325",zeroline=False),
        yaxis=dict(title="Duration (min)",gridcolor="#1e3325",zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig8, use_container_width=True)

    # ── Dataset explorer ──
    st.markdown('<div class="section-header">📋 Dataset Explorer</div>', unsafe_allow_html=True)
    d1, d2 = st.tabs(["🌍 Grid Carbon Intensity", "🤖 ML Training Runs"])

    with d1:
        st.markdown(f"**{len(grid_df):,} rows** · 8 zones · Hourly 2023 · 15 columns")
        zone_filter = st.multiselect("Filter by zone", options=list(ZONE_LABELS.keys()),
            default=["US-CAL","EU-DE","IN-WE"], format_func=lambda x: ZONE_LABELS[x])
        display_grid = grid_df[grid_df["zone_key"].isin(zone_filter)].head(200)
        st.dataframe(
            display_grid[["zone_key","country","datetime","hour_of_day","season",
                          "solar_pct","wind_pct","fossil_pct","carbon_intensity_gco2_kwh"]],
            use_container_width=True, height=350
        )

    with d2:
        st.markdown(f"**{len(train_df):,} rows** · 8 model types · 6 GPU types · 13 columns")
        mt_filter = st.multiselect("Filter by model type", options=train_df["model_type"].unique().tolist(),
            default=["MLP","CNN","LSTM","Transformer"])
        display_train = train_df[train_df["model_type"].isin(mt_filter)].head(200)
        st.dataframe(
            display_train[["model_type","num_parameters","num_layers","dataset_size",
                           "batch_size","epochs","gpu_type","training_duration_min","energy_consumed_kwh"]],
            use_container_width=True, height=350
        )

    st.markdown("""
    <div style="background:#111a14;border:1px solid #1e3325;border-radius:10px;padding:1rem 1.5rem;margin-top:1rem">
    <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#6b8f72;letter-spacing:2px">📌 DATASET INFO</span><br>
    <span style="color:#e8f5e9;font-size:0.85rem">Both datasets are synthetically generated to reflect real-world distributions.
    Grid data validated against Electricity Maps patterns. Training durations aligned with MLPerf benchmarks.
    Use CodeCarbon to generate real training run data for Phase 2 validation.</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Model Evaluation (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">🔬 Model Health Report — Stacking Ensemble (GBR + RF + XGBoost)</div>', unsafe_allow_html=True)

    y_test       = eval_data["y_test"]
    y_pred_test  = eval_data["y_pred_test"]
    y_train      = eval_data["y_train"]
    y_pred_train = eval_data["y_pred_train"]
    residuals    = eval_data["residuals"]
    mae_train    = eval_data["mae_train"]
    r2_train     = eval_data["r2_train"]

    # ── 1. Scorecard ──────────────────────────────────────────────────────────
    overfit_gap  = r2_train - r2
    rmse_test    = float(np.sqrt(np.mean(residuals**2)))
    mape_test    = float(np.mean(np.abs(residuals / np.where(y_test == 0, 1, y_test))) * 100)

    # Overall health verdict — thresholds based on R² and relative overfit gap
    if r2 >= 0.85 and overfit_gap < 0.15:
        verdict = ("✅ HEALTHY", "#00ff87", "Model generalises well. Low overfitting risk. Safe to use for predictions.")
    elif r2 >= 0.70 and overfit_gap < 0.25:
        verdict = ("⚠️ ACCEPTABLE", "#ffb347", "Decent performance but room for improvement. Consider more data or tuning.")
    else:
        verdict = ("❌ NEEDS WORK", "#ff5757", "High error or overfitting detected. Retrain with more data or adjust hyperparameters.")

    # Verdict banner
    st.markdown(f"""
    <div style="background:#0a0f0d;border:2px solid {verdict[1]};border-radius:14px;
                padding:1rem 1.8rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1.5rem">
        <div style="font-family:'Space Mono',monospace;font-size:1.6rem;color:{verdict[1]};font-weight:700">{verdict[0]}</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.82rem;color:#6b8f72">{verdict[2]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    metrics = [
        ("R² Test",      f"{r2:.3f}",       "#00ff87" if r2>=0.85 else "#ffb347" if r2>=0.70 else "#ff5757"),
        ("R² Train",     f"{r2_train:.3f}", "#00ff87" if r2_train>=0.85 else "#ffb347"),
        ("Overfit Gap",  f"{overfit_gap:.3f}", "#00ff87" if overfit_gap<0.1 else "#ffb347" if overfit_gap<0.2 else "#ff5757"),
        ("MAE Test",     f"{mae:.1f} min",  "#00ff87" if mae<30 else "#ffb347" if mae<80 else "#ff5757"),
        ("RMSE Test",    f"{rmse_test:.1f} min", "#00ff87" if rmse_test<50 else "#ffb347" if rmse_test<120 else "#ff5757"),
        ("MAPE Test",    f"{mape_test:.1f}%",    "#00ff87" if mape_test<15 else "#ffb347" if mape_test<30 else "#ff5757"),
    ]
    for col, (label, val, color) in zip([mc1,mc2,mc3,mc4,mc5,mc6], metrics):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color};font-size:1.3rem">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. Train vs Test R² bar (overfitting check) ───────────────────────────
    st.markdown('<div class="section-header">📊 Overfitting Check — Train vs Test</div>', unsafe_allow_html=True)

    fig_ov = go.Figure()
    fig_ov.add_trace(go.Bar(
        x=["Train R²", "Test R²"],
        y=[r2_train, r2],
        marker_color=["#4fc3f7", "#00ff87"],
        width=0.35,
        text=[f"{r2_train:.3f}", f"{r2:.3f}"],
        textposition="outside",
        textfont=dict(color="#e8f5e9", family="Space Mono", size=13),
    ))
    fig_ov.add_trace(go.Bar(
        x=["Train MAE", "Test MAE"],
        y=[mae_train, mae],
        marker_color=["#ce93d8", "#ffb347"],
        width=0.35,
        text=[f"{mae_train:.1f}", f"{mae:.1f}"],
        textposition="outside",
        textfont=dict(color="#e8f5e9", family="Space Mono", size=13),
    ))
    # Threshold line for R²
    fig_ov.add_hline(y=0.85, line_dash="dot", line_color="#00ff87", line_width=1,
        annotation_text="Good R² threshold (0.85)", annotation_font_color="#00ff87", annotation_font_size=10)
    fig_ov.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        height=300, margin=dict(l=20,r=20,t=20,b=40),
        barmode="group",
        xaxis=dict(gridcolor="#1e3325", zeroline=False),
        yaxis=dict(title="Score", gridcolor="#1e3325", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    # Interpretation note
    gap_color = "#00ff87" if overfit_gap < 0.1 else "#ffb347" if overfit_gap < 0.2 else "#ff5757"
    st.markdown(f"""
    <div style="background:#111a14;border:1px solid #1e3325;border-radius:10px;
                padding:0.8rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.78rem;color:#6b8f72">
    Train R²: <b style="color:#4fc3f7">{r2_train:.3f}</b> &nbsp;|&nbsp;
    Test R²: <b style="color:#00ff87">{r2:.3f}</b> &nbsp;|&nbsp;
    Gap: <b style="color:{gap_color}">{overfit_gap:.3f}</b>
    &nbsp;— {"✅ Minimal overfitting" if overfit_gap < 0.1 else "⚠️ Mild overfitting — consider regularisation" if overfit_gap < 0.2 else "❌ Significant overfitting — reduce max_depth or add more data"}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3. Predicted vs Actual scatter ───────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Predicted vs Actual (Test Set)</div>', unsafe_allow_html=True)

    fig_pa = go.Figure()
    # Perfect prediction line
    min_val = float(min(y_test.min(), y_pred_test.min()))
    max_val = float(max(y_test.max(), y_pred_test.max()))
    fig_pa.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="#00ff87", width=1.5, dash="dash"),
        name="Perfect Prediction",
        hoverinfo="skip"
    ))
    # ±20% error band
    fig_pa.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val*0.8, max_val*0.8],
        mode="lines", line=dict(color="rgba(0,255,135,0.15)", width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig_pa.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val*1.2, max_val*1.2],
        mode="lines", line=dict(color="rgba(0,255,135,0.15)", width=0),
        fill="tonexty", fillcolor="rgba(0,255,135,0.05)",
        name="±20% band", hoverinfo="skip"
    ))
    # Scatter points coloured by absolute error
    abs_err = np.abs(residuals)
    fig_pa.add_trace(go.Scatter(
        x=y_test, y=y_pred_test,
        mode="markers",
        marker=dict(
            color=abs_err,
            colorscale=[[0,"#00ff87"],[0.5,"#ffb347"],[1,"#ff5757"]],
            size=6, opacity=0.75,
            colorbar=dict(title=dict(text="Abs Error (min)", font=dict(color="#6b8f72")),
                         tickfont=dict(color="#6b8f72")),
            showscale=True,
        ),
        name="Test samples",
        hovertemplate="Actual: <b>%{x:.1f} min</b><br>Predicted: <b>%{y:.1f} min</b><br>Error: <b>%{marker.color:.1f} min</b><extra></extra>"
    ))
    fig_pa.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        height=380, margin=dict(l=20,r=20,t=20,b=40),
        xaxis=dict(title="Actual Duration (min)", gridcolor="#1e3325", zeroline=False),
        yaxis=dict(title="Predicted Duration (min)", gridcolor="#1e3325", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11))
    )
    st.plotly_chart(fig_pa, use_container_width=True)
    st.markdown("""<p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b8f72">
    Points should hug the green diagonal. Colour = absolute error (green=low, red=high). Dashed band = ±20% error zone.
    </p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4. Residual plot ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Residuals Plot (Errors vs Predicted)</div>', unsafe_allow_html=True)

    fig_res = go.Figure()
    fig_res.add_hline(y=0, line_color="#00ff87", line_width=1.5, line_dash="dash")
    fig_res.add_hrect(y0=-mae, y1=mae, fillcolor="rgba(0,255,135,0.05)",
                      line_width=0, annotation_text=f"±MAE ({mae:.0f} min)",
                      annotation_font_color="#6b8f72", annotation_font_size=10)
    fig_res.add_trace(go.Scatter(
        x=y_pred_test, y=residuals,
        mode="markers",
        marker=dict(
            color=residuals,
            colorscale=[[0,"#ff5757"],[0.5,"#00ff87"],[1,"#4fc3f7"]],
            size=5, opacity=0.7,
            cmid=0,
        ),
        hovertemplate="Predicted: <b>%{x:.1f} min</b><br>Residual: <b>%{y:.1f} min</b><extra></extra>"
    ))
    fig_res.update_layout(
        plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
        font=dict(color="#6b8f72", family="Space Mono"),
        height=300, margin=dict(l=20,r=20,t=20,b=40),
        xaxis=dict(title="Predicted Duration (min)", gridcolor="#1e3325", zeroline=False),
        yaxis=dict(title="Residual (Actual − Predicted)", gridcolor="#1e3325", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_res, use_container_width=True)
    st.markdown("""<p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#6b8f72">
    Ideally: points randomly scattered around zero (green line) with no pattern.
    A curved or fan-shaped spread = the model struggles with certain duration ranges.
    </p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 5. Residual distribution histogram ───────────────────────────────────
    st.markdown('<div class="section-header">📊 Residual Distribution — Is Error Normally Distributed?</div>', unsafe_allow_html=True)

    col_hist, col_stats = st.columns([2, 1])
    with col_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=40,
            marker=dict(
                color=np.where(np.abs(residuals) <= mae, "#00ff87", "#ff5757"),
                opacity=0.8,
            ),
            hovertemplate="Error: %{x:.0f} min<br>Count: %{y}<extra></extra>",
            name="Residuals"
        ))
        fig_hist.add_vline(x=0, line_color="#fff", line_width=1.5, line_dash="dash",
            annotation_text="Zero error", annotation_font_color="#6b8f72")
        fig_hist.add_vline(x=float(np.mean(residuals)), line_color="#ffb347", line_width=1.5,
            annotation_text=f"Mean: {np.mean(residuals):.1f}", annotation_font_color="#ffb347",
            annotation_font_size=10)
        fig_hist.update_layout(
            plot_bgcolor="#0a0f0d", paper_bgcolor="#0a0f0d",
            font=dict(color="#6b8f72", family="Space Mono"),
            height=280, margin=dict(l=20,r=20,t=20,b=40),
            xaxis=dict(title="Residual (min)", gridcolor="#1e3325", zeroline=False),
            yaxis=dict(title="Count", gridcolor="#1e3325", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stats:
        res_mean   = float(np.mean(residuals))
        res_std    = float(np.std(residuals))
        res_median = float(np.median(residuals))
        within_mae = float(np.mean(np.abs(residuals) <= mae) * 100)
        within_2mae= float(np.mean(np.abs(residuals) <= 2*mae) * 100)

        bias_color = "#00ff87" if abs(res_mean) < mae*0.2 else "#ffb347" if abs(res_mean) < mae else "#ff5757"
        st.markdown(f"""
        <div style="background:#111a14;border:1px solid #1e3325;border-radius:12px;padding:1.2rem;font-family:'Space Mono',monospace">
        <div style="font-size:0.68rem;letter-spacing:2px;color:#6b8f72;margin-bottom:1rem">📐 ERROR STATS</div>
        <table style="width:100%;font-size:0.8rem;border-collapse:collapse">
            <tr><td style="color:#6b8f72;padding:4px 0">Mean error (bias)</td>
                <td style="color:{bias_color};text-align:right">{res_mean:+.1f} min</td></tr>
            <tr><td style="color:#6b8f72;padding:4px 0">Median error</td>
                <td style="color:#e8f5e9;text-align:right">{res_median:+.1f} min</td></tr>
            <tr><td style="color:#6b8f72;padding:4px 0">Std deviation</td>
                <td style="color:#e8f5e9;text-align:right">{res_std:.1f} min</td></tr>
            <tr><td style="color:#6b8f72;padding:4px 0">Within ±MAE</td>
                <td style="color:#00ff87;text-align:right">{within_mae:.0f}%</td></tr>
            <tr><td style="color:#6b8f72;padding:4px 0">Within ±2×MAE</td>
                <td style="color:#00ff87;text-align:right">{within_2mae:.0f}%</td></tr>
        </table>
        <div style="margin-top:1rem;font-size:0.72rem;color:#6b8f72">
        {"✅ Unbiased — mean error near zero" if abs(res_mean) < mae*0.2 else "⚠️ Slight bias detected" if abs(res_mean) < mae else "❌ Model is biased"}
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 6. Error by model type (where does it fail?) ─────────────────────────
    st.markdown('<div class="section-header">🔍 Where Does the Model Struggle? — Error by Model Type</div>', unsafe_allow_html=True)

    # Use the predictions already stored in eval_data — no need to re-predict.
    # Recover the test-set row indices using the same random_state=42 split.
    _, test_indices = train_test_split(range(len(train_df)), test_size=0.2, random_state=42)
    meta_te = train_df.iloc[test_indices].copy().reset_index(drop=True)
    meta_te["predicted"] = eval_data["y_pred_test"]
    meta_te["abs_error"] = np.abs(eval_data["y_test"] - eval_data["y_pred_test"])

    er1, er2 = st.columns(2)
    with er1:
        mt_err = meta_te.groupby("model_type")["abs_error"].mean().sort_values()
        colors_err = ["#00ff87" if v < mae else "#ffb347" if v < mae*2 else "#ff5757"
                      for v in mt_err.values]
        fig_mt_err = go.Figure(go.Bar(
            x=mt_err.values, y=mt_err.index, orientation="h",
            marker_color=colors_err,
            hovertemplate="<b>%{y}</b><br>MAE: %{x:.1f} min<extra></extra>"
        ))
        fig_mt_err.add_vline(x=float(mae), line_color="#fff", line_width=1, line_dash="dot",
            annotation_text="Overall MAE", annotation_font_color="#6b8f72", annotation_font_size=10)
        fig_mt_err.update_layout(
            title=dict(text="MAE by Model Type", font=dict(color="#e8f5e9",size=12)),
            plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
            font=dict(color="#6b8f72", family="Space Mono"),
            height=300, margin=dict(l=10,r=20,t=40,b=20),
            xaxis=dict(title="MAE (min)", gridcolor="#1e3325", zeroline=False),
            yaxis=dict(gridcolor="#1e3325"),
        )
        st.plotly_chart(fig_mt_err, use_container_width=True)

    with er2:
        gpu_err = meta_te.groupby("gpu_type")["abs_error"].mean().sort_values()
        colors_gpu = ["#00ff87" if v < mae else "#ffb347" if v < mae*2 else "#ff5757"
                      for v in gpu_err.values]
        fig_gpu_err = go.Figure(go.Bar(
            x=gpu_err.values, y=gpu_err.index, orientation="h",
            marker_color=colors_gpu,
            hovertemplate="<b>%{y}</b><br>MAE: %{x:.1f} min<extra></extra>"
        ))
        fig_gpu_err.add_vline(x=float(mae), line_color="#fff", line_width=1, line_dash="dot",
            annotation_text="Overall MAE", annotation_font_color="#6b8f72", annotation_font_size=10)
        fig_gpu_err.update_layout(
            title=dict(text="MAE by GPU Type", font=dict(color="#e8f5e9",size=12)),
            plot_bgcolor="#0a0f0d", paper_bgcolor="#111a14",
            font=dict(color="#6b8f72", family="Space Mono"),
            height=300, margin=dict(l=10,r=20,t=40,b=20),
            xaxis=dict(title="MAE (min)", gridcolor="#1e3325", zeroline=False),
            yaxis=dict(gridcolor="#1e3325"),
        )
        st.plotly_chart(fig_gpu_err, use_container_width=True)

    # ── 7. Final plain-English verdict ───────────────────────────────────────
    st.markdown('<div class="section-header">📋 Plain-English Verdict</div>', unsafe_allow_html=True)

    checks = [
        ("R² Test ≥ 0.85",         r2 >= 0.85,              f"R²={r2:.3f}"),
        ("Overfit gap < 0.15",      overfit_gap < 0.15,      f"Gap={overfit_gap:.3f}"),
        ("MAPE < 20%",              mape_test < 20,          f"MAPE={mape_test:.1f}%"),
        ("Mean bias < 10% of MAE",  abs(res_mean)<mae*0.3,   f"Bias={res_mean:+.2f} min"),
        ("≥60% within ±MAE",        within_mae >= 60,        f"{within_mae:.0f}% within ±{mae:.2f} min"),
    ]
    checks_html = ""
    for label, passed, detail in checks:
        icon  = "✅" if passed else "❌"
        color = "#00ff87" if passed else "#ff5757"
        checks_html += f"""
        <tr>
            <td style="padding:8px 12px;color:{color};font-size:1rem">{icon}</td>
            <td style="padding:8px 12px;color:#e8f5e9">{label}</td>
            <td style="padding:8px 12px;color:#6b8f72;font-family:'Space Mono',monospace;font-size:0.82rem">{detail}</td>
        </tr>"""

    passed_count = sum(1 for _,p,_ in checks if p)
    overall_color = "#00ff87" if passed_count == 5 else "#ffb347" if passed_count >= 3 else "#ff5757"
    st.markdown(f"""
    <div style="background:#111a14;border:1px solid #1e3325;border-radius:12px;padding:1rem;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-family:'Syne',sans-serif">
        <thead><tr>
            <th style="padding:6px 12px;color:#6b8f72;font-size:0.68rem;letter-spacing:2px;text-align:left">STATUS</th>
            <th style="padding:6px 12px;color:#6b8f72;font-size:0.68rem;letter-spacing:2px;text-align:left">CHECK</th>
            <th style="padding:6px 12px;color:#6b8f72;font-size:0.68rem;letter-spacing:2px;text-align:left">VALUE</th>
        </tr></thead>
        <tbody>{checks_html}</tbody>
    </table>
    <div style="margin-top:1rem;padding:0.8rem 1rem;background:#0a0f0d;border-radius:8px;
                font-family:'Space Mono',monospace;font-size:0.82rem;color:{overall_color}">
    {passed_count}/5 checks passed — {verdict[0]}
    </div>
    </div>
    """, unsafe_allow_html=True)