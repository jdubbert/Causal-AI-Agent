"""
Generate three synthetic demo datasets for the Causal Inference Agent.

1. ab_test.csv          — Email campaign A/B test (checking account activation)
2. did.csv              — Marketing campaign DiD (revenue lift across markets)
3. synthetic_control.csv — Geo-targeted campaign synthetic control (DMA conversions)
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT_DIR = Path("causal_agent/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. A/B TEST — Email campaign → checking-account feature activation
# ─────────────────────────────────────────────────────────────────────────────

n_ab = 4000
treatment = np.random.choice(["control", "treatment"], size=n_ab, p=[0.5, 0.5])

# Baseline activation rate ~12 %, treatment lifts it to ~15.5 % (≈ +3.5 pp)
base_rate = 0.12
lift = 0.035
prob = np.where(treatment == "treatment", base_rate + lift, base_rate)
activated = np.random.binomial(1, prob)

# Add some realistic covariates the profiler can see (but analysis ignores)
age = np.random.normal(42, 12, n_ab).clip(18, 80).astype(int)
tenure_months = np.random.exponential(30, n_ab).clip(1, 240).astype(int)

ab_df = pd.DataFrame({
    "customer_id": range(1, n_ab + 1),
    "treatment": treatment,
    "activated": activated,
    "age": age,
    "tenure_months": tenure_months,
})

ab_df.to_csv(OUT_DIR / "ab_test.csv", index=False)
print(f"✅ ab_test.csv  — {len(ab_df)} rows, "
      f"ctrl rate={ab_df[ab_df.treatment=='control'].activated.mean():.3f}, "
      f"treat rate={ab_df[ab_df.treatment=='treatment'].activated.mean():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIFFERENCE-IN-DIFFERENCES — Marketing campaign → revenue by market
# ─────────────────────────────────────────────────────────────────────────────

weeks = list(range(1, 21))          # 20 weeks: 1-10 pre, 11-20 post
markets = [f"market_{i}" for i in range(1, 21)]  # 20 markets
treated_markets = markets[:6]       # first 6 are treatment markets

rows_did = []
for market in markets:
    is_treated = market in treated_markets
    base_revenue = np.random.normal(50000, 5000)  # market-level baseline

    for week in weeks:
        # Common time trend: ~1 % weekly growth
        trend = base_revenue * (1 + 0.01 * (week - 1))

        # Treatment effect kicks in post (week 11+): ~8 % lift
        treatment_effect = trend * 0.08 if (is_treated and week >= 11) else 0

        revenue = trend + treatment_effect + np.random.normal(0, 1500)

        rows_did.append({
            "market": market,
            "week": week,
            "treatment": "treatment" if is_treated else "control",
            "post": int(week >= 11),
            "revenue": round(revenue, 2),
        })

did_df = pd.DataFrame(rows_did)
did_df.to_csv(OUT_DIR / "did.csv", index=False)
print(f"✅ did.csv      — {len(did_df)} rows, {len(markets)} markets, "
      f"{len(treated_markets)} treated, 10 pre + 10 post weeks")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYNTHETIC CONTROL — Geo-targeted campaign → weekly conversions by DMA
# ─────────────────────────────────────────────────────────────────────────────

weeks_sc = list(range(1, 53))        # 52 weeks (1 year)
treatment_start = 27                  # campaign starts week 27
dmas = [f"DMA_{i:03d}" for i in range(1, 16)]  # 15 DMAs
treated_dma = "DMA_001"

rows_sc = []
# Give each DMA its own baseline level and trend
dma_params = {}
for dma in dmas:
    dma_params[dma] = {
        "base": np.random.normal(500, 80),
        "trend": np.random.normal(2.0, 0.5),
        "seasonality_amp": np.random.normal(40, 10),
    }

# Make some donor DMAs correlate with treated DMA
# (this helps synthetic control find a good match)
for dma in ["DMA_003", "DMA_007", "DMA_011"]:
    dma_params[dma]["base"] = dma_params[treated_dma]["base"] + np.random.normal(0, 15)
    dma_params[dma]["trend"] = dma_params[treated_dma]["trend"] + np.random.normal(0, 0.3)

for dma in dmas:
    p = dma_params[dma]
    for week in weeks_sc:
        base = p["base"] + p["trend"] * week
        seasonality = p["seasonality_amp"] * np.sin(2 * np.pi * week / 52)
        noise = np.random.normal(0, 20)

        # Treatment effect: ~12 % lift starting week 27, only for treated DMA
        effect = 0
        if dma == treated_dma and week >= treatment_start:
            effect = base * 0.12

        conversions = base + seasonality + noise + effect

        rows_sc.append({
            "dma": dma,
            "week": week,
            "conversions": round(max(conversions, 0), 1),
            "treated": 1 if dma == treated_dma else 0,
            "post": int(week >= treatment_start),
        })

sc_df = pd.DataFrame(rows_sc)
sc_df.to_csv(OUT_DIR / "synthetic_control.csv", index=False)
print(f"✅ synthetic_control.csv — {len(sc_df)} rows, {len(dmas)} DMAs, "
      f"treated={treated_dma}, pre=wk1-26, post=wk27-52")

print("\n🎉 All demo datasets saved to causal_agent/data/")
