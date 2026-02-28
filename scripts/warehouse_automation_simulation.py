# Data-Driven Warehouse Automation Optimization
# Dataset: logistics_dataset.csv
# Outputs:
#  - cleaned dataframe
#  - automation candidate scoring
#  - simulated "robot-assisted" picking scenario (before vs after)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Basic cleaning and loading the datset
DATA_PATH = "data/logistics_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Parse dates
df["last_restock_date"] = pd.to_datetime(df["last_restock_date"], errors="coerce")

# Basic checks - double checking data
numeric_cols = [
    "stock_level", "reorder_point", "reorder_frequency_days", "lead_time_days",
    "daily_demand", "demand_std_dev", "item_popularity_score",
    "picking_time_seconds", "handling_cost_per_unit", "unit_price",
    "holding_cost_per_unit_day", "stockout_count_last_month",
    "order_fulfillment_rate", "total_orders_last_month", "turnover_ratio",
    "layout_efficiency_score", "forecasted_demand_next_7d", "KPI_score"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows missing key fields
key_fields = ["zone", "picking_time_seconds", "daily_demand", "layout_efficiency_score"]
df = df.dropna(subset=key_fields).reset_index(drop=True)

print("Rows, Cols:", df.shape)
print(df.head(3))

# Baseline KPIs
baseline = {
    "avg_picking_time_sec": float(df["picking_time_seconds"].mean()),
    "median_picking_time_sec": float(df["picking_time_seconds"].median()),
    "p90_picking_time_sec": float(df["picking_time_seconds"].quantile(0.90)),
    "avg_fulfillment_rate": float(df["order_fulfillment_rate"].mean()),
    "avg_stockouts_last_month": float(df["stockout_count_last_month"].mean()),
}
print("\nBaseline KPIs:")
for k, v in baseline.items():
    print(f"  {k}: {v:.3f}")

# Zone-level view
zone_kpis = (
    df.groupby("zone", as_index=False)
      .agg(
          items=("item_id", "count"),
          avg_pick=("picking_time_seconds", "mean"),
          p90_pick=("picking_time_seconds", lambda s: s.quantile(0.90)),
          avg_demand=("daily_demand", "mean"),
          avg_layout_eff=("layout_efficiency_score", "mean"),
          avg_fulfill=("order_fulfillment_rate", "mean"),
          avg_stockouts=("stockout_count_last_month", "mean"),
      )
      .sort_values(["avg_pick", "avg_demand"], ascending=[False, False])
)

print("\nTop zones by avg picking time (potential automation targets):")
print(zone_kpis.head(10))

# plot to showcase distribution of picking time
plt.figure()
plt.hist(df["picking_time_seconds"].dropna(), bins=40)
plt.title("Distribution of Picking Time (seconds)")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.show()

# "Automation Candidate" SKUs - (simple, explainable scoring)
# Candidate intuition:
# - High picking time
# - High demand
# - Low layout efficiency
# - High handling cost
# - Higher stockouts / lower fulfillment
def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if np.isclose(mx - mn, 0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

df["score_pick_time"] = minmax(df["picking_time_seconds"])
df["score_demand"] = minmax(df["daily_demand"])
df["score_layout_bad"] = 1 - minmax(df["layout_efficiency_score"])  # low efficiency => higher score
df["score_handling_cost"] = minmax(df["handling_cost_per_unit"])

# Weights
W_PICK, W_DEMAND, W_LAYOUT, W_COST = 0.40, 0.30, 0.20, 0.10
df["automation_candidate_score"] = (
    W_PICK * df["score_pick_time"] +
    W_DEMAND * df["score_demand"] +
    W_LAYOUT * df["score_layout_bad"] +
    W_COST * df["score_handling_cost"]
)

# Flag top X % as candidates
TOP_PCT = 0.25  # top 25% as automation candidates
threshold = df["automation_candidate_score"].quantile(1 - TOP_PCT)
df["automation_candidate_flag"] = (df["automation_candidate_score"] >= threshold).astype(int)

print("\nAutomation candidates flagged:", int(df["automation_candidate_flag"].sum()), "of", len(df))

# Simulate robot-assisted picking
# Assumptions:
# - Robots reduce picking time for candidate items by 20–35% depending on layout & congestion
# - Robot "speedup" is bigger when layout efficiency is low (robots help more in messy zones)
# - Small overhead (handoff / scanning) added back to robot-assisted picks
rng = np.random.default_rng(42)

# Speedup base between 0.20 and 0.35
base_speedup = rng.uniform(0.20, 0.35, size=len(df))

# Extra speedup when layout is poor (layout_eff low => roughly around +0.10)
layout_bonus = (1 - minmax(df["layout_efficiency_score"])) * 0.10

speedup = np.clip(base_speedup + layout_bonus, 0.15, 0.45)  # realistic bounds

# Overhead (seconds) for robot-assisted picks: e.g., staging / scan / handoff
robot_overhead_sec = rng.normal(loc=8, scale=3, size=len(df))
robot_overhead_sec = np.clip(robot_overhead_sec, 2, 20)

df["robot_assisted"] = df["automation_candidate_flag"]

# Manual baseline
df["manual_pick_time_seconds"] = df["picking_time_seconds"].astype(float)

# Robot-assisted time:
# - If robot_assisted == 1: reduced time + overhead
# - Else: same as manual
df["robot_pick_time_seconds"] = np.where(
    df["robot_assisted"] == 1,
    df["manual_pick_time_seconds"] * (1 - speedup) + robot_overhead_sec,
    df["manual_pick_time_seconds"]
)

# Compute a simple "robot utilization proxy"
robot_workload = df.loc[df["robot_assisted"] == 1, "robot_pick_time_seconds"].sum()
total_workload = df["robot_pick_time_seconds"].sum()
df["robot_workload_share_proxy"] = robot_workload / total_workload if total_workload > 0 else np.nan

# Before vs after summary
manual_avg = df["manual_pick_time_seconds"].mean()
robot_avg = df["robot_pick_time_seconds"].mean()
improvement_pct = (manual_avg - robot_avg) / manual_avg * 100

print("\nBefore vs After (Picking Time):")
print(f"  Manual avg pick time: {manual_avg:.2f} sec")
print(f"  Robot-assisted scenario avg pick time: {robot_avg:.2f} sec")
print(f"  Avg pick time improvement: {improvement_pct:.1f}%")
print(f"  Robot workload share (proxy): {df['robot_workload_share_proxy'].iloc[0]:.2%}")

# Plot (before vs after)
plt.figure()
plt.hist(df["manual_pick_time_seconds"], bins=40, alpha=0.6, label="Manual")
plt.hist(df["robot_pick_time_seconds"], bins=40, alpha=0.6, label="Robot-assisted scenario")
plt.title("Picking Time: Manual vs Robot-assisted Scenario")
plt.xlabel("Seconds")
plt.ylabel("Count")
plt.legend()
plt.show()

# Save enhanced dataset
OUT_PATH = "data/dataOutput/logistics_dataset_enhanced_robot_sim.csv"
df.to_csv(OUT_PATH, index=False)
print("\nSaved enhanced dataset to:", OUT_PATH)

# Small zone-level summary
zone_compare = (
    df.groupby("zone", as_index=False)
      .agg(
          items=("item_id", "count"),
          manual_avg_pick=("manual_pick_time_seconds", "mean"),
          robot_avg_pick=("robot_pick_time_seconds", "mean"),
          candidate_share=("robot_assisted", "mean"),
          avg_demand=("daily_demand", "mean"),
          avg_layout_eff=("layout_efficiency_score", "mean"),
      )
)
zone_compare["pick_time_improvement_pct"] = (
    (zone_compare["manual_avg_pick"] - zone_compare["robot_avg_pick"])
    / zone_compare["manual_avg_pick"] * 100
)

ZONE_OUT = "data/dataOutput/zone_summary_before_after.csv"
zone_compare.to_csv(ZONE_OUT, index=False)
print("Saved zone summary to:", ZONE_OUT)
