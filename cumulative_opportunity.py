from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import KDTree
from math import radians, cos, sin, asin, sqrt
import pyproj

BASE_DIR    = Path(r"D:\nyc shp\nyc_toilet_esda")
CSV_PATH    = BASE_DIR / "baseline_final.csv"
PUBLIC_PATH = BASE_DIR / "public_toilet_975.csv"
HEX_PATH    = BASE_DIR / "outputs" / "h3_grid_res8.gpkg"
TRACT_PATH  = BASE_DIR / "tl_2025_36_tract" / "tl_2025_36_tract.shp"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

nyc_counties = ["005", "047", "061", "081", "085"]

# 步行速度 83m/min，算各时间圈对应的直线距离
WALK_M_PER_MIN = 83.0
THRESHOLDS_M = {
    "within_5min":  5  * WALK_M_PER_MIN,
    "within_10min": 10 * WALK_M_PER_MIN,
    "within_15min": 15 * WALK_M_PER_MIN,
}


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


# 读供给点
print("读供给点...")
df = pd.read_csv(CSV_PATH)
df.columns = (df.columns.astype(str).str.strip()
              .str.replace("\ufeff", "", regex=False)
              .str.replace("ï»¿", "", regex=False))
supply_semi = df[["lat", "lon"]].dropna().copy()
supply_semi.columns = ["lat", "lng"]

pub = pd.read_csv(PUBLIC_PATH)
pub.columns = pub.columns.str.strip()
supply_pub = pub[["lat", "lon"]].dropna().copy()
supply_pub.columns = ["lat", "lng"]

all_supply = pd.concat([supply_semi, supply_pub], ignore_index=True).dropna().reset_index(drop=True)
print(f"semi-public: {len(supply_semi)}, public: {len(supply_pub)}, 合计: {len(all_supply)}")

supply_lats = all_supply["lat"].values
supply_lngs = all_supply["lng"].values

hex_gdf = gpd.read_file(str(HEX_PATH))
print(f"格子: {len(hex_gdf)}")

# KDTree初筛半径：15分钟圈1245米，换成度数 + 1.5倍余量
max_deg  = 1245 / 85000 * 1.5
poi_tree = KDTree(np.column_stack([supply_lats, supply_lngs]))

print("\n开始算 Cumulative Opportunity...")
results = []
total   = len(hex_gdf)

for i, (_, row) in enumerate(hex_gdf.iterrows()):
    if i % 200 == 0:
        print(f"  {i}/{total}  {i/total*100:.1f}%")

    clat = row["centroid_lat"]
    clng = row["centroid_lon"]

    candidate_idx = poi_tree.query_ball_point([clat, clng], r=max_deg)

    counts = {k: 0 for k in THRESHOLDS_M}
    for idx in candidate_idx:
        dist_m = haversine_m(clat, clng, supply_lats[idx], supply_lngs[idx])
        for col, threshold in THRESHOLDS_M.items():
            if dist_m <= threshold:
                counts[col] += 1

    results.append({"h3_index": row["h3_index"], **counts})

print("算完了")

cum_df     = pd.DataFrame(results)
hex_result = hex_gdf.merge(cum_df, on="h3_index", how="left")
hex_result.to_file(str(OUTPUT_DIR / "hex_cumulative.gpkg"), driver="GPKG")

for col in ["within_5min", "within_10min", "within_15min"]:
    v = hex_result[col]
    print(f"\n{col}:  0个: {(v==0).sum()} ({(v==0).mean()*100:.1f}%)  均值: {v.mean():.1f}  中位数: {v.median():.1f}")

# 出图（5分钟和10分钟）
tracts = gpd.read_file(TRACT_PATH)
tracts.columns = (tracts.columns.astype(str).str.strip()
                  .str.replace("\ufeff", "", regex=False)
                  .str.replace("ï»¿", "", regex=False))
tracts = tracts[tracts["COUNTYFP"].isin(nyc_counties)].to_crs(epsg=2263)
nyc_outline    = tracts.dissolve()
tracts["borough"] = tracts["COUNTYFP"].map({
    "005": "Bronx", "047": "Brooklyn", "061": "Manhattan",
    "081": "Queens", "085": "Staten Island"
})
borough_bounds = tracts.dissolve(by="borough")

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
b_labels = {
    "Manhattan":     (-74.005, 40.728),
    "Brooklyn":      (-73.950, 40.645),
    "Queens":        (-73.820, 40.700),
    "Bronx":         (-73.880, 40.860),
    "Staten Island": (-74.155, 40.580),
}

cmap_blue = LinearSegmentedColormap.from_list("Blues7", [
    "#eff3ff", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"
])

scenarios = [
    ("within_5min",  "Toilets within 5-min walk"),
    ("within_10min", "Toilets within 10-min walk"),
]

fig, axes = plt.subplots(1, 2, figsize=(22, 16))
hex_plot  = hex_result.to_crs(epsg=3857)

for ax, (col, title) in zip(axes, scenarios):
    v        = hex_result[col]
    vmax     = max(int(v.quantile(0.95)), 1)
    n_zero   = int((v == 0).sum())
    pct_zero = (v == 0).mean() * 100

    nyc_outline.to_crs(epsg=3857).plot(ax=ax, color="#e0e0e0", zorder=1)

    # 灰色标出没有厕所的格子
    zero_hex = hex_plot[hex_plot[col] == 0]
    if len(zero_hex) > 0:
        zero_hex.plot(ax=ax, color="#b0b0b0", linewidth=0.08, edgecolor="white", zorder=2)

    nonzero_hex = hex_plot[hex_plot[col] > 0]
    if len(nonzero_hex) > 0:
        nonzero_hex.plot(
            column=col, ax=ax, cmap=cmap_blue, vmin=1, vmax=vmax,
            legend=True,
            legend_kwds={"label": "Number of toilets reachable", "shrink": 0.55, "pad": 0.02, "aspect": 30},
            linewidth=0.08, edgecolor="white", zorder=3)

    borough_bounds.to_crs(epsg=3857).boundary.plot(
        ax=ax, linewidth=0.9, color="white", alpha=0.7, zorder=4)

    for name, (lon, lat) in b_labels.items():
        x, y = transformer.transform(lon, lat)
        ax.text(x, y, name, fontsize=8, color="#333", fontweight="bold",
                ha="center", va="center", zorder=5)

    ax.legend(
        handles=[mpatches.Patch(color="#b0b0b0", label=f"No toilet in reach  (n={n_zero}, {pct_zero:.1f}%)")],
        loc="lower left", fontsize=8, framealpha=0.85, edgecolor="none")

    ax.set_title(
        f"{title}\nMedian: {v.median():.0f}  |  Mean: {v.mean():.1f}  |  95th pct max: {vmax}",
        fontsize=11)
    ax.set_axis_off()

plt.suptitle(
    "Cumulative Opportunity — Toilets Reachable by Walking\n"
    "Public (n=975) + Semi-public (n=4,111)  |  NYC 2025",
    fontsize=14)
plt.tight_layout()

out = OUTPUT_DIR / "fig_cumulative_opportunity.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"图保存到: {out}")
