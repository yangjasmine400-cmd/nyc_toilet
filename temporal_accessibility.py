from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import KDTree
import pyproj

# 路径
BASE_DIR    = Path(r"D:\nyc shp\nyc_toilet_esda")
CSV_PATH    = BASE_DIR / "baseline_final.csv"
PUBLIC_PATH = BASE_DIR / "public_toilet_975.csv"
HEX_PATH    = BASE_DIR / "outputs" / "h3_grid_res8.gpkg"
GRAPH_PATH  = BASE_DIR / "nyc_walk_graph.pkl"
TRACT_PATH  = BASE_DIR / "tl_2025_36_tract" / "tl_2025_36_tract.shp"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

nyc_counties = ["005", "047", "061", "081", "085"]

# 时段字段说明：
# weekday_day_available   工作日白天开放
# weekday_night_available 工作日夜间开放
# weekend_day_available   周末白天
# weekend_night_available 周末夜间
# 这里只跑工作日白天 vs 工作日夜间


def build_node_index(G):
    node_ids  = np.array(list(G.nodes()))
    node_lats = np.array([G.nodes[n]["y"] for n in node_ids])
    node_lons = np.array([G.nodes[n]["x"] for n in node_ids])
    tree = KDTree(np.column_stack([node_lats, node_lons]))
    return tree, node_ids


def snap(tree, node_ids, lat, lon):
    _, idx = tree.query([lat, lon])
    return int(node_ids[idx])


# 读路网
print("读路网...")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
print(f"节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")

node_tree, node_ids = build_node_index(G)

# 读数据，按时段筛供给点
print("按时段筛供给点...")

df = pd.read_csv(CSV_PATH)
df.columns = (df.columns.astype(str).str.strip()
              .str.replace("\ufeff", "", regex=False)
              .str.replace("ï»¿", "", regex=False))

time_cols = ["weekday_day_available", "weekday_night_available",
             "weekend_day_available", "weekend_night_available"]
for col in time_cols:
    if col not in df.columns:
        print(f"警告：找不到列 {col}，现有列: {[c for c in df.columns if 'avail' in c.lower()]}")

for col in time_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

day_semi   = df[df["weekday_day_available"] == 1][["lat", "lon"]].dropna().copy()
day_semi.columns = ["lat", "lng"]

night_semi = df[df["weekday_night_available"] == 1][["lat", "lon"]].dropna().copy()
night_semi.columns = ["lat", "lng"]

# public toilet：白天全部算，夜间也全部算（没有细化时段字段，保守处理）
pub = pd.read_csv(PUBLIC_PATH)
pub.columns = pub.columns.str.strip()
pub_supply = pub[["lat", "lon"]].dropna().copy()
pub_supply.columns = ["lat", "lng"]

day_supply   = pd.concat([day_semi,   pub_supply], ignore_index=True).dropna()
night_supply = pd.concat([night_semi, pub_supply], ignore_index=True).dropna()

print(f"白天供给点: {len(day_supply)}")
print(f"夜间供给点: {len(night_supply)}  (减少了 {len(day_supply) - len(night_supply)} 个)")

# 读H3网格
hex_gdf = gpd.read_file(str(HEX_PATH))
print(f"六边形格子: {len(hex_gdf)}")


def calc_tt(hex_gdf, supply_df, G, node_tree, node_ids, label, K=15):
    print(f"\n跑 {label}...")

    supply_lats  = supply_df["lat"].values
    supply_lngs  = supply_df["lng"].values
    supply_nodes = np.array([
        snap(node_tree, node_ids, lat, lng)
        for lat, lng in zip(supply_lats, supply_lngs)
    ])

    poi_tree = KDTree(np.column_stack([supply_lats, supply_lngs]))

    results = []
    total   = len(hex_gdf)

    for i, (_, row) in enumerate(hex_gdf.iterrows()):
        if i % 200 == 0:
            print(f"  {i}/{total}  {i/total*100:.1f}%")

        clat = row["centroid_lat"]
        clng = row["centroid_lon"]
        orig = snap(node_tree, node_ids, clat, clng)

        k_actual   = min(K, len(supply_df))
        _, idx     = poi_tree.query([clat, clng], k=k_actual)
        dest_nodes = list(set(supply_nodes[idx].tolist()))

        min_time = np.nan
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, orig, cutoff=1800, weight="travel_time"
            )
            times = [lengths[d] for d in dest_nodes if d in lengths]
            if times:
                min_time = min(times)
        except Exception:
            pass

        results.append({
            "h3_index": row["h3_index"],
            "tt_min": round(min_time / 60, 2) if not np.isnan(min_time) else np.nan
        })

    return pd.DataFrame(results)


# 跑两遍
df_day   = calc_tt(hex_gdf, day_supply,   G, node_tree, node_ids, "白天")
df_night = calc_tt(hex_gdf, night_supply, G, node_tree, node_ids, "夜间")

# 合并 + 算差值
hex_result = hex_gdf.copy()
hex_result = hex_result.merge(df_day.rename(columns={"tt_min": "tt_day"}),   on="h3_index", how="left")
hex_result = hex_result.merge(df_night.rename(columns={"tt_min": "tt_night"}), on="h3_index", how="left")
hex_result["tt_diff"] = hex_result["tt_night"] - hex_result["tt_day"]

hex_result.to_file(str(OUTPUT_DIR / "hex_travel_time_temporal.gpkg"), driver="GPKG")

v_day   = hex_result["tt_day"].dropna()
v_night = hex_result["tt_night"].dropna()
v_diff  = hex_result["tt_diff"].dropna()

print(f"\n白天  中位数 {v_day.median():.1f}  均值 {v_day.mean():.1f}")
print(f"夜间  中位数 {v_night.median():.1f}  均值 {v_night.mean():.1f}")
print(f"差值  中位数 {v_diff.median():.1f}  >5min: {(v_diff>5).mean()*100:.1f}%  >10min: {(v_diff>10).mean()*100:.1f}%")

# 出图
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

cmap_red  = LinearSegmentedColormap.from_list("YlOrRd", [
    "#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"
])
cmap_diff = LinearSegmentedColormap.from_list("WtRd", [
    "#ffffff", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"
])


def draw_base(ax):
    nyc_outline.to_crs(epsg=3857).plot(ax=ax, color="#f0f0f0", zorder=1)
    borough_bounds.to_crs(epsg=3857).boundary.plot(
        ax=ax, linewidth=0.8, color="white", alpha=0.6, zorder=4)
    for name, (lon, lat) in b_labels.items():
        x, y = transformer.transform(lon, lat)
        ax.text(x, y, name, fontsize=7, color="#333", fontweight="bold",
                ha="center", va="center", zorder=5)
    ax.set_axis_off()


vmax_tt   = min(max(int(v_day.quantile(0.95)), int(v_night.quantile(0.95))) + 2, 20)
vmax_diff = min(int(v_diff.quantile(0.95)) + 2, 20)

fig, axes = plt.subplots(1, 3, figsize=(33, 16))
hex_plot  = hex_result.to_crs(epsg=3857)

# 白天
draw_base(axes[0])
hex_plot[hex_plot["tt_day"].notna()].plot(
    column="tt_day", ax=axes[0], cmap=cmap_red, vmin=0, vmax=vmax_tt,
    legend=True, legend_kwds={"label": "Walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[0].set_title(
    f"Daytime — nearest open toilet\n"
    f"Supply: {len(day_supply)}  |  Median: {v_day.median():.1f} min  |  Mean: {v_day.mean():.1f} min",
    fontsize=10)

# 夜间
draw_base(axes[1])
hex_plot[hex_plot["tt_night"].notna()].plot(
    column="tt_night", ax=axes[1], cmap=cmap_red, vmin=0, vmax=vmax_tt,
    legend=True, legend_kwds={"label": "Walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[1].set_title(
    f"Nighttime — nearest open toilet\n"
    f"Supply: {len(night_supply)}  |  Median: {v_night.median():.1f} min  |  Mean: {v_night.mean():.1f} min",
    fontsize=10)

# 差值
draw_base(axes[2])
pos_diff = hex_plot[hex_plot["tt_diff"] > 0]
if len(pos_diff) > 0:
    pos_diff.plot(
        column="tt_diff", ax=axes[2], cmap=cmap_diff, vmin=0, vmax=vmax_diff,
        legend=True, legend_kwds={"label": "Extra time at night (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
        linewidth=0.05, edgecolor="none", zorder=2)

zero_diff = hex_plot[hex_plot["tt_diff"] <= 0]
if len(zero_diff) > 0:
    zero_diff.plot(ax=axes[2], color="#e8e8e8", linewidth=0.05, edgecolor="none", zorder=2)

axes[2].set_title(
    f"Night vs. Day — extra walking time\n"
    f"Median: {v_diff.median():.1f} min  |  >5min: {(v_diff>5).mean()*100:.1f}%  |  >10min: {(v_diff>10).mean()*100:.1f}%",
    fontsize=10)
axes[2].legend(
    handles=[mpatches.Patch(color="#e8e8e8", label="No change at night")],
    loc="lower left", fontsize=8, framealpha=0.85, edgecolor="none")

plt.suptitle(
    "Temporal Accessibility — Daytime vs. Nighttime\n"
    "Semi-public (time-restricted) + Public  |  NYC 2025",
    fontsize=14)
plt.tight_layout()

out = OUTPUT_DIR / "fig_travel_time_temporal.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"图保存到: {out}")
