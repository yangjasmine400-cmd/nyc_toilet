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

BASE_DIR    = Path(r"D:\nyc shp\nyc_toilet_esda")
CSV_PATH    = BASE_DIR / "baseline_final.csv"
PUBLIC_PATH = BASE_DIR / "public_toilet_975.csv"
HEX_PATH    = BASE_DIR / "outputs" / "h3_grid_res8.gpkg"
GRAPH_PATH  = BASE_DIR / "nyc_walk_graph.pkl"
TRACT_PATH  = BASE_DIR / "tl_2025_36_tract" / "tl_2025_36_tract.shp"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

nyc_counties = ["005", "047", "061", "081", "085"]


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

# 读供给点（public + semi-public全部合并）
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

# 预计算供给点对应的路网节点
print("预计算供给点路网节点（要跑一会儿）...")
supply_node_ids = np.array([
    snap(node_tree, node_ids, r.lat, r.lng)
    for _, r in all_supply.iterrows()
])
supply_node_set = list(set(supply_node_ids.tolist()))
print(f"去重后供给节点: {len(supply_node_set)}")

poi_coords = np.column_stack([all_supply["lat"], all_supply["lng"]])
poi_tree   = KDTree(poi_coords)

# 读H3
hex_gdf = gpd.read_file(str(HEX_PATH))
print(f"格子: {len(hex_gdf)}")

# 核心计算：最近 + 第二近 + penalty
# 思路：Dijkstra跑出到所有候选供给节点的距离，排序取前两个
# penalty = 第二近 - 最近，表示被拒后要多走多远
print("\n开始算 Bathroom Resilience（预计30-90分钟）...")

results = []
total = len(hex_gdf)
K = 20  # 多取一些候选点，保证能找到第二近

for i, (_, row) in enumerate(hex_gdf.iterrows()):
    if i % 100 == 0:
        print(f"  {i}/{total}  {i/total*100:.1f}%")

    clat = row["centroid_lat"]
    clng = row["centroid_lon"]
    orig = snap(node_tree, node_ids, clat, clng)

    k_actual   = min(K, len(all_supply))
    _, idx     = poi_tree.query([clat, clng], k=k_actual)
    dest_nodes = list(set(supply_node_ids[idx].tolist()))

    nearest        = np.nan
    second_nearest = np.nan
    penalty        = np.nan

    try:
        lengths = nx.single_source_dijkstra_path_length(
            G, orig, cutoff=3600, weight="travel_time"
        )
        times = sorted([lengths[d] for d in dest_nodes if d in lengths])

        if len(times) >= 1:
            nearest = times[0]
        if len(times) >= 2:
            second_nearest = times[1]
            penalty = second_nearest - nearest

    except Exception:
        pass

    results.append({
        "h3_index":    row["h3_index"],
        "nearest_min": round(nearest / 60, 2)        if not np.isnan(nearest)        else np.nan,
        "second_min":  round(second_nearest / 60, 2) if not np.isnan(second_nearest) else np.nan,
        "penalty_min": round(penalty / 60, 2)         if not np.isnan(penalty)         else np.nan,
    })

print("算完了")

res_df     = pd.DataFrame(results)
hex_result = hex_gdf.merge(res_df, on="h3_index", how="left")
hex_result.to_file(str(OUTPUT_DIR / "hex_resilience.gpkg"), driver="GPKG")

v_nearest = hex_result["nearest_min"].dropna()
v_second  = hex_result["second_min"].dropna()
v_penalty = hex_result["penalty_min"].dropna()

print(f"\n最近厕所  中位数 {v_nearest.median():.1f}  均值 {v_nearest.mean():.1f}")
print(f"第二近    中位数 {v_second.median():.1f}  均值 {v_second.mean():.1f}")
print(f"Penalty   中位数 {v_penalty.median():.1f}  均值 {v_penalty.mean():.1f}")
print(f"  >5min:  {(v_penalty>5).sum()} 个 ({(v_penalty>5).mean()*100:.1f}%)")
print(f"  >10min: {(v_penalty>10).sum()} 个 ({(v_penalty>10).mean()*100:.1f}%)")
print(f"  >15min: {(v_penalty>15).sum()} 个 ({(v_penalty>15).mean()*100:.1f}%)")

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

cmap_red     = LinearSegmentedColormap.from_list("YlOrRd", [
    "#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"
])
cmap_penalty = LinearSegmentedColormap.from_list("WtRd", [
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


vmax_tt      = min(max(int(v_nearest.quantile(0.95)), int(v_second.quantile(0.95))) + 2, 20)
vmax_penalty = min(int(v_penalty.quantile(0.95)) + 2, 20)

fig, axes = plt.subplots(1, 3, figsize=(33, 16))
hex_plot  = hex_result.to_crs(epsg=3857)

draw_base(axes[0])
hex_plot[hex_plot["nearest_min"].notna()].plot(
    column="nearest_min", ax=axes[0], cmap=cmap_red, vmin=0, vmax=vmax_tt,
    legend=True, legend_kwds={"label": "Walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[0].set_title(f"Nearest toilet\nMedian: {v_nearest.median():.1f} min  |  Mean: {v_nearest.mean():.1f} min", fontsize=11)

draw_base(axes[1])
hex_plot[hex_plot["second_min"].notna()].plot(
    column="second_min", ax=axes[1], cmap=cmap_red, vmin=0, vmax=vmax_tt,
    legend=True, legend_kwds={"label": "Walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[1].set_title(f"Second nearest toilet\nMedian: {v_second.median():.1f} min  |  Mean: {v_second.mean():.1f} min", fontsize=11)

draw_base(axes[2])
hex_plot[hex_plot["penalty_min"].notna()].plot(
    column="penalty_min", ax=axes[2], cmap=cmap_penalty, vmin=0, vmax=vmax_penalty,
    legend=True, legend_kwds={"label": "Penalty if rejected (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[2].set_title(
    f"Penalty if first toilet is unavailable\n"
    f"Median: {v_penalty.median():.1f} min  |  >5min: {(v_penalty>5).mean()*100:.1f}%  |  >10min: {(v_penalty>10).mean()*100:.1f}%",
    fontsize=11)

plt.suptitle(
    "Bathroom Resilience — Cost of Being Rejected\n"
    "Public (n=975) + Semi-public (n=4,111)  |  NYC 2025",
    fontsize=14)
plt.tight_layout()

out = OUTPUT_DIR / "fig_bathroom_resilience.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"图保存到: {out}")
