from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import KDTree
import pyproj

BASE_DIR    = Path(r"D:\nyc shp\nyc_toilet_esda")
SHP_PATH    = BASE_DIR / "public_shp" / "public_toilet.shp"
HEX_PATH    = BASE_DIR / "outputs" / "h3_grid_res8.gpkg"
GRAPH_PATH  = BASE_DIR / "nyc_walk_graph.pkl"
TRACT_PATH  = BASE_DIR / "tl_2025_36_tract" / "tl_2025_36_tract.shp"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

nyc_counties = ["005", "047", "061", "081", "085"]

# 读public toilet数据，分成两组：所有人 vs 只有轮椅可用的（Fully Accessible）
print("读public toilet...")
pub = gpd.read_file(str(SHP_PATH))
pub.columns = pub.columns.str.strip()
pub = pub.to_crs(epsg=4326)
pub["lat"] = pub.geometry.y
pub["lon"] = pub.geometry.x

# 只保留NYC五区范围内的
pub = pub[
    (pub["lat"] >= 40.47) & (pub["lat"] <= 40.92) &
    (pub["lon"] >= -74.26) & (pub["lon"] <= -73.70)
].copy()

print(f"公共厕所: {len(pub)}")
print(pub["accessibil"].value_counts())

fully   = pub[pub["accessibil"] == "Fully Accessible"].copy()
all_pub = pub.copy()

print(f"Fully Accessible: {len(fully)}")
print(f"全部: {len(all_pub)}")

# 读路网
print("\n读路网...")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

node_ids  = np.array(list(G.nodes()))
node_lats = np.array([G.nodes[n]["y"] for n in node_ids])
node_lons = np.array([G.nodes[n]["x"] for n in node_ids])
node_tree = KDTree(np.column_stack([node_lats, node_lons]))


def snap(lat, lon):
    _, idx = node_tree.query([lat, lon])
    return int(node_ids[idx])


hex_gdf = gpd.read_file(str(HEX_PATH))
print(f"H3格子: {len(hex_gdf)}")


def get_supply_nodes(df):
    coords = np.column_stack([df["lat"].values, df["lon"].values])
    tree   = KDTree(coords)
    nodes  = np.array([snap(r.lat, r.lon) for _, r in df.iterrows()])
    return tree, nodes


print("\n预计算路网节点...")
coords_all,  nodes_all  = get_supply_nodes(all_pub)
coords_full, nodes_full = get_supply_nodes(fully)


def calc_travel_time(hex_gdf, poi_tree, supply_nodes, label):
    print(f"\n跑 {label}...")
    results = []
    K = 15
    total = len(hex_gdf)

    for i, (_, row) in enumerate(hex_gdf.iterrows()):
        if i % 100 == 0:
            print(f"  {i}/{total}  {i/total*100:.1f}%")

        clat = row["centroid_lat"]
        clng = row["centroid_lon"]
        orig = snap(clat, clng)

        k = min(K, len(supply_nodes))
        _, idx = poi_tree.query([clat, clng], k=k)
        dests  = list(set(supply_nodes[idx].tolist()))

        min_time = np.nan
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, orig, cutoff=3600, weight="travel_time"
            )
            times = [lengths[d] for d in dests if d in lengths]
            if times:
                min_time = min(times)
        except Exception:
            pass

        results.append({
            "h3_index":    row["h3_index"],
            "min_time_min": round(min_time / 60, 2) if not np.isnan(min_time) else np.nan
        })

    return pd.DataFrame(results)


df_all  = calc_travel_time(hex_gdf, KDTree(np.column_stack([all_pub["lat"].values, all_pub["lon"].values])), nodes_all,  "所有公厕")
df_full = calc_travel_time(hex_gdf, KDTree(np.column_stack([fully["lat"].values,   fully["lon"].values])), nodes_full, "Fully Accessible")

# 合并 + 差值
hex_result = hex_gdf.copy()
hex_result = hex_result.merge(df_all.rename(columns={"min_time_min":  "tt_all"}),        on="h3_index", how="left")
hex_result = hex_result.merge(df_full.rename(columns={"min_time_min": "tt_accessible"}), on="h3_index", how="left")
hex_result["tt_diff"] = hex_result["tt_accessible"] - hex_result["tt_all"]

hex_result.to_file(str(OUTPUT_DIR / "hex_accessibility_disparity.gpkg"), driver="GPKG")

v1   = hex_result["tt_all"].dropna()
v2   = hex_result["tt_accessible"].dropna()
diff = hex_result["tt_diff"].dropna()

print(f"\n普通用户  中位数 {v1.median():.1f}  均值 {v1.mean():.1f}")
print(f"轮椅用户  中位数 {v2.median():.1f}  均值 {v2.mean():.1f}")
print(f"差值      中位数 {diff.median():.1f}")
print(f"  >5min extra:  {(diff>5).sum()} 个 ({(diff>5).mean()*100:.1f}%)")
print(f"  >10min extra: {(diff>10).sum()} 个 ({(diff>10).mean()*100:.1f}%)")

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


vmax_shared = min(max(int(v1.quantile(0.95)), int(v2.quantile(0.95))) + 2, 30)
vmax_diff   = min(int(diff.quantile(0.95)) + 2, 20)

fig, axes = plt.subplots(1, 3, figsize=(33, 16))
hex_plot  = hex_result.to_crs(epsg=3857)

draw_base(axes[0])
hex_plot[hex_plot["tt_all"].notna()].plot(
    column="tt_all", ax=axes[0], cmap=cmap_red, vmin=0, vmax=vmax_shared,
    legend=True, legend_kwds={"label": "Walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[0].set_title(f"All users — nearest public toilet\nMedian: {v1.median():.1f} min  |  Mean: {v1.mean():.1f} min", fontsize=10)

draw_base(axes[1])
hex_plot[hex_plot["tt_accessible"].notna()].plot(
    column="tt_accessible", ax=axes[1], cmap=cmap_red, vmin=0, vmax=vmax_shared,
    legend=True, legend_kwds={"label": "Walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[1].set_title(f"Wheelchair users — nearest fully accessible toilet\nMedian: {v2.median():.1f} min  |  Mean: {v2.mean():.1f} min", fontsize=10)

draw_base(axes[2])
hex_plot[hex_plot["tt_diff"].notna()].plot(
    column="tt_diff", ax=axes[2], cmap=cmap_diff, vmin=0, vmax=vmax_diff,
    legend=True, legend_kwds={"label": "Extra walking time (min)", "shrink": 0.55, "pad": 0.02, "aspect": 30},
    linewidth=0.05, edgecolor="none", zorder=2)
axes[2].set_title(
    f"Extra walking time for wheelchair users\n"
    f"Median: {diff.median():.1f} min  |  >5min: {(diff>5).mean()*100:.1f}%  |  >10min: {(diff>10).mean()*100:.1f}%",
    fontsize=10)

plt.suptitle(
    "Accessibility Disparity — Public Toilets Only\n"
    "All users vs. Wheelchair users (Fully Accessible only)  |  NYC 2025",
    fontsize=13)
plt.tight_layout()

out = OUTPUT_DIR / "fig_accessibility_disparity.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"图保存到: {out}")
