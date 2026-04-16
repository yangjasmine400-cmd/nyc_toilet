import time
import requests
import pandas as pd
from pathlib import Path

API_KEY = ""  # Google Places API Key 
BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "v41_clean.csv"
out_path = BASE_DIR / "check_sampling_v2.xlsx"
progress_path = BASE_DIR / "check_sampling_v2_progress.csv"

areas = [
    {"area_name": "Midtown Manhattan",       "min_lat": 40.7520, "max_lat": 40.7640, "min_lon": -73.9950, "max_lon": -73.9770},
    {"area_name": "Downtown Brooklyn",       "min_lat": 40.6885, "max_lat": 40.6975, "min_lon": -74.0100, "max_lon": -73.9850},
    {"area_name": "St George Staten Island", "min_lat": 40.6400, "max_lat": 40.6485, "min_lon": -74.0820, "max_lon": -74.0700},
]

categories = ["coffee", "fast_food", "gas_station", "grocery", "pharmacy"]

query_map = {
    "coffee":      "coffee shop",
    "fast_food":   "fast food restaurant",
    "gas_station": "gas station",
    "grocery":     "grocery store",
    "pharmacy":    "pharmacy",
}

STEP = 0.002
RADIUS = 700.0

API_URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = "places.id,places.displayName.text,places.formattedAddress,places.location"


def frange(start, stop, step):
    result = []
    x = start
    while x <= stop + 1e-12:
        result.append(round(x, 6))
        x += step
    return result


def call_api(query, lat, lon):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": FIELDS,
    }
    body = {
        "textQuery": query,
        "pageSize": 20,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": RADIUS,
            }
        },
    }
    resp = requests.post(API_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def in_box(lat, lon, area):
    return (area["min_lat"] <= lat <= area["max_lat"] and
            area["min_lon"] <= lon <= area["max_lon"])


def do_sampling(area, category):
    q = query_map[category]
    seen = set()
    rows = []
    n_calls = 0

    for lat in frange(area["min_lat"], area["max_lat"], STEP):
        for lon in frange(area["min_lon"], area["max_lon"], STEP):
            n_calls += 1
            try:
                data = call_api(q, lat, lon)
            except Exception as e:
                print(f"api error at {lat},{lon}: {e}")
                continue

            for p in data.get("places", []):
                pid = p.get("id", "")
                loc = p.get("location", {})
                plat = loc.get("latitude")
                plon = loc.get("longitude")

                if not pid or plat is None or plon is None:
                    continue
                if pid in seen:
                    continue
                if not in_box(plat, plon, area):
                    continue

                seen.add(pid)
                rows.append({
                    "place_id": pid,
                    "name": (p.get("displayName") or {}).get("text", ""),
                    "address": p.get("formattedAddress", ""),
                    "lat": plat,
                    "lon": plon,
                    "category": category,
                    "area_name": area["area_name"],
                })

            time.sleep(0.08)

    return pd.DataFrame(rows), n_calls


def get_cat_col(df):
    for c in ["final_category_v4_2", "final_category_v4_1", "final_category_v4", "final_category"]:
        if c in df.columns:
            return c
    raise ValueError("no category column found")


def get_baseline(df, area, category, cat_col):
    tmp = df.copy()
    tmp[cat_col] = tmp[cat_col].astype(str).str.strip().str.lower()
    tmp = tmp[tmp[cat_col] == category]

    tmp["_lat"] = pd.to_numeric(tmp["lat"], errors="coerce")
    tmp["_lon"] = pd.to_numeric(tmp["lon"], errors="coerce")
    tmp = tmp[
        (tmp["_lat"] >= area["min_lat"]) & (tmp["_lat"] <= area["max_lat"]) &
        (tmp["_lon"] >= area["min_lon"]) & (tmp["_lon"] <= area["max_lon"])
    ]

    if "place_id" in tmp.columns:
        tmp["place_id"] = tmp["place_id"].astype(str).str.strip()
        tmp = tmp[tmp["place_id"] != ""].drop_duplicates("place_id")

    return tmp


def main():
    df = pd.read_csv(input_path)
    cat_col = get_cat_col(df)

    all_compare = []
    all_google = []

    for area in areas:
        print(f"\n--- {area['area_name']} ---")
        for cat in categories:
            print(f"  {cat}...", end=" ", flush=True)

            baseline_df = get_baseline(df, area, cat, cat_col)
            n_base = len(baseline_df)

            gdf, n_calls = do_sampling(area, cat)
            n_new = len(gdf)

            if n_base > 0:
                diff_pct = (n_new - n_base) / n_base
            elif n_new > 0:
                diff_pct = 1.0
            else:
                diff_pct = 0.0

            if diff_pct <= 0.10:
                verdict = "STABLE"
            elif diff_pct <= 0.20:
                verdict = "MODERATE_INCREASE"
            else:
                verdict = "CHECK_MORE"

            print(f"base={n_base} google={n_new} diff={diff_pct:.1%} -> {verdict}")

            all_compare.append({
                "area_name": area["area_name"],
                "category": cat,
                "baseline_count": n_base,
                "google_count": n_new,
                "diff": n_new - n_base,
                "diff_pct": diff_pct,
                "n_calls": n_calls,
                "verdict": verdict,
            })

            pd.DataFrame(all_compare).to_csv(progress_path, index=False)

            if not gdf.empty:
                all_google.append(gdf)

    compare_df = pd.DataFrame(all_compare)
    google_df = pd.concat(all_google, ignore_index=True) if all_google else pd.DataFrame()

    status = (compare_df.groupby("verdict").size()
              .reset_index(name="count")
              .sort_values("count", ascending=False))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        compare_df.to_excel(writer, index=False, sheet_name="compare")
        status.to_excel(writer, index=False, sheet_name="status")
        google_df.to_excel(writer, index=False, sheet_name="google_rows")

    print(f"\ndone -> {out_path}")


if __name__ == "__main__":
    main()