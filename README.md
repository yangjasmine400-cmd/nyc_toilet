# POI Spatial Sampling & Validation Tool

## Objective

This tool was developed to validate the completeness of POI (Point of Interest) datasets by cross-referencing them with the Google Places API. The core motivation was to quantify potential data gaps:

> "To what extent does our baseline dataset represent the ground truth of urban facilities?"

---

## Rationale

During my research, I observed significant variance between different POI data sources. To ensure data integrity, I implemented this script to systematically cross-check baseline data against live results from Google’s high-frequency updated database.

---

## How it works

The script executes a four-stage pipeline:

1. **Grid Sampling**  
   Discretizes geographic bounding boxes into micro-grid cells (Step: 0.002° ≈ 220m at mid-latitudes).

2. **API Querying**  
   Performs localized searchText queries at each grid centroid, utilizing locationBias to maximize the recall of neighboring venues.

3. **Data Alignment**  
   Normalizes diverse category labels to align with the Google Places schema.

4. **Stability Assessment**  
   Compares the density of baseline records vs. API records to generate a diagnostic verdict:
   - **STABLE**: Minimal variance detected.  
   - **MODERATE_INCREASE**: New POIs identified; update recommended.  
   - **CHECK_MORE**: Significant discrepancy; requires manual audit.
