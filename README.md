# NYC Public Restroom Accessibility Study

A spatial analysis of public and semi-public restroom access across 
New York City's five boroughs, conducted as part of research at 
Auburn University's EcoTrans Lab.

## Research Question

Public restrooms are basic urban infrastructure — but their 
distribution across neighborhoods is uneven in ways that rarely 
get measured. This project asks: who bears the cost of inadequate 
restroom access, and does that burden fall disproportionately on 
specific populations or times of day?

## Data Sources

- **Public restrooms**: NYC Open Data (deduplicated from 6,775 
  stall-level records to 975 unique facilities)
- **Semi-public facilities**: Google Places API — cafes, fast food, 
  pharmacies, gas stations, supermarkets (n=4,111)
- **Street network**: OpenStreetMap pedestrian network via OSMnx 
  (936,335 nodes, 2,398,410 edges)
- **Census geography**: 2025 NYC Census Tracts (TIGER/Line)
- **Spatial grid**: H3 Resolution 8 hexagonal grid (1,653 cells, 
  ~400–500m scale)

## Analysis Modules

### 1. `sampling_check.py` — Data Validation
Cross-validates the semi-public POI dataset against live Google 
Places API results to quantify coverage gaps before analysis.

### 2. `cumulative_opportunity.py` — Baseline Accessibility
Counts reachable facilities within 5, 10, and 15-minute walking 
circles for each grid cell. Finds that 54.4% of cells have zero 
restrooms within a 5-minute walk.

### 3. `temporal_accessibility.py` — Time-of-Day Analysis
Compares daytime vs. nighttime accessibility using facility 
opening hours. At night, 641 of 975 public restrooms close, 
and the share of cells with >10-minute walk times rises from 
15.7% to 23.3%.

### 4. `accessibility_disparity.py` — Wheelchair Accessibility Gap
Compares travel times for general users vs. wheelchair users 
(restricted to the 577 fully accessible public facilities). 
In 21.3% of cells, wheelchair users face more than 5 extra 
minutes of walking.

### 5. `bathroom_resilience.py` — Network Redundancy
Measures the penalty of being turned away from the nearest 
facility — the extra time required to reach the second-nearest 
option. Median penalty is 1.0 minute citywide, but 14.7% of 
cells face a penalty over 5 minutes, concentrated in the Bronx, 
eastern Queens, and Staten Island.

## Key Finding

The same neighborhoods — Bronx northeast, eastern Queens, 
Staten Island — appear as deficit areas across every dimension: 
total supply, daytime accessibility, nighttime coverage, 
wheelchair access, and network redundancy. This spatial overlap 
suggests a systemic pattern rather than isolated gaps.

## Tech Stack

Python · GeoPandas · NetworkX (Dijkstra) · PySAL (LISA / 
Getis-Ord G*) · H3 · Matplotlib · Google Places API · OSMnx

## Status

Analysis complete. Visualization and write-up in progress.
