# Inputs

Everything the runs read. Countries are ISO3 codes (169 countries; SWT is Eswatini), scenarios are 16, 47 and
150 Tg of soot, and year 0 is the pre-catastrophe year where a file has years. Calories in columns and arrays named `tcal` are in
10^12 kcal. The paper's Methods and
Supplementary Material give the sources and assumptions behind each file; the comment lines (#) at the top of
some files describe their columns.

## drivers/: upstream data, used as given

| File | Content |
|---|---|
| `crops_{16,47,150}Tg.npz` | crop production per country, crop and year 0 to 10, tonnes and Tcal, from the crop-model simulations of Blouin et al. (2025), no relocation |
| `grass_{16,47,150}Tg.csv` | pasture production per country, years 1 to 10, as a ratio to the pre-catastrophe level (Xia et al., 2022) |
| `animal_baseline.csv` | milk and meat per country, herd and product, tonnes and Tcal, FAOSTAT 2019 to 2021 |
| `countries.csv` | the countries and their regions |

## Built inputs

| File | Content |
|---|---|
| `ruminant_herds.csv` | herds per country (heads, live weight) and their energy and protein requirements |
| `stream_properties.csv` | ME, crude protein and fiber of the grass and concentrate streams per region and herd |
| `protein_streams.csv`, `protein_residues.csv` | rumen-degradable share and digestible undegraded share of the protein of each stream and crop residue |
| `residue_feed_values.csv` | per crop: residue-to-product ratio, ME, crude protein and NDF of the residue, with ranges |
| `residue_ratio_functions.csv` | the residue-to-product ratio as a function of yield, per crop |
| `crop_yield.csv` | pre-catastrophe harvested area, production and yield per country and crop (SPAM 2020) |
| `residue_reserve.csv` | shares of residue left on the field and taken by other uses, per region |
| `residue_calendar.csv` | share of each crop's production harvested in each month, per country |
| `residue_clearing.npz` | how long the residue may stay on the field before the crop is sown again |
| `residue_decay_crops.csv` | decay rate of each crop's residue left on the field (RUSLE2) |
| `decay_temperature.npz` | monthly air temperature where each crop's residue lies, per scenario and year |
| `cold_deficit.npz` | cold stress: extra maintenance energy per country, scenario, year and animal group |
| `grass_climate.csv` | share of each herd's animals on temperate grass |
| `grain_feed.csv` | national upper bound on the grain and bran that rations may give ruminants |
| `residue_access.csv` | reach: share of a country's residue within 10 to 200 km of animals that can eat it, per scenario, year and demand ratio (method in `src/reach/access.py`) |
| `residue_access_crops.npz` | the same per crop, on the pre-catastrophe grid |
| `benchmark_smerald.csv` | cereal residue per country from Smerald et al. (2023), for comparison with ours |
| `population.csv` | population per country, UN WPP 2024, 1 July 2020 |

## map_context/: for drawing maps only

`countries.geojson` (Natural Earth 1:10m Admin 0 Countries, public domain, simplified for plotting) and
`maps_05deg.npz` (0.5-degree residue, demand and 50 km reach shares, pre-catastrophe and per scenario and year).
