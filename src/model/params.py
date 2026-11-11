"""Scalar parameters of the mechanism: central value, low, high, source.

The per-crop inputs (residue ratio, ME, protein, NDF) and the per-region
reserve shares carry their ranges in their own data files; here are the
scalars. The Monte Carlo draws every entry from a PERT distribution on
(low, central, high); the tornado moves one at a time to low and high.
A ``*_mult`` parameter multiplies a data column that has its own ranges
(its low and high are then taken from that file, see runs/sampling.py).
"""

from __future__ import annotations

# name: (central, low, high, unit, reference, note)
SCALARS = {
    "access_radius_km": (50.0, 10.0, 200.0, "km", "assumption: residue reaches animals of the same country at most this far away", "interpolated in log radius between the tabulated radii"),
    "reserve_other_mult": (1.0, 0.5, 1.5, "multiplier", "Smerald et al. 2023 regional shares (residue_reserve.csv)", "on the regional share of other off-field uses (fuel, construction, industry)"),
    "clearing_shift": (0.0, -1.0, 1.0, "months", "Sacks et al. 2010 planting periods, 60 days on average (production-weighted)", "added to the time from harvest to the crop's next sowing, when field residue nobody has eaten is gone: sets the stock standing at onset and the field part lost at sowing, of the stock at onset and of every later harvest"),
    "field_decay_factor": (1.0, 0.4, 1.2, "multiplier", "spread of the RUSLE2 site fits (Table 10.2)", "on every crop's RUSLE2 field decay coefficient"),
    "k_stored": (0.008, 0.003, 0.025, "per month", "Lemus 2020 (Mississippi State University Extension, Publication 2540): covered hay bales; low, bales kept in a barn; high, bales left uncovered on bare ground", "baled or stacked residue"),
    "baled_share": (0.5, 0.1, 0.9, "share", "assumption; range from current practice: stover grazed where it stands (Thierfelder et al. 2024, Zambia) to straw threshed at the homestead and stacked (Tesfaye and Chairatanayuth 2007, East Shoa)", "share of the additional residue that is baled or stacked"),
    "store_cap_years": (1.0, 1.0, 1.0, "years", "assumption: bale only for need", "baled or stacked residue carried out of a year is at most this many years of the additional residue the country's herds ate that year, in DM; the rest is not kept and counts as lost; inf: no cap; not sampled"),
    "acclim_rate": (0.0091, 0.0, 0.0091, "per C", "NRC 1981, beef chapter, adjustment for metabolic acclimatization: maintenance +0.91% per C of prior exposure below 20 C, on mean monthly temperatures", "counted on the drop of the annual mean temperature from the pre-catastrophe climate; low end: NRC 1981 notes that cattle probably never fully acclimatize to the extremes where seasons swing widely, and IPCC 2019 (Vol. 4 Ch. 10, Eq. 10.2) applies its cold adjustment only in the one to two months before acclimation"),
    "cold_wet": (0.0, 0.0, 1.0, "share", "NRC 1981 Table 17: dry pregnant beef cow, 500 kg, -25 C and 1.45% per C dry with low wind, -7.3 C and 2.04% per C in wet snow with a 10 mph (16 km/h) wind", "share of the cold season spent wet or in the wind; moves every lower critical temperature up by up to 13.7 C (-21 to -7.3 C for the cow) and every cost per degree by up to x1.41 (2.04/1.45)"),
    "shelter": (0.0, 0.0, 0.0, "share", "not used in the paper", "share of the herd housed: +5 C; a switch kept at 0, not sampled"),
    "ndf_cap_large": (1.35, 0.90, 1.50, "% of live weight per day", "Bateki and Dickhoefer 2019: 1.35 adopted for cattle accustomed to fibre-rich diets, on the authority of Mertens 1987, 2009, not fitted; they say it may still be low for the (sub)tropics. Mertens 1994 (Texas Anim Nutr Council, pp. 2-13): 1.2 +/- 0.1 is the maximum cows in mid to late lactation eat comfortably without losing milk; Mertens 2009 (WCDS 21:303-319) gives 1.25, SD 0.1, as the optimum, not a maximum. Low end 0.90: pregnant Hereford and Charolais cows on late-cut silages ate 0.92 to 1.18 (Jardstedt et al. 2018, lowest on the least digestible silage). High end 1.50: Ohio State Extension rule of thumb for beef cows on hay, 1.2 to 1.5, the higher value for better forage (Little and Wiseman 2024, ANR-0149). No source measures grain-fed cattle switched onto straw, the herds this ceiling binds", "capacity is the larger of this ceiling and ndf_baseline_factor x the pre-catastrophe NDF intake"),
    "ndf_cap_small": (1.50, 1.20, 2.00, "% of live weight per day", "Oliveira et al. 2020, PLoS ONE 15:e0244201: average NDF intake of growing hair sheep in tropical feedlots, 1.5% of BW (low goodness of fit). Low 1.2: Mertens 1994 (Proc. Texas Anim. Nutr. Council, pp. 2-13), 'Preliminary evaluation suggests that the NDF intake constraint of ... sheep (Mertens, 1973) is also approximately 1.2% BW/d'; Agriculture Victoria 2015 drought rule, maximum DM intake = 120/NDF% of live weight, i.e. 1.2% as NDF. High 2.0: Cannas, Thonney and Lunesu 2016 (Proc. 22nd DSANA Symposium, pp. 69-80) estimate the NDF intake at which rumen fill starts to restrict lactating ewes at 1.77% (90 kg) to 2.10% (45 kg), their own scaling of Mertens 1987 by adult weight^-0.25, and cite measured 1.76-2.28% (p. 74, secondary)", "one value for sheep and goats: fed the same straw diets, goats ate as much DM per kg W^0.75 as sheep or more (Brown and Johnson 1985; Wahed and Owen 1986; Domingue et al. 1991; abstracts); capacity is the larger of this ceiling and ndf_baseline_factor x the pre-catastrophe NDF intake"),
    "ndf_baseline_factor": (1.10, 1.00, 1.20, "multiplier", "judgement: no source gives the room above the pre-catastrophe NDF intake of herds already at the ceiling; Bateki and Dickhoefer 2019 write that 0.0135 kg/kg BW 'may still be low' for the (sub)tropics and their best model treats fill as a soft limit", "NDF capacity = max(absolute ceiling, factor x pre-catastrophe NDF intake) for every herd: a margin over each herd's own intake; at 1.10 it sets the limit of 543 of 657 stocked herds (496 above the ceiling, 47 just below it; central inputs)"),
    "requirement_mult": (1.0, 0.70, 1.30, "multiplier", "symmetric choice covering most published maintenance estimates, -10% to +40% of the feeding standards: lactating cows 0.09-0.14 v. 0.10 Mcal NEL (NASEM 2021 ch. 3); 0.62-0.64 v. AFRC 0.48 MJ ME/kg^0.75 (Corbett and Freer 2003, p. 617); warm climates +19-24% (Salah et al. 2014, p. 1443)", "on all net energy requirements"),
    "protein_mult": (1.0, 0.70, 1.30, "multiplier", "other direct estimates of maintenance 9-24% above the model values (Salah et al. 2014 cattle 4.35, sheep and goats 3.51; Souza et al. 2021 Saanen goats 3.8); SD of MP-use efficiency 0.10 around 0.66 (NASEM 2021 ch. 6)", "on metabolizable protein requirements"),
    "growth_mult": (1.0, 0.5, 2.0, "multiplier", "25% to 100% of the fastest growth the herd's fertility, mortality and age at first calving allow (FAO 2018); central: half", "on the maximum herd growth, which is half that fastest rate"),
    "grass_me_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "ME of grass and other roughages: one position for all regions and herds"),
    "grass_cp_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "crude protein of grass and other roughages: one position for all regions and herds"),
    "grass_ndf_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "NDF of grass and other roughages: one position for all regions and herds"),
    "conc_me_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "ME of grains and bran: one position for all regions and herds"),
    "conc_cp_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "crude protein of grains and bran: one position for all regions and herds"),
    "conc_ndf_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "NDF of grains and bran: one position for all regions and herds"),
    "coproduct_me_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "ME of co-products: one position for all regions and herds"),
    "coproduct_cp_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "crude protein of co-products: one position for all regions and herds"),
    "coproduct_ndf_pos": (0.0, -1.0, 1.0, "position", "lowest to highest Feedipedia table average of the materials", "NDF of co-products: one position for all regions and herds"),
    "urea_max_dm": (0.0, 0.0, 0.0, "share of diet DM", "EFSA 2012, doi:10.2903/j.efsa.2012.2624", "explicit supplement cap, at most 0.01; not sampled"),
    "rpr_mult": (1.0, 0.0, 0.0, "position", "ranges of residue_feed_values.csv", "common position of all crops' residue ratios within their ranges (-1 low, +1 high), the same for every crop"),
    "residue_me_mult": (1.0, 0.0, 0.0, "position", "ranges of residue_feed_values.csv", "same for residue ME"),
    "residue_cp_mult": (1.0, 0.0, 0.0, "position", "ranges of residue_feed_values.csv", "same for residue crude protein"),
    "residue_ndf_mult": (1.0, 0.0, 0.0, "position", "ranges of residue_feed_values.csv", "same for residue NDF"),
}
POSITION_PARAMS = ("rpr_mult", "residue_me_mult", "residue_cp_mult", "residue_ndf_mult")
NOT_SAMPLED = ("shelter", "urea_max_dm", "store_cap_years")

#: Cold: NRC 1981 values for animals at maintenance in dry, calm weather, as
#: (animals whose coldest pre-catastrophe month is below 5 C, animals in warmer places).
#: Lower critical temperature, C. Cattle and buffaloes: beef cow at maintenance -21 (Table 1, Webster 1974);
#: 8 mm hair at maintenance 7 (Table 1; NRC has no value for tropical breeds). Sheep and goats: fleece of
#: 100 mm at maintenance -3 and of 50 mm 9 (Table 1); NRC has no values for goats, treated as sheep.
COLD_LCT = {"large": (-21.0, 7.0), "small": (-3.0, 9.0)}
#: Cost per C below it, share of maintenance. Cattle and buffaloes: 1.45%, the dry pregnant cow of Table 17 dry
#: with low wind (a share of her whole heat production, applied to maintenance). Sheep and goats: Table 24, 50 kg,
#: at the insulation of the same fleece (wool about 0.007 C per kcal per m2 per day per cm): 0.07 for 100 mm
#: gives 0.7%, 0.035 for 50 mm gives 1.55% (between 1.8% at 0.03 and 1.3% at 0.04).
COLD_COEF = {"large": (0.0145, 0.0145), "small": (0.007, 0.0155)}
#: cold_wet = 1: the Table 17 cow in wet snow and a 10 mph wind, -7.3 C and 2.04% per C
COLD_WET_SHIFT = -7.3 - COLD_LCT["large"][0]
COLD_WET_FACTOR = 0.0204 / COLD_COEF["large"][0]

#: Straw treatment with urea or ammonia, as (low, central, high).
#: Digestibility gain, points (OM digestibility in Chenost, DM in Walli): central 10, "The average rise in digestibility is
#: 10 points" (Chenost and Kayouli 1997, section 5.3.1) and "approximately 10 percentage units" (Walli 2011,
#: p. 11); range 5 to 20, "improved digestibility (5-20% units)" (Idowu et al. 2026, abstract).
TREATMENT_DIGESTIBILITY_POINTS = (5.0, 10.0, 20.0)
#: ME per point of OM digestibility, MJ per kg DM: Feedipedia ME / OMD of wheat straw 6.8 / 48.4,
#: barley straw 6.5 / 47.5 and maize stover 6.9 / 49.0 (0.140, 0.137, 0.141; Feedipedia),
#: rounded to 0.14. Assumes ME in proportion to digestibility.
ME_PER_DIGESTIBILITY_POINT = 0.14
#: Crude protein gain, percentage points of DM: untreated 3-5% and treated 9-10% (Chenost and Kayouli 1997,
#: Table 8, after INRA 1988, ammoniated wheat and barley straw); untreated 4-6% and treated 9-12%
#: (Idowu et al. 2026, Table 1). The columns are not paired: central 5.5 is the gap between the midpoints,
#: 3 to 8 the widest gap Idowu's columns allow.
TREATMENT_CP_POINTS = (3.0, 5.5, 8.0)
#: Intake gain, as the divisor of the straw's NDF at the fill limit: central 1.3, "about 30 %" over 17 rice-straw
#: trials (Chenost and Kayouli 1997, section 5.3.3); range 1.1 to 1.4, "increased intake (10-40%)"
#: (Idowu et al. 2026, abstract). Walli 2011 gives 10 to 15%.
TREATMENT_INTAKE_GAIN = (1.1, 1.3, 1.4)
#: ranges sampled in the treated-straw Monte Carlo (PERT on low, central, high)
TREATMENT_RANGES = {"me_add": tuple(round(p * ME_PER_DIGESTIBILITY_POINT, 3) for p in TREATMENT_DIGESTIBILITY_POINTS),
                    "cp_add": TREATMENT_CP_POINTS, "ndf_intake_gain": TREATMENT_INTAKE_GAIN}
TREATMENT = {k: v[1] for k, v in TREATMENT_RANGES.items()}

#: Bedding sensitivity: share of residue production kept from the feed for bedding. Cereal straw and
#: stover used as bedding are about 10% of crop by-products in North America and Oceania, Western and Eastern Europe
#: (Wirsenius 2000, Fig. 3.67), little elsewhere; RUSS is included for its housed winters.
BEDDING_SHARE = 0.10
BEDDING_REGIONS = ("NA", "OC", "WE", "EE", "RUSS")


def defaults() -> dict:
    theta = {k: v[0] for k, v in SCALARS.items()}
    for k in POSITION_PARAMS:
        theta[k] = 0.0
    theta.update(onset="uniform", treatment=False, protein_model="mp",
                 usual_residue="ranked")   # "ranked": usual residue taken by ME/NDF; "mix": as a fixed mixture
    return theta
