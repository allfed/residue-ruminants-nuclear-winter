"""Pre-catastrophe calibration of the herds' protein requirements and intake caps.

For every stocked herd, on the inputs of the 47 Tg scenario, the maintenance and production protein
requirements and the NDF intake cap as given by the requirement equations (raw) and as the model uses
them (fitted): protein requirements scaled down where the pre-catastrophe ration's metabolizable protein
falls short of them, intake cap raised where the pre-catastrophe NDF intake, with its margin, exceeds it.

    python -m src.runs.calibration

Writes results/calibration_check.csv (one row per country and stocked herd) and
results/calibration_summary.csv (per herd type: herds adjusted and mean raw and fitted values).
"""
import pandas as pd
from src import inputs as I
from src.model.prepare import prepare


def calibration():
    p=prepare(47)
    rows=[]
    for i,c in enumerate(p.iso3):
        for h,hd in enumerate(p.herds):
            if p.n0[i,h]<=0: continue
            rows.append(dict(iso3=c,herd=hd,heads=p.n0[i,h],raw_mp_m=p.extras["raw_mp_m"][i,h],
                fitted_mp_m=p.mp_m[i,h],raw_mp_p=p.extras["raw_mp_p"][i,h],fitted_mp_p=p.mp_p[i,h],
                inferred_baseline_mp=p.extras["mp_head0"][i,h],raw_ndf_cap=p.extras["raw_ndf_cap"][i,h],
                fitted_ndf_cap=p.ndf_cap[i,h],protein_adjusted=bool(p.extras["cp_short"][i,h]),
                intake_adjusted=bool(p.extras["ndf_floor"][i,h])))
    df=pd.DataFrame(rows)
    df.to_csv(I.RESULTS/"calibration_check.csv",index=False)
    df.groupby("herd").agg(national_herds=("heads","size"),protein_adjusted=("protein_adjusted","sum"),
        intake_adjusted=("intake_adjusted","sum"),raw_mp_m=("raw_mp_m","mean"),fitted_mp_m=("fitted_mp_m","mean"),
        raw_mp_p=("raw_mp_p","mean"),fitted_mp_p=("fitted_mp_p","mean"),
        raw_ndf_cap=("raw_ndf_cap","mean"),fitted_ndf_cap=("fitted_ndf_cap","mean")).to_csv(I.RESULTS/"calibration_summary.csv")
    print("Raw and fitted calibration table complete",flush=True)


if __name__=="__main__":
    calibration()
