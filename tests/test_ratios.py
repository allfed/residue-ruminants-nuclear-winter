import numpy as np

from src import inputs as I
from src.model import ratios as R


def test_ratio_falls_with_yield_and_stays_within_the_floor():
    crops, iso, fn, area, bounds = R._tables()
    for k, sub in fn.groupby("crop_code"):
        lo, hi, _ = bounds[k]
        y = np.linspace(lo, hi, 50)
        for f in sub.itertuples():
            r = R.evaluate(f.form, f.p1, f.p2, y, f.dm_product)
            assert np.all(np.diff(r) <= 1e-12), (k, f.source)
    c = R.central(150)
    j = crops.index("MAIZ")
    floor = np.median([R.evaluate(f.form, f.p1, f.p2, bounds["MAIZ"][0], f.dm_product)
                       for f in fn[fn.crop_code == "MAIZ"].itertuples()])
    assert c[:, j, :].max() <= floor + 1e-9


def test_central_ratio_near_reference_at_world_yield():
    crops, iso, fn, area, bounds = R._tables()
    fv = I.read_data("residue_feed_values.csv").set_index("crop_code")
    for k in ("WHEA", "MAIZ", "RICE", "BARL"):
        sub = fn[fn.crop_code == k]
        world = bounds[k][2]
        r = np.median([R.evaluate(f.form, f.p1, f.p2, world, f.dm_product) for f in sub.itertuples()])
        assert abs(r / fv.loc[k, "rpr"] - 1) < 0.1, (k, r, fv.loc[k, "rpr"])


def test_hold_keeps_prewar_ratio_and_range_brackets_central():
    c = R.central(47, hold=True)
    assert np.allclose(c, c[:, :, :1])
    assert np.allclose(R.central(47)[:, :, 0], c[:, :, 0])
    c, lo, hi = R._ratios(47, False)
    assert np.all(lo <= c + 1e-12) and np.all(hi >= c - 1e-12) and np.all(lo >= 0)
    assert np.allclose(R.positioned(47, -1.0), lo) and np.allclose(R.positioned(47, 1.0), hi)
    assert np.allclose(R.positioned(47, 0.0), c)
