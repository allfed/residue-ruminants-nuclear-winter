"""A fixed-coefficient, ration-level MP approximation and explicit urea.

NRC 2001 pp. 58, 65: microbial CP is limited by 0.13 times discounted
TDN and RDP/1.18; absorbed microbial true protein is 0.64 times microbial
CP. Feed RUP digestibility is feed-specific (Table 15-2a). This is not the
complete NRC model: TDN is approximated from ME (DE = 4.409 Mcal/kg TDN,
Eq. 2-1; ME = 0.82 DE, p. 327), passage is fixed at 0.06/h, and endogenous
protein is not counted as supply, as in NASEM 2021 ch. 6; the maintenance
requirement is Wilkerson et al. 1993's empirical regression, which absorbs it.
"""
import numpy as np

MICROBIAL_MP = 0.64
MICROBIAL_CP_PER_TDN = 0.13
RDP_PER_MICROBIAL_CP = 1.18
UREA_CP = 0.46 * 6.25
# EFSA 2012: 1% complete-feed DM, equivalent to 0.3 g/kg body weight/day;
# EU 839/2012 annex: at most 30% dietary N from urea-N, with adaptation and
# easily digestible carbohydrate. The ceiling applies to supplemental urea only:
# retained treatment N is not counted, since the treatment trials fed treated
# straw as the main or sole roughage.
MAX_ADDED_N_SHARE = 0.30
UREA_PER_KG_BW = 0.0003


def coefficients(prep, t):
    """MP inequalities after eliminating urea: coeff @ feed_DM + offset >= need.

    Urea upper bounds are all affine in the diet; requiring each resulting
    nitrogen inequality enforces the smallest cap. Urea remains separately
    recorded as the minimum dose needed for the selected production level.
    """
    cp = prep.cp[:, :, t]
    if prep.protein_model == "cp_proxy":
        energy = nitrogen = MICROBIAL_MP * cp
        ucoef = MICROBIAL_MP * UREA_CP
    else:
        energy = MICROBIAL_MP * MICROBIAL_CP_PER_TDN * prep.tdn[:, :, t] + prep.drup[:, :, t]
        nitrogen = MICROBIAL_MP / RDP_PER_MICROBIAL_CP * prep.rdp[:, :, t] + prep.drup[:, :, t]
        ucoef = MICROBIAL_MP / RDP_PER_MICROBIAL_CP * UREA_CP
    coeff, offset = [], []
    zero = np.zeros(prep.n0.shape)
    if prep.protein_model != "cp_proxy":
        coeff.append(energy); offset.append(zero)
    if prep.urea_max_dm > 0:
        alpha = prep.urea_max_dm / (1 - prep.urea_max_dm)
        nitrogen_room = MAX_ADDED_N_SHARE * cp / ((1-MAX_ADDED_N_SHARE) * UREA_CP)
        coeff.extend([nitrogen + ucoef * alpha, nitrogen + ucoef * nitrogen_room, nitrogen])
        offset.extend([zero, zero, ucoef * UREA_PER_KG_BW * prep.extras["live_weight"]])
    else:
        coeff.append(nitrogen); offset.append(zero)
    return np.stack(coeff, axis=-2), np.stack(offset, axis=-1)


def rows(model, tdn, rdp, drup, cp):
    """MP supplied per kg DM of each feed, one row per limit (..., L, F), and per kg of
    supplemental urea (L,), for rations with urea as an explicit variable (ration.py).
    NRC: microbial CP is the lesser of 0.13 TDN and RDP/1.18, 64% of it absorbed."""
    if model == "cp_proxy":
        return (MICROBIAL_MP * cp)[..., None, :], np.array([MICROBIAL_MP * UREA_CP])
    energy = MICROBIAL_MP * MICROBIAL_CP_PER_TDN * tdn + drup
    nitrogen = MICROBIAL_MP / RDP_PER_MICROBIAL_CP * rdp + drup
    return np.stack([energy, nitrogen], axis=-2), np.array([0.0, MICROBIAL_MP / RDP_PER_MICROBIAL_CP * UREA_CP])


def added_n_room(prep, t):
    """Dietary CP-equivalent margin for the cap on supplemental urea N."""
    return MAX_ADDED_N_SHARE * prep.cp[:, :, t]


def supplement(prep, t, x, production):
    """Minimum supplemental urea, kg/head/day, for an already feasible diet."""
    if prep.urea_max_dm == 0:
        return np.zeros_like(production)
    if prep.protein_model == "cp_proxy":
        feed = prep.cp[:, :, t] * MICROBIAL_MP
        coefficient = MICROBIAL_MP * UREA_CP
    else:
        feed = MICROBIAL_MP / RDP_PER_MICROBIAL_CP * prep.rdp[:, :, t] + prep.drup[:, :, t]
        coefficient = MICROBIAL_MP / RDP_PER_MICROBIAL_CP * UREA_CP
    m, p = prep.mp_m, prep.mp_p
    if x.ndim == 4:
        feed, m, p = feed[:, :, None, :], m[:, :, None], p[:, :, None]
    supply = (x * feed).sum(-1)
    need = m + production * p
    return np.maximum(need - supply, 0) / coefficient
