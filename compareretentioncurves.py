# -*- coding: utf-8 -*-
import os, runpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# ================================ SETTINGS ================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))      # folder that holds this script
except NameError:                                            # run cell-by-cell (e.g. Spyder): use the working folder
    _HERE = os.getcwd()
BASE = _HERE                                                 # repository folder
CODE_DIR = os.path.join(BASE, "Input_main")
PATH_CODE2 = next((os.path.join(CODE_DIR, n) for n in ("Code2.py", "code2.py")
                   if os.path.isfile(os.path.join(CODE_DIR, n))), os.path.join(CODE_DIR, "Code2.py"))
OUT_DIR = os.path.join(BASE, "Output", "RetentionCurves")
os.makedirs(OUT_DIR, exist_ok=True)

DROUGHT_LEVELS = ['Optimum', 'Moderate Drought', 'Severe Drought']   # static treatments only
CAMPAIGNS = ['Cam1']                                  # campaign used for each treatment ...
TREATMENT_CAMPAIGNS = {'Optimum': ['Cam1', 'Cam2']}   # ... except OptConst: both, for more wet-end points
MIN_H_CM = 10.2              # drop TEROS 21 readings below ~1 kPa (not physically possible)
N_BOOT = 500                 # bootstrap replicates (resampled within each treatment)
EQ_MARGIN_H = 2.0            # curves equivalent if the 95 % CI of their mean offset is within 2x in |h|
RNG = np.random.default_rng(42)
H0_CM = 10.0 ** 6.8          # PDI: theta = 0 at oven dryness, h0 = 10^6.8 cm
RHO_S = 2.65                 # particle density, g cm-3 (porosity caps theta_s)
THR_MAX_PDI = 0.20           # upper bound of the PDI adsorptive water content

SOIL_COLORS = {"Silt Loam": "#8B5A2B", "Sandy Loam": "#C9A97C"}
DL_MARKERS = {'Optimum': 'o', 'Moderate Drought': 's', 'Severe Drought': 'D'}
DL_DISPLAY = {'Optimum': 'OptConst', 'Moderate Drought': 'IntConst', 'Severe Drought': 'LowConst'}

# ================================ DATA (from Code2.py) ================================
_ns = runpy.run_path(PATH_CODE2)
combined_df = _ns["combined_df"]
HYP_KPA_PER_CM = _ns["CM_H2O_TO_KPA"]          # Code2.py stored HYPROP h in kPa with this factor
RHO_B = _ns["RHO_B"]                           # bulk density, g cm-3
KPA2CM = 1.0 / 0.0980665                       # TEROS 21 reads kPa; work in cm (pF = log10|h|)
SOILS = [("Silt Loam", _ns["HydrProp_Silt"]), ("Sandy Loam", _ns["HydrProp_Sand"])]
plt.close("all")                               # Code2.py's check plots are not needed here


# ================================ VG-PDI RETENTION MODEL ================================
# theta(h) = (theta_s - theta_r) * Se_cap(h) + theta_r * Se_ad(h); theta(h0) = 0.
# Peters (2013) WRR 49:6765; Iden & Durner (2014) WRR 50:7530. h in cm, alpha in cm-1.
def _pdi_theta_of_h(h_cm, thr, ths, alpha, n):
    h = np.asarray(h_cm, float)
    m = 1.0 - 1.0 / n
    G, G0 = (1.0 + (alpha * h) ** n) ** (-m), (1.0 + (alpha * H0_CM) ** n) ** (-m)
    se_cap = np.clip((G - G0) / max(1.0 - G0, 1e-12), 0.0, 1.0)
    x, xa, x0 = np.log10(np.clip(h, 1e-6, None)), np.log10(1.0 / alpha), np.log10(H0_CM)
    b = 0.1 + 0.2 / n ** 2 * (1.0 - np.exp(-((thr / max(ths - thr, 1e-9)) ** 2)))
    se_ad = np.clip(1.0 + (x - xa + b * np.logaddexp(0.0, (xa - x) / b)) / (xa - x0), 0.0, 1.0)
    return (ths - thr) * se_cap + thr * se_ad

_PDI_XGRID = np.linspace(-2.0, np.log10(H0_CM), 900)

def _pdi_logh_of_theta(theta, thr, ths, alpha, n):
    """log10|h| [cm] from theta, by inverting theta(h) on a monotone grid."""
    th_g = np.minimum.accumulate(_pdi_theta_of_h(10.0 ** _PDI_XGRID, thr, ths, alpha, n))
    return np.interp(np.asarray(theta, float), th_g[::-1], _PDI_XGRID[::-1])

def fit_pdi(theta, h, ths_cap):
    """4-parameter PDI fit (residuals in log10|h|) -> callable theta -> log10|h|, or None."""
    theta, h = np.asarray(theta, float), np.asarray(h, float)
    ok = np.isfinite(theta) & np.isfinite(h) & (h > 0) & (theta > 0)
    theta, h = theta[ok], h[ok]
    if theta.size < 5 or np.ptp(theta) < 1e-4:
        return None
    lo, hi = float(theta.min()), float(theta.max())
    ths_hi = max(float(ths_cap), 1.002 * hi)
    lb, ub = [0.0, 1.001 * hi, 1e-4, 1.02], [THR_MAX_PDI, ths_hi, 5.0, 12.0]
    p0 = np.clip([min(0.5 * lo, 0.05), min(1.05 * hi, ths_hi), 0.02, 1.6], np.array(lb) + 1e-9, np.array(ub) - 1e-9)
    try:
        p, _ = curve_fit(_pdi_logh_of_theta, theta, np.log10(h), p0=p0, bounds=(lb, ub), maxfev=40000)
    except Exception:
        return None
    return lambda t, pp=p: _pdi_logh_of_theta(t, *pp)


# ================================ FIGURE ================================
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 10, 'axes.linewidth': 0.9, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'mathtext.default': 'regular'})
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharey=True)
stat_txt, ylo, yhi = [], [], []

for ax, (soil, hyp), panel in zip(axes, SOILS, ['(a)', '(b)']):
    col = SOIL_COLORS[soil]
    por_cap = 1.02 * (1.0 - RHO_B[soil] / RHO_S)             # theta_s cannot exceed porosity

    # HYPROP reference curve, |h| back in cm
    hyp_th = pd.to_numeric(hyp['Theta'], errors='coerce').values
    hyp_h = pd.to_numeric(hyp['h_Theta'], errors='coerce').values / HYP_KPA_PER_CM
    ok = np.isfinite(hyp_th) & np.isfinite(hyp_h) & (hyp_h > 0)
    hyp_th, hyp_h = hyp_th[ok], hyp_h[ok]
    o = np.argsort(hyp_th)
    hyp_th, hyp_h = hyp_th[o], hyp_h[o]
    f_h_of_th = interp1d(hyp_th, np.log10(hyp_h), bounds_error=False, fill_value=np.nan)
    o2 = np.argsort(np.log10(hyp_h))
    f_th_of_h = interp1d(np.log10(hyp_h)[o2], hyp_th[o2], bounds_error=False, fill_value=np.nan)

    # sensor observations: theta from box weights + TEROS 21 |h|
    sel = pd.Series(False, index=combined_df.index)
    for dl in DROUGHT_LEVELS:
        sel |= (combined_df['DroughtLevel'] == dl) & combined_df['Cam'].isin(TREATMENT_CAMPAIGNS.get(dl, CAMPAIGNS))
    sub = combined_df[(combined_df['Soil'] == soil) & sel][['SWC_v', 'SMP_teros21', 'DroughtLevel']].copy()
    sub[['SWC_v', 'SMP_teros21']] = sub[['SWC_v', 'SMP_teros21']].apply(pd.to_numeric, errors='coerce')
    sub = sub.dropna(subset=['SWC_v', 'SMP_teros21'])
    sub = sub[(sub['SWC_v'] > 0) & (sub['SMP_teros21'] > 0)]
    sub['h_cm'] = sub['SMP_teros21'].astype(float) * KPA2CM
    sub = sub[sub['h_cm'] > MIN_H_CM]
    th_obs, h_obs = sub['SWC_v'].values, sub['h_cm'].values

    # PDI fit and bootstrap (resampled within each treatment)
    model = fit_pdi(th_obs, h_obs, por_cap) if th_obs.size > 4 else None
    th_grid = np.linspace(th_obs.min(), th_obs.max(), 250)   # no extrapolation
    boot = np.full((N_BOOT, th_grid.size), np.nan)
    groups = [g for g in (np.where(sub['DroughtLevel'].values == d)[0] for d in DROUGHT_LEVELS) if g.size]
    if model is not None and th_obs.size >= 5:
        for b in range(N_BOOT):
            idx = (np.concatenate([RNG.choice(g, size=g.size, replace=True) for g in groups])
                   if len(groups) > 1 else RNG.choice(th_obs.size, size=th_obs.size, replace=True))
            if np.ptp(th_obs[idx]) < 1e-4:
                continue
            mb = fit_pdi(th_obs[idx], h_obs[idx], por_cap)
            if mb is not None:
                boot[b] = mb(th_grid)
    with np.errstate(invalid='ignore'):
        hi_b = np.nanpercentile(boot, 97.5, axis=0)
    have_band = np.isfinite(hi_b).any()

    # n, mean offset of the fitted curve from HYPROP, bootstrap CI, equivalence
    n_pair = int(np.isfinite(f_th_of_h(np.log10(h_obs))).sum())
    lh_hyp = f_h_of_th(th_grid)
    off_mean = off_lo = off_hi = np.nan
    equiv = 'n/a'
    if model is not None:
        off_mean = float(np.nanmean(model(th_grid) - lh_hyp))
        if np.isfinite(boot).any():
            d = np.nanmean(boot - lh_hyp[None, :], axis=1)
            d = d[np.isfinite(d)]
            if d.size >= 20:
                off_lo, off_hi = np.percentile(d, [2.5, 97.5])
                mg = np.log10(EQ_MARGIN_H)
                equiv = 'yes' if (off_lo > -mg and off_hi < mg) else 'no'

    # plot: HYPROP curve, fitted curve, observations
    ax.plot(hyp_th * 100, hyp_h, '-', color=col, lw=2.4, zorder=4, solid_capstyle='round')
    if model is not None:
        ax.plot(th_grid * 100, 10.0 ** model(th_grid), '--', color=col, lw=1.9, dashes=(5, 2.5), zorder=5)
    for dl in DROUGHT_LEVELS:
        d = sub[sub['DroughtLevel'] == dl]
        if len(d):
            ax.plot(d['SWC_v'] * 100, d['h_cm'], DL_MARKERS[dl], ls='none', ms=5.5, mfc='white', mec='0.15', mew=0.9, zorder=6)
    ax.set_yscale('log')
    ylo.append(float(sub['h_cm'].min()))
    yhi.append(max(sub['h_cm'].max(), np.nanmax(10 ** hi_b) if have_band else 0))
    ax.set_xlim(max(0.0, min(th_obs.min(), hyp_th.min()) * 100 - 2), max(th_obs.max(), hyp_th.max()) * 100 + 2)
    ax.set_xlabel('Volumetric water content  (Vol %)')
    for sp in ('top', 'right', 'bottom', 'left'):
        ax.spines[sp].set_visible(True)
    ax.tick_params(which='both', top=True, right=True, direction='in', length=3.5)
    ax.grid(True, which='major', lw=0.4, color='0.92', zorder=0)
    ax.text(0.975, 0.965, f'{panel} {soil}', transform=ax.transAxes, ha='right', va='top', fontsize=10.5, fontweight='bold')
    stat_txt.append([f'$n$ = {n_pair}', f'offset $\\times${10 ** off_mean:.2f} [{10 ** off_lo:.2f}, {10 ** off_hi:.2f}]',
                     'equivalent' if equiv == 'yes' else 'not equivalent'] if np.isfinite(off_lo) else [f'$n$ = {n_pair}'])

axes[0].set_ylabel(r'Matric potential, $|h|$  (cm)')
axes[0].set_ylim(10 ** np.floor(np.log10(min(ylo) * 0.7)), 10 ** np.ceil(np.log10(max(yhi) * 1.4)))

# legends: (a) curves, (b) treatment symbols, each with that panel's statistics
curve_h = [Line2D([], [], color='0.25', lw=2.4, label='HYPROP'),
           Line2D([], [], color='0.25', lw=1.9, ls='--', dashes=(5, 2.5), label='Sensor-based fit (PDI)')]
marker_h = [Line2D([], [], ls='none', marker=DL_MARKERS[d], ms=5.5, mfc='white', mec='0.15', label=DL_DISPLAY[d])
            for d in DROUGHT_LEVELS]
for k, ax in enumerate(axes):
    h = (curve_h if k == 0 else marker_h) + [Line2D([], [], ls='none', marker='', label=t) for t in stat_txt[k]]
    lg = ax.legend(handles=h, loc='lower left', frameon=True, fontsize=7.8, handlelength=1.8, handletextpad=0.6,
                   labelspacing=0.4, borderpad=0.5, framealpha=0.93, edgecolor='0.75', facecolor='white')
    lg.get_frame().set_linewidth(0.6)

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'FigS1_retention_sensor_vs_HYPROP.{ext}'), dpi=600, bbox_inches='tight')
print(f'[DONE] Fig. S1 written to: {OUT_DIR}')
