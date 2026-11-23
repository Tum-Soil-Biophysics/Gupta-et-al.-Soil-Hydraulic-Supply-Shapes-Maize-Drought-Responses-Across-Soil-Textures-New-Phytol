
import re, sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy.optimize import curve_fit
from scipy import stats, integrate

# ================================ SETTINGS ================================
BASE_DIR = Path(__file__).resolve().parent     # repository folder
DATA_DIR = BASE_DIR / "Hyprob_data"     # HYPROP files (Rep*, Sandy2, Silt1)
PLANT_DIR = BASE_DIR                    # NormTrans.xlsx
OUT_DIR = BASE_DIR / "Output" / "Drydown"
PLANT_FILE = "NormTrans.xlsx"
POOLED_FILES = {"Sandy Loam": "Sandy2.xlsx", "Silt Loam": "Silt1.xlsx"}   # joint fits (not replicates)
REP_GLOB = "Rep*.xls*"
SH_PAR, SH_STAT = "Fitting-Parameter value", "Fitting-Statistical analysis"
SH_TH, SH_K = "Fitting-Retention \u0398(pF)", "Fitting-Conductivity K(pF)"
BULK_DENSITY = {"Sandy Loam": 1.28, "Silt Loam": 1.14}   # g cm-3
SOIL_ORDER = ["Silt Loam", "Sandy Loam"]
PF_DRY = 4.2                   # dry limit of Phi = wilting point (not the PDI parameter pF_dry)
DRY_LIMITS = (PF_DRY, 6.8, 8.0, 10.0, 12.0)              # audited in Tables S4b / S4c
SPAN_PF = (2.0, 4.0)           # window for the span-normalised gap (Table 1)
BAND_SECOND, BAND_STEP = (0.35, 0.45), 0.01              # second threshold = band of Tr/Trmax
N_BOOT, RNG_SEED, DPI = 8000, 11, 300
PF_MIN, N_GRID = -2.0, 200_001
PF_GRID = np.linspace(PF_MIN, PF_DRY, N_GRID)

# ---- style ----
FS, FS_AXIS, FS_PANEL, FS_LEGEND = 23, 26, 26, 21
FIG_SCALE, PUB_FIG_PAD = 1.30, 1.22
TICK_W, TICK_L, SPINE_W = 1.2, 5, 1.2
CAPSIZE, ELW, BAR_EDGE_W, HATCH = 6, 1.6, 1.2, "xx"
COL_SOIL = {"Silt Loam": "#8B5A2B", "Sandy Loam": "#C9A97C"}
COL_GENO = {"Alpenfex": "#2E7D32", "Braunes Schindelmeiser": "#66A861", "Pfarrkirchner": "#A5D6A7",
            "P8888": "#5E3C99", "Luigi CS": "#8E71C1", "Benedictio KWS": "#D6C6EA"}
GENO_ORDER = ["Alpenfex", "Braunes Schindelmeiser", "Pfarrkirchner", "Benedictio KWS", "Luigi CS", "P8888"]
COL_SCALE = {"theta": "#4D4D4D", "SMP": "#0B3C5D", "K(h)": "#2E86AB", "MFP phi": "#63C7B2"}
COL_ACC, COL_CI = "#C0392B", "#1B5E4F"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": SPINE_W, "savefig.bbox": "tight"})

# internal scale keys -> labels (figures: mathtext, tables: plain text)
SCALES = [("VWC", "theta"), ("pF", "SMP"), ("logK", "K(h)"), ("logphi", "MFP phi")]
NAMES = dict(SCALES)
PAIR_LIST = [("logphi", "logK"), ("logphi", "pF"), ("logK", "pF"), ("logphi", "VWC"), ("logK", "VWC"), ("pF", "VWC")]
DISP_SCALE = {"theta": r"$\theta$", "SMP": r"$h$", "K(h)": r"$K(h)$", "MFP phi": r"$\Phi(h)$"}
DISP_TXT = {"theta": "\u03b8", "SMP": "h", "K(h)": "K(h)", "MFP phi": "\u03a6(h)"}
LO, HI = BAND_SECOND
BAND_LABEL = f"{LO:.0%}-{HI:.0%} of max"
BAND = np.round(np.arange(LO, HI + BAND_STEP / 2, BAND_STEP), 6)


def band_math(bold=False):
    sub = f"{LO:.2f}-{HI:.2f}"
    return r"$\mathbf{\theta_{" + sub + r"}}$" if bold else r"$\theta_{" + sub + r"}$"

def _dthr_b(t):
    return band_math(True) if t == BAND_LABEL else {"theta_crit": r"$\mathbf{\theta_{crit}}$"}.get(t, t)

_dsc = lambda t: DISP_SCALE.get(t, t)
_txt = lambda t: DISP_TXT.get(t, t)
tostr = lambda df, f: df.to_string(index=False, float_format=lambda v: f"{v:{f}}")

def _spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(SPINE_W)
    ax.tick_params(direction="inout", length=TICK_L, width=TICK_W, labelsize=FS)
    ax.set_axisbelow(True)

def _panel(ax, label, x=0.02, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=FS_PANEL, fontweight="bold",
            clip_on=False, zorder=20)

def _cum(x, f):
    """Cumulative trapezoid integral of f over x, starting at 0."""
    return np.concatenate([[0.0], np.cumsum(np.diff(x) * (f[1:] + f[:-1]) / 2.0)])

def _phi(g, lk):
    """Phi on pF grid g = integral of K dh from each point to the dry end (dh = ln10 * h * dpF)."""
    cum = _cum(g, 10.0 ** lk * np.log(10.0) * 10.0 ** g)
    return cum[-1] - cum


# ================================ 1. HYPROP PDI FITS ================================
def _num(v):
    """Float from a HYPROP cell ('1.00E-1*' = parameter on its bound)."""
    try:
        return float(v.replace("*", "").strip() if isinstance(v, str) else v)
    except (TypeError, ValueError):
        return np.nan

def read_sheet(path, sheet):
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError as e:
        return e

def build_set(path):
    """One HYPROP workbook: tabulated theta(pF) and log10 K(pF) + parameters. None if it has no fit."""
    th, kk = read_sheet(path, SH_TH), read_sheet(path, SH_K)
    for x in (th, kk):
        if isinstance(x, Exception):
            print(f"  SKIPPED {path.name}: {x}")
            return None
    th.columns, kk.columns = ["pF", "theta"], ["pF", "logK"]
    th, kk = th.apply(pd.to_numeric, errors="coerce").dropna(), kk.apply(pd.to_numeric, errors="coerce").dropna()
    if len(th) < 50 or len(kk) < 50:
        print(f"  SKIPPED {path.name}: only {len(th)} retention / {len(kk)} "
              f"conductivity points -> no fitted curve behind this file.")
        return None
    if th.theta.max() > 1.5:                      # Vol% -> cm3/cm3
        th["theta"] = th.theta / 100.0
    th, kk = th.sort_values("pF"), kk.sort_values("pF")
    if th.pF.min() > PF_MIN or kk.pF.max() < PF_DRY - 1e-9:
        print(f"  WARNING {path.name}: exported curves span pF {th.pF.min():.2f}-{kk.pF.max():.2f}, which does not cover "
              f"the requested {PF_MIN:.2f}-{PF_DRY:.2f}; edges are extrapolated as constants.")
    par, d = read_sheet(path, SH_PAR), {}
    if not isinstance(par, Exception):
        par.columns = [str(c).strip() for c in par.columns]
        d = {str(r["Parameter"]).strip(): _num(r["Value"]) for _, r in par.iterrows()}
        d["_at_bound"] = sorted(str(r["Parameter"]).strip() for _, r in par.iterrows()
                                if isinstance(r["Value"], str) and "*" in r["Value"])
    d["_tab_th"] = (th.pF.values.copy(), th.theta.values.copy())     # full export (beyond PF_DRY)
    d["_tab_k"] = (kk.pF.values.copy(), kk.logK.values.copy())
    d["pf_grid"] = PF_GRID
    d["th_grid"] = tg = np.interp(PF_GRID, th.pF.values, th.theta.values)
    d["logK_grid"] = np.interp(PF_GRID, kk.pF.values, kk.logK.values)
    d["phi_grid"] = _phi(PF_GRID, d["logK_grid"])
    d["_th_rev"], d["_pf_rev"], d["th_wet"], d["th_dry"] = tg[::-1], PF_GRID[::-1], float(tg[0]), float(tg[-1])
    st = read_sheet(path, SH_STAT)
    st = {} if isinstance(st, Exception) else dict(zip(st.iloc[:, 0].astype(str), st.iloc[:, 1]))
    d["rmse_th"], d["rmse_K"] = pd.to_numeric(st.get("RMSE_TH"), errors="coerce"), pd.to_numeric(st.get("RMSE_K"), errors="coerce")
    d["file"] = path.name
    return d

def load_parameter_sets(folder):
    """Every replicate fit; soil from 'sandy'/'silt' and replicate from 'Rep<k>' in the file name."""
    sets, seen = {}, {}
    pooled = {v.lower() for v in POOLED_FILES.values()}
    for f in sorted(folder.glob(REP_GLOB)):
        if f.name.lower() in pooled or f.name.startswith("~$"):
            continue
        low = f.stem.lower()
        soil = "Sandy Loam" if "sand" in low else "Silt Loam" if "silt" in low else None
        if soil is None:
            print(f"  SKIPPED {f.name}: cannot tell which soil from the filename.")
            continue
        m = re.search(r"rep\s*_?(\d+)", low) or re.search(r"(\d+)(?!.*\d)", low)
        rep = m.group(1) if m else None
        d = build_set(f)
        if d is None:
            continue
        d["soil"] = soil
        if rep is None:
            rep = str(sum(1 for p in sets.values() if p["soil"] == soil) + 1)
        tag = f"{soil.split()[0]} R{rep}"
        while tag in sets:
            rep = str(int(rep) + 1) if rep.isdigit() else rep + "b"
            tag = f"{soil.split()[0]} R{rep}"
        sets[tag] = d
        seen.setdefault(soil, []).append(tag)
        print(f"  loaded  {f.name:22s} -> {tag:11s}  n={d.get('n', np.nan):.3f}  Ks={d.get('Ks', np.nan):.1f}  "
              f"tau={d.get('tau', np.nan):+.3f}  omega={d.get('omega', np.nan):.2e}  a={d.get('a', np.nan):+.3f}  "
              f"RMSE_th={d['rmse_th']:.4f}" + (f"   [on bound: {', '.join(d['_at_bound'])}]" if d.get("_at_bound") else ""))
    if not sets:
        sys.exit(f"No usable HYPROP files matching {REP_GLOB} in {folder}.")
    for soil, tags in seen.items():
        print(f"  -> {soil}: {len(tags)} replicate fit(s)  [{', '.join(tags)}]")
    return sets

def pooled_parameters(soil, folder):
    """Joint HYPROP fit of one soil (Sandy2.xlsx / Silt1.xlsx)."""
    d = build_set(folder / POOLED_FILES[soil])
    if d is None:
        sys.exit(f"{POOLED_FILES[soil]} missing or without a fitted curve.")
    d.update(soil=soil, source=f"joint fit over replicates ({POOLED_FILES[soil]})")
    return d


# ================================ 2. theta -> pF, log10 K, log10 phi ================================
def convert(p, theta):
    """theta -> (pF, log10 K, log10 phi); NaN outside (theta at the dry limit, theta at PF_MIN)."""
    theta = np.atleast_1d(np.asarray(theta, float))
    ok = (theta > p["th_dry"]) & (theta < p["th_wet"])
    pf, lk, lp = (np.full_like(theta, np.nan) for _ in range(3))
    if ok.any():
        pfg = p.get("pf_grid", PF_GRID)
        pf[ok] = np.interp(theta[ok], p["_th_rev"], p["_pf_rev"])
        lk[ok] = np.interp(pf[ok], pfg, p["logK_grid"])
        ph = np.interp(pf[ok], pfg, p["phi_grid"])
        with np.errstate(divide="ignore", invalid="ignore"):
            lp[ok] = np.where(ph > 0, np.log10(ph), np.nan)
    return pf, lk, lp

def _pdi_parts(p, h):
    a_, n, thr, ths = p["alpha"], p["n"], p["th_r"], p["th_s"]
    m, x, x0 = 1 - 1 / n, np.log10(h), p["pF_dry"]
    xa = np.log10(1.0 / a_)
    G, G0 = (1 + (a_ * h) ** n) ** (-m), (1 + (a_ * 10 ** x0) ** n) ** (-m)
    b = 0.1 + 0.2 / n ** 2 * (1 - np.exp(-(thr / (ths - thr)) ** 2))
    Sad = 1 + 1 / (xa - x0) * (x - xa + b * np.log(1 + np.exp((xa - x) / b)))
    return m, x0, xa, G, G0, Sad

def pdi_theta(p, h):
    """Independent PDI retention (only to audit the tabulated curves)."""
    m, x0, xa, G, G0, Sad = _pdi_parts(p, h)
    return (p["th_s"] - p["th_r"]) * ((G - G0) / (1 - G0)) + p["th_r"] * Sad

def pdi_logK(p, h):
    """Independent PDI conductivity: Mualem capillary + Peters (2013) film term."""
    m, x0, xa, G, G0, Sad = _pdi_parts(p, h)
    Sc = np.clip((G - G0) / (1 - G0), 1e-300, 1.0)
    F = lambda g: 1 - (1 - g ** (1 / m)) ** m
    Kcap = Sc ** p["tau"] * ((F(G) - F(G0)) / (1 - F(G0))) ** 2
    Kfilm = 10.0 ** (p["a"] * (1 - Sad) * (x0 - xa))
    with np.errstate(divide="ignore"):
        return np.log10(p["Ks"] * ((1 - p["omega"]) * Kcap + p["omega"] * Kfilm))

def validate_model(sets, log):
    """(1) Phi grid convergence vs a 4x coarser grid; (2) tabulated curves vs PDI from the parameter sheet."""
    log("\nHydraulic model validation")
    worst = 0.0
    for tag, p in sets.items():
        coarse = PF_GRID[::4]
        integ = (10.0 ** p["logK_grid"][::4]) * np.log(10.0) * 10.0 ** coarse
        cum = _cum(coarse, integ)
        phi_c, ref = cum[-1] - cum, p["phi_grid"][::4]
        m = ref > 0
        worst = max(worst, float(np.max(np.abs(phi_c[m] - ref[m]) / ref[m])))
    log(f"  phi grid convergence ({N_GRID} vs {len(PF_GRID[::4])} points): max relative change {worst:.2e}")
    if worst > 1e-4:
        log("  WARNING: phi has not converged; increase N_GRID.")
    log("  tabulated curves vs PDI evaluated from the parameter sheet:")
    for tag, p in sets.items():
        if not all(np.isfinite(p.get(k, np.nan)) for k in ("alpha", "n", "th_r", "th_s", "Ks", "tau", "omega", "a", "pF_dry")):
            log(f"    {tag:11s} skipped (parameter sheet incomplete)")
            continue
        h = 10.0 ** PF_GRID
        dth, dk = np.abs(pdi_theta(p, h) - p["th_grid"]), np.abs(pdi_logK(p, h) - p["logK_grid"])
        log(f"    {tag:11s} theta: max {dth.max():.4f} cm3/cm3   log10 K: median {np.median(dk):.4f}, "
            f"max {dk.max():.3f} at pF {PF_GRID[int(np.argmax(dk))]:.2f}")

def _tail(p):
    """Exported curves (pf_th, theta, pf_k, logK) and the log-linear fit of the dry tail of log K."""
    pf_th, thk, pfk, lk = (*p["_tab_th"], *p["_tab_k"]) if "_tab_th" in p and "_tab_k" in p else \
        (p.get("pf_grid", PF_GRID), p["th_grid"], p.get("pf_grid", PF_GRID), p["logK_grid"])
    t = pfk >= pfk.max() - 0.8
    slope, icpt = np.polyfit(pfk[t], lk[t], 1)
    return pf_th, thk, pfk, lk, slope, icpt

def _rebuild_at_dry(p, pf_dry):
    """Copy of one soil's curves with Phi referenced to another dry limit (K extrapolated past the export)."""
    pf_th, thk, pfk, lk, slope, icpt = _tail(p)
    g = np.linspace(PF_MIN, pf_dry, N_GRID)
    lknew = np.where(g <= pfk.max(), np.interp(g, pfk, lk), icpt + slope * g)
    thnew = np.interp(g, pf_th, thk)
    q = dict(p)
    q.update(pf_grid=g, th_grid=thnew, logK_grid=lknew, phi_grid=_phi(g, lknew), _th_rev=thnew[::-1], _pf_rev=g[::-1],
             th_wet=float(thnew[0]), th_dry=float(thnew[-1]))
    return q

def dry_limit_sensitivity(params, fits, log):
    """Table S4c: whole analysis re-run with Phi referenced to each limit in DRY_LIMITS."""
    rows = []
    for pf_dry in DRY_LIMITS:
        Yx = yardstick(thresholds_on_scales(fits, {s: _rebuild_at_dry(p, pf_dry) for s, p in params.items()}))
        for thr in Yx.Threshold.unique():
            y = Yx[Yx.Threshold == thr].set_index("key")
            d_ = {k: float(y.loc[k, "d"]) for k, _ in SCALES}
            rows.append({"Dry limit (pF)": pf_dry, "Reference": "yes" if pf_dry == PF_DRY else "", "Threshold": thr,
                         **{f"|d| {_txt(nm)}": d_[k] for k, nm in SCALES},
                         "n on \u03a6": int(y.loc["logphi", "n_sandy"] + y.loc["logphi", "n_silt"]),
                         "order_holds": bool(d_["logphi"] < d_["logK"] < d_["pF"]),
                         "supply_below_state": bool(max(d_["logphi"], d_["logK"]) < min(d_["pF"], d_["VWC"]))})
    S = pd.DataFrame(rows)
    phicol = f"|d| {_txt('MFP phi')}"
    base = S[S["Dry limit (pF)"] == PF_DRY].set_index("Threshold")[phicol]
    S["\u0394 |d| \u03a6 vs reference"] = S[phicol] - S.Threshold.map(base)
    log("\nDry-limit sensitivity (Table S4c): the ordering re-tested at each limit")
    log(S.to_string(index=False, float_format=lambda v: f"{v:.4f}", formatters={"Dry limit (pF)": lambda v: f"{v:.1f}"}))
    nref = int(S.loc[S["Dry limit (pF)"] == PF_DRY, "n on \u03a6"].max())
    for pf_dry in DRY_LIMITS:
        sub = S[S["Dry limit (pF)"] == pf_dry]
        lost = nref - int(sub["n on \u03a6"].max())
        log(f"    pF {pf_dry:>4.1f}:  order holds in {int(sub.order_holds.sum())}/{len(sub)} thresholds, "
            f"supply < state in {int(sub.supply_below_state.sum())}/{len(sub)}"
            + (f", {lost} plant-threshold value{'s' if lost != 1 else ''} lost off the dry end" if lost else ""))
    if not S.order_holds.all():
        log("    NOTE: the ordering does NOT hold at every dry limit -- see the "
            "rows above before citing Table S4c as a robustness check.")
    return S

def integration_audit(params, fits, log):
    """Table S4: Phi at the mean threshold suction under 6 integration schemes; Table S4b: other dry limits."""
    thr_pf = {}
    for soil, p in params.items():
        g = fits[fits.Soil == soil]
        thr_pf[(soil, "theta_crit")] = float(np.nanmean(convert(p, g.theta_crit.values)[0]))
        th = np.array([np.nanmean([theta_at_fraction(r, f) for f in BAND]) for r in g.itertuples()])
        thr_pf[(soil, BAND_LABEL)] = float(np.nanmean(convert(p, th)[0]))

    def phi_at(p, target, scheme):
        pfk, lk = p.get("pf_grid", PF_GRID), p["logK_grid"]
        if scheme == "quad":
            f = lambda x: 10.0 ** np.interp(x, pfk, lk) * np.log(10.0) * 10.0 ** x
            return float(integrate.quad(f, target, PF_DRY, limit=500)[0])
        npts = {"coarse": (N_GRID - 1) // 16 + 1, "finer": (N_GRID - 1) * 4 + 1}.get(scheme, N_GRID)
        g = np.linspace(PF_MIN, PF_DRY, npts)
        K = 10.0 ** np.interp(g, pfk, lk)
        if scheme == "h-space":
            cum = _cum(10.0 ** g, K)
            return float(np.interp(target, g, cum[-1] - cum))
        f = K * np.log(10.0) * 10.0 ** g
        cum = _cum(g, f)
        if scheme == "simpson":
            cum = cum * (float(integrate.simpson(f, x=g)) / cum[-1])
        return float(np.interp(target, g, cum[-1] - cum))

    labels = {"reference": f"Trapezoidal, uniform in pF, {N_GRID:,} points (reference)",
              "simpson": "Simpson's rule, uniform in pF", "quad": "Adaptive quadrature (no fixed grid)",
              "h-space": "Trapezoidal, integrated in h (logarithmic spacing)",
              "coarse": f"Trapezoidal, {((N_GRID-1)//16)+1:,} points (16x coarser)",
              "finer": f"Trapezoidal, {((N_GRID-1)*4)+1:,} points (4x finer)"}
    A = pd.DataFrame([{"Scheme": lab, **{f"{soil} | {thr}": phi_at(params[soil], t, sc) for (soil, thr), t in thr_pf.items()}}
                      for sc, lab in labels.items()])
    cols = [c for c in A.columns if c != "Scheme"]
    A["Max deviation (%)"] = 100.0 * (A[cols].div(A[cols].iloc[0]) - 1.0).abs().max(axis=1)
    log("\nIntegration audit (Table S4): phi [cm2/d] at the mean threshold suction")
    log(tostr(A, ".6f"))
    log(f"  maximum deviation from the reference scheme across all schemes and thresholds: "
        f"{A['Max deviation (%)'].max():.2e} % -> phi is independent of the integration scheme")
    log("  dry reference limit (tail extrapolated log-linearly beyond the fits):")
    conv = []
    for soil, p in params.items():
        _, _, pfk, lk, slope, icpt = _tail(p)

        def phi_to(ref, target):
            g = np.linspace(PF_MIN, ref, 300_001)
            cum = _cum(g, 10.0 ** np.where(g <= pfk.max(), np.interp(g, pfk, lk), icpt + slope * g) * np.log(10.0) * 10.0 ** g)
            return float(np.interp(target, g, cum[-1] - cum))
        for (s_, thr), target in thr_pf.items():
            if s_ == soil:
                base = phi_to(PF_DRY, target)
                for ref in DRY_LIMITS:
                    v = phi_to(ref, target)
                    conv.append(dict(Soil=soil, Threshold=thr, upper_limit_pF=ref, phi=v, tail_removed=base - v,
                                     change_pct=100.0 * (v / base - 1.0), film_exponent=slope))
        log(f"    {soil:11s} K ~ h^{slope:.2f} in the dry tail (a > 1 required for convergence)")
    C = pd.DataFrame(conv)
    log(tostr(C, ".6f"))
    return A, C


# ================================ 3. PLANT DATA AND THRESHOLDS ================================
def linear_plateau(x, plateau, slope, xc):
    """Plateau above xc, linear decline below."""
    return np.where(x >= xc, plateau, plateau - slope * (xc - x))

def load_plant_data(folder):
    d = pd.read_excel(folder / PLANT_FILE)
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns={"NormalisedTrans (cm³ h⁻¹ cm⁻²)": "NormTrans"})     # Excel header -> short name
    missing = {"Soil", "Genotype", "Replicate", "NormTrans", "GWC"} - set(d.columns)
    if missing:
        sys.exit(f"{PLANT_FILE} is missing columns: {sorted(missing)}")
    unknown = set(d.Soil.unique()) - set(BULK_DENSITY)
    if unknown:
        sys.exit(f"No bulk density for soil(s): {sorted(unknown)}. Add them to BULK_DENSITY.")
    d["BD"] = d.Soil.map(BULK_DENSITY)
    d["VWC"] = d.GWC * d.BD
    print(f"  {len(d)} observations, {d.Soil.nunique()} soils x {d.Genotype.nunique()} genotypes x "
          f"{d.Replicate.nunique()} replicates")
    return d

def fit_theta_space(d):
    """One linear-plateau fit per plant (NormTrans vs VWC)."""
    rows = []
    for (soil, gen, rep), g in d.groupby(["Soil", "Genotype", "Replicate"]):
        x, y = g.VWC.values, g.NormTrans.values
        y95 = np.percentile(y, 95)
        try:
            p, _ = curve_fit(linear_plateau, x, y, p0=[y95, y95 / 0.05, np.median(x)],
                             bounds=([0, 1e-9, x.min()], [3 * y95, np.inf, x.max()]), maxfev=400_000)
        except RuntimeError:
            print(f"  fit failed: {soil} {gen} {rep}")
            continue
        r2 = 1 - np.sum((y - linear_plateau(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append(dict(Soil=soil, Genotype=gen, Replicate=rep, R2=r2, plateau=p[0], slope=p[1], theta_crit=p[2]))
    out = pd.DataFrame(rows)
    print(f"  {len(out)} plant-level fits, R2 {out.R2.min():.2f}-{out.R2.max():.2f}")
    return out

def theta_at_fraction(row, frac):
    """theta at which the fitted response reaches `frac` of its plateau."""
    return row.theta_crit - row.plateau * (1.0 - frac) / row.slope

def band_values(p, g):
    """Per plant: mean of the CONVERTED values over the band (undefined fractions dropped)."""
    out = {k: [] for k in ("VWC", "pF", "logK", "logphi", "n_band")}
    for row in g.itertuples():
        th = np.array([theta_at_fraction(row, f) for f in BAND])
        pf, lk, lp = convert(p, th)
        ok = np.isfinite(lp)
        with np.errstate(invalid="ignore"):
            out["VWC"].append(np.nanmean(np.where(ok, th, np.nan)) if ok.any() else np.nan)
            out["pF"].append(np.nanmean(pf) if np.isfinite(pf).any() else np.nan)
            out["logK"].append(np.nanmean(lk) if np.isfinite(lk).any() else np.nan)
            out["logphi"].append(np.nanmean(lp) if ok.any() else np.nan)
        out["n_band"].append(int(ok.sum()))
    return out

def thresholds_on_scales(fits, params):
    """Each plant's theta_crit and band threshold as theta, pF, log K, log phi."""
    rows = []
    for soil, g in fits.groupby("Soil"):
        pf, lk, lp = convert(params[soil], g.theta_crit.values)
        ids = dict(Soil=soil, Genotype=g.Genotype.values, Replicate=g.Replicate.values)
        rows.append(pd.DataFrame(dict(**ids, Threshold="theta_crit", VWC=g.theta_crit.values, pF=pf, logK=lk, logphi=lp, n_band=1)))
        rows.append(pd.DataFrame(dict(**ids, Threshold=BAND_LABEL, **band_values(params[soil], g))))
    out = pd.concat(rows, ignore_index=True)
    part = out[(out.Threshold == BAND_LABEL) & (out.n_band < len(BAND)) & (out.n_band > 0)]
    if len(part):
        print(f"  {len(part)} replicate(s) defined over only part of the {BAND_LABEL} band (theta reaches theta_r inside it):")
        for r in part.itertuples():
            print(f"     {r.Soil:11s} {r.Genotype:24s} {r.Replicate}  {r.n_band}/{len(BAND)} fractions")
    return out

# ================================ 4. STATISTICS ================================
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, 0, 0
    sp = np.sqrt(((a.std(ddof=1) ** 2) * (len(a) - 1) + (b.std(ddof=1) ** 2) * (len(b) - 1)) / (len(a) + len(b) - 2))
    return abs(b.mean() - a.mean()) / sp, sp, len(a), len(b)

def soils_d(s, key):
    return cohens_d(s.loc[s.Soil == "Sandy Loam", key], s.loc[s.Soil == "Silt Loam", key])

def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

def yardstick(T):
    """Pooled within-soil SD and |d| per threshold x scale."""
    rows = []
    for thr in T.Threshold.unique():
        s = T[T.Threshold == thr]
        for key, nm in SCALES:
            a, b = s.loc[s.Soil == "Sandy Loam", key], s.loc[s.Soil == "Silt Loam", key]
            d, sp, na, nb = cohens_d(a, b)
            rows.append(dict(Threshold=thr, key=key, Scale=nm, d=d, pooled_SD=sp, n_sandy=na, n_silt=nb,
                             mid=(np.nanmean(a) + np.nanmean(b)) / 2))
    return pd.DataFrame(rows)

def bootstrap_ordering(T, seed=RNG_SEED, nboot=N_BOOT):
    """Genotype cluster bootstrap: CI of |d| per scale and of paired differences."""
    rng = np.random.default_rng(seed)
    gens = sorted(T.Genotype.unique())
    ci, pairs = [], []
    for thr in T.Threshold.unique():
        s = T[T.Threshold == thr]
        draws = {k: [] for k, _ in SCALES}
        for _ in range(nboot):
            sub = pd.concat([s[s.Genotype == g] for g in rng.choice(gens, len(gens), replace=True)])
            for key, _ in SCALES:
                draws[key].append(soils_d(sub, key)[0])
        draws = {k: np.asarray(v)[np.isfinite(v)] for k, v in draws.items()}
        for key, nm in SCALES:
            ci.append(dict(Threshold=thr, key=key, Scale=nm, d=soils_d(s, key)[0],
                           lo=np.percentile(draws[key], 2.5), hi=np.percentile(draws[key], 97.5)))
        for k1, k2 in PAIR_LIST:
            n = min(len(draws[k1]), len(draws[k2]))
            diff = draws[k1][:n] - draws[k2][:n]
            pairs.append(dict(Threshold=thr, Comparison=f"{NAMES[k1]} vs {NAMES[k2]}", k1=k1, k2=k2, diff=diff.mean(),
                              lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5),
                              p_boot=max((diff >= 0).mean(), 1.0 / nboot)))
    return pd.DataFrame(ci), pd.DataFrame(pairs)

def wilcoxon_pairs(T, Y):
    """Per-genotype separation in SD units (Welch t, Holm), then signed-rank test across genotypes."""
    gens = sorted(T.Genotype.unique())
    per = []
    for thr in T.Threshold.unique():
        for key, nm in SCALES:
            sd = Y.loc[(Y.Threshold == thr) & (Y.key == key), "pooled_SD"].iloc[0]
            for g in gens:
                s = T[(T.Threshold == thr) & (T.Genotype == g)]
                a, b = s.loc[s.Soil == "Sandy Loam", key].dropna(), s.loc[s.Soil == "Silt Loam", key].dropna()
                per.append(dict(Threshold=thr, Scale=nm, Genotype=g, dg=abs(b.mean() - a.mean()) / sd,
                                se=np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) / sd,
                                p_soil=stats.ttest_ind(a, b, equal_var=False).pvalue))
    P = pd.DataFrame(per)
    P["p_holm"] = np.nan
    for _, idx in P.groupby(["Threshold", "Scale"]).groups.items():
        P.loc[idx, "p_holm"] = holm(P.loc[idx, "p_soil"].values)
    P["sig"] = P.p_holm.map(stars)
    rows = []
    for thr in T.Threshold.unique():
        for k1, k2 in PAIR_LIST:
            x = P[(P.Threshold == thr) & (P.Scale == NAMES[k1])].set_index("Genotype").dg[gens]
            y = P[(P.Threshold == thr) & (P.Scale == NAMES[k2])].set_index("Genotype").dg[gens]
            rows.append(dict(Threshold=thr, Comparison=f"{NAMES[k1]} vs {NAMES[k2]}", mean_diff=(x - y).mean(),
                             p_wilcoxon=stats.wilcoxon(x, y).pvalue, n_pairs=len(gens)))
    return P, pd.DataFrame(rows)


# ================================ 5. NATIVE-SPACE FITS (Fig 2) ================================
NATIVE = [("VWC", "theta", +1), ("pF", "SMP", -1), ("logK", "K(h)", +1), ("logphi", "MFP phi", +1)]

def rel_trans(g, fits):
    """Tr / Tr_max of every observation (plateau of its own plant)."""
    plateau = fits.set_index(["Soil", "Genotype", "Replicate"]).plateau
    return g.NormTrans.values / plateau.reindex(list(zip(g.Soil, g.Genotype, g.Replicate))).values

def observations_on_scales(d, fits, params):
    out = []
    for soil, g in d.groupby("Soil"):
        g = g.copy()
        g["rel"] = rel_trans(g, fits)
        g["pF"], g["logK"], g["logphi"] = convert(params[soil], g.VWC.values)
        out.append(g)
    return pd.concat(out, ignore_index=True)

def collapse_test(obs):
    """Response fitted against each descriptor per soil and pooled; max/mean vertical gap between the soil curves."""
    rows, keep = [], {}
    for key, nm, sgn in NATIVE:
        sub = obs[[key, "rel", "Soil"]].replace([np.inf, -np.inf], np.nan).dropna()
        x, y = sgn * sub[key].values, sub.rel.values
        p0 = [1.0, (y.max() - y.min()) / (x.max() - x.min() + 1e-9), np.median(x)]
        pc, _ = curve_fit(linear_plateau, x, y, p0=p0, bounds=([0.3, 1e-9, x.min()], [2.5, np.inf, x.max()]), maxfev=400_000)
        sst = np.sum((y - y.mean()) ** 2)
        r2c = 1 - np.sum((y - linear_plateau(x, *pc)) ** 2) / sst
        ps, ss = {}, 0.0
        for soil, g in sub.groupby("Soil"):
            xa, ya = sgn * g[key].values, g.rel.values
            ps[soil], _ = curve_fit(linear_plateau, xa, ya, p0=p0, bounds=([0.3, 1e-9, xa.min()], [2.5, np.inf, xa.max()]),
                                    maxfev=400_000)
            ss += np.sum((ya - linear_plateau(xa, *ps[soil])) ** 2)
        si, sa = sub.loc[sub.Soil == "Silt Loam", key], sub.loc[sub.Soil == "Sandy Loam", key]
        lo, hi = max(si.min(), sa.min()), min(si.max(), sa.max())
        ulo, uhi = min(si.min(), sa.min()), max(si.max(), sa.max())
        xx = np.linspace(lo, hi, 2001)
        sep = np.abs(np.clip(linear_plateau(sgn * xx, *ps["Sandy Loam"]), 0.0, 1.05)
                     - np.clip(linear_plateau(sgn * xx, *ps["Silt Loam"]), 0.0, 1.05))
        rows.append(dict(key=key, Scale=nm, sgn=sgn, sep_mean=float(sep.mean()), sep_max=float(sep.max()),
                         overlap_lo=float(lo), overlap_hi=float(hi), overlap_width=float(hi - lo),
                         overlap_pct_of_union=float(100.0 * max(0.0, hi - lo) / (uhi - ulo)),
                         R2_common=r2c, R2_separate=1 - ss / sst, dR2=(1 - ss / sst) - r2c,
                         breakpoint_gap=abs(ps["Silt Loam"][2] - ps["Sandy Loam"][2])))
        keep[key] = (sub, pc, ps, sgn, (lo, hi))
    return pd.DataFrame(rows), keep


# ================================ 6. TABLE S1 (all replicate curve pairings) ================================
def curve_pairings(fits, sets):
    sandy = [t for t, p in sets.items() if p["soil"] == "Sandy Loam"]
    silt = [t for t, p in sets.items() if p["soil"] == "Silt Loam"]

    def values(p, g, lab):
        if lab == "theta_crit":
            pf, lk, lp = convert(p, g.theta_crit.values)
            return {"VWC": g.theta_crit.values, "pF": pf, "logK": lk, "logphi": lp}
        return {k: np.asarray(v, float) for k, v in band_values(p, g).items()}
    rows = []
    for ta, tb in itertools.product(sandy, silt):
        for lab in ("theta_crit", BAND_LABEL):
            va = values(sets[ta], fits[fits.Soil == "Sandy Loam"], lab)
            vb = values(sets[tb], fits[fits.Soil == "Silt Loam"], lab)
            r = {"pair": f"{ta} x {tb}", "threshold": lab}
            for key, nm in SCALES:
                dv, _, na, nb = cohens_d(va[key], vb[key])
                r[f"d_{nm}"], r[f"n_{nm}"] = dv, f"{na}/{nb}"
            r["order_holds"] = bool(r["d_MFP phi"] < r["d_K(h)"] < r["d_SMP"])
            state = min(r.get("d_theta", np.inf), r["d_SMP"])
            r["supply_below_state"] = bool(r["d_MFP phi"] < state and r["d_K(h)"] < state)
            r["MFP_smallest"] = bool(r["d_MFP phi"] == min(r[f"d_{nm}"] for _, nm in SCALES))
            rows.append(r)
    ren = {f"{x}_{nm}": f"{x}_{_txt(nm)}" for _, nm in SCALES for x in ("d", "n")}
    return pd.DataFrame(rows).rename(columns={**ren, "MFP_smallest": f"{_txt('MFP phi')}_smallest"})


# ================================ 7. TABLE 1 ================================
def span_of_scale(p, key):
    """Range the variable covers between the two SPAN_PF limits (log10 units; cm3 cm-3 for theta)."""
    lo, hi = SPAN_PF
    mean_at = lambda col, pf: float(np.interp(pf, PF_GRID, p[col]))
    if key == "VWC":
        return abs(mean_at("th_grid", hi) - mean_at("th_grid", lo))
    if key == "pF":
        return hi - lo
    if key == "logK":
        return abs(mean_at("logK_grid", hi) - mean_at("logK_grid", lo))
    v = np.interp([lo, hi], PF_GRID, p["phi_grid"])
    return abs(np.log10(v[1]) - np.log10(v[0]))

def table_raw(T, params):
    """Raw threshold values, fold-difference and gap as % of the scale's span."""
    unit = {"VWC": "% VWC", "pF": "kPa", "logK": "cm/d", "logphi": "cm2/d"}
    rows = []
    for thr in T.Threshold.unique():
        s = T[T.Threshold == thr]
        for key, nm in [("VWC", "theta")] + list(SCALES):
            a, b = s.loc[s.Soil == "Sandy Loam", key].dropna(), s.loc[s.Soil == "Silt Loam", key].dropna()
            span = float(np.mean([span_of_scale(params[k], key) for k in params]))
            gap = abs(b.mean() - a.mean())
            if key == "VWC":
                sandy, silt = a.mean() * 100, b.mean() * 100
                fold, gap_rep, span_rep = max(a.mean(), b.mean()) / min(a.mean(), b.mean()), gap * 100, span * 100
            else:
                cv = (lambda v: 10 ** v * 0.0980665) if key == "pF" else (lambda v: 10 ** v)
                sandy, silt, fold, gap_rep, span_rep = cv(a.mean()), cv(b.mean()), 10 ** gap, gap, span
            rows.append(dict(Threshold=thr, Variable=_txt(nm), Unit=unit[key], Sandy=sandy, Silt=silt, fold=fold,
                             gap=gap_rep, span=span_rep, pct_of_span=100 * gap / span, n_sandy=len(a), n_silt=len(b)))
    return pd.DataFrame(rows)


# ================================ 8. TABLE WRITERS (HTML, PNG) ================================
CSS = ("body{font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#1f2933;margin:24px;background:#fff}"
       "h3{font-size:15px;font-weight:600;margin:0 0 4px}"
       "p.note{color:#6b7280;font-size:12px;margin:0 0 12px;max-width:900px;line-height:1.5}"
       "table{border-collapse:collapse;font-size:12.5px}"
       "th{background:#2b3a4a;color:#fff;font-weight:600;text-align:center;padding:8px 12px;border:1px solid #2b3a4a;white-space:nowrap}"
       "td{padding:7px 12px;border:1px solid #dfe3e8;text-align:center;white-space:nowrap}"
       "td.l{text-align:left}tr:nth-child(odd) td{background:#eef2f7}tr:nth-child(even) td{background:#fff}"
       "td.red{color:#c0392b}td.b{font-weight:600}")

def write_html_table(df, path, title, note="", highlight=(), floatfmt=None):
    """Self-contained HTML table; columns in `highlight` in red."""
    floatfmt = floatfmt or {}

    def cell(col, v):
        if isinstance(v, (float, np.floating)):
            txt = floatfmt.get(col, "{:.4g}").format(v) if np.isfinite(v) else ""
        else:
            txt = "" if v is None else str(v)
        cls = (["red"] if col in highlight else []) + ([] if isinstance(v, (int, float, np.floating)) else ["l"])
        return f'<td class="{" ".join(cls)}">{txt}</td>' if cls else f"<td>{txt}</td>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(cell(c, r[c]) for c in df.columns) + "</tr>" for _, r in df.iterrows())
    note_html = f'<p class="note">{note}</p>' if note else ""
    path.write_text(f"<style>{CSS}</style><h3>{title}</h3>{note_html}"
                    f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", encoding="utf-8")

def png_table(df, path, title, note="", floatfmt=None, red_cols=(), fontsize=8.0):
    """PNG table: dark header, zebra rows; red_cols in red (True/yes green, False/no red bold)."""
    d = df.copy()
    for c in d.columns:
        fmt = (floatfmt or {}).get(c)
        if fmt is not None:
            d[c] = [fmt.format(v) if pd.notna(v) else "" for v in d[c]]
        elif d[c].dtype.kind == "f":
            d[c] = [f"{v:.4g}" if pd.notna(v) else "" for v in d[c]]
        else:
            d[c] = d[c].astype(str)
    raw_w = [max(len(str(c)), *(len(v) for v in d[c])) + 2 for c in d.columns]
    row_h = 0.30
    fig_h = row_h * (len(d) + 1) + 0.95 + (0.30 if note else 0.0)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.105 * sum(raw_w)), fig_h), dpi=300)
    ax.axis("off")
    tbl = ax.table(cellText=d.values, colLabels=list(d.columns), colWidths=np.array(raw_w, float) / float(sum(raw_w)),
                   cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    red_idx = [i for i, c in enumerate(d.columns) if c in red_cols]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0.6)
        cell.set_height(row_h / (fig_h - 0.95))
        if r == 0:
            cell.set_facecolor("#2E3B4E")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        cell.set_facecolor("#FFFFFF" if r % 2 else "#EEF2F7")
        if c in red_idx:
            txt = str(d.iat[r - 1, c]).strip()
            if txt in {"True", "TRUE", "yes"}:
                cell.set_text_props(color="#1B6B4A", fontweight="bold")
            elif txt in {"False", "FALSE", "no"}:
                cell.set_text_props(color="#C0392B", fontweight="bold")
            else:
                cell.set_text_props(color="#C0392B")
        if c == 0:
            cell.set_text_props(ha="left")
            cell._text.set_x(0.02)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=fontsize + 2.2, fontweight="bold")
    if note:
        fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=fontsize - 0.8, color="#333333", wrap=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")

def per_genotype_table(PERGEN, log):
    """Table S2: per-genotype soil separation (as in Fig 1e,f)."""
    rows = []
    for thr in PERGEN.Threshold.unique():
        for gen in sorted(PERGEN.Genotype.unique()):
            r = {"Threshold": thr, "Genotype": gen}
            for _, nm in SCALES:
                q = PERGEN[(PERGEN.Threshold == thr) & (PERGEN.Scale == nm) & (PERGEN.Genotype == gen)]
                if len(q):
                    q, sy = q.iloc[0], _txt(nm)
                    r[f"{sy} sep."], r[f"{sy} SE"], r[f"{sy} sig."] = q.dg, q.se, q.sig
            rows.append(r)
    G = pd.DataFrame(rows)
    log("\nPer-genotype soil separation (Table S2), as plotted in Fig. 1e,f")
    log(tostr(G, ".3f"))
    n_sig = {_txt(nm): int((G[f"{_txt(nm)} sig."] != "ns").sum()) for _, nm in SCALES}
    log(f"  genotypes with a significant soil contrast (of {len(G)} genotype x threshold combinations): "
        + ", ".join(f"{nm} {n}" for nm, n in n_sig.items()))
    return G


# ================================ 9. FIGURES ================================
def save(fig, out):
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.name}")

def _dumbbell_panel(ax, thr, T, Y, CI, gens, letter, show_legend=False, show_xlabel=True):
    """Soil means placed around the soil-pair midpoint in pooled-SD units (distance = |d|), with bootstrap CI."""
    ticks, labs, y = [], [], 0
    s_ = T[T.Threshold == thr]
    for key, nm in SCALES:
        yr = Y[(Y.Threshold == thr) & (Y.key == key)].iloc[0]
        ci = CI[(CI.Threshold == thr) & (CI.key == key)].iloc[0]
        A, B = s_.loc[s_.Soil == "Sandy Loam", key], s_.loc[s_.Soil == "Silt Loam", key]
        mid = (A.mean() + B.mean()) / 2
        xa, xb = (A.mean() - mid) / yr.pooled_SD, (B.mean() - mid) / yr.pooled_SD
        c0, hw = (xa + xb) / 2, abs(xb - xa) / 2
        for seg in ([c0 - hw * ci.hi / ci.d, c0 - hw * ci.lo / ci.d], [c0 + hw * ci.lo / ci.d, c0 + hw * ci.hi / ci.d]):
            ax.plot(seg, [y, y], color=COL_CI, lw=9, alpha=.30, solid_capstyle="butt", zorder=2)
        ax.plot([xa, xb], [y, y], color="0.25", lw=2.6, zorder=3)
        for soil in SOIL_ORDER:
            gm = [s_.loc[(s_.Soil == soil) & (s_.Genotype == g), key].dropna().mean() for g in gens]
            ax.scatter((np.array(gm) - mid) / yr.pooled_SD, [y] * len(gm), s=170, color=COL_SOIL[soil],
                       edgecolor="black", lw=SPINE_W, zorder=5)
        ax.text(0.61, y, f"|d| = {ci.d:.2f}  [{ci.lo:.2f}, {ci.hi:.2f}]", transform=ax.get_yaxis_transform(),
                fontsize=FS - 4, va="center", fontweight="bold" if nm == "MFP phi" else "normal", color=COL_SCALE[nm])
        ticks.append(y)
        labs.append(_dsc(nm))
        y -= 1
    ax.axvline(0, color="0.45", ls=":", lw=2.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=FS_AXIS - 2)
    ax.set_xlim(-7.0, 16.2)
    ax.set_ylim(min(ticks) - 1.85, max(ticks) + 0.85)
    if show_xlabel:
        ax.set_xlabel("Distance from soil-pair midpoint (pooled within-soil SD)"
                      "   \u2192   line length = |Cohen's d|", fontsize=FS_AXIS)
    _spines(ax)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, ls=":", alpha=.35)
    _panel(ax, f"({letter})  {_dthr_b(thr)}", x=-0.075, y=1.22)
    if show_legend:
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL_SOIL[s], label=s, mec="black", mew=SPINE_W, ms=13)
                           for s in SOIL_ORDER] + [Line2D([], [], color=COL_CI, lw=9, alpha=.30, label="95% bootstrap CI of |d|")],
                  loc="lower left", bbox_to_anchor=(0.005, 0.005), ncol=3, frameon=True, framealpha=.92,
                  fontsize=FS_LEGEND - 3, handlelength=1.3, handletextpad=.45, labelspacing=.28, columnspacing=1.1,
                  borderaxespad=0)

def figure1(d, fits, T, Y, CI, PERGEN, out):
    """(a,b) dry-down curves per soil, (c,d) pooled soil separation per threshold, (e,f) per-genotype separation."""
    thresholds = list(T.Threshold.unique())
    gens = [g for g in GENO_ORDER if g in d.Genotype.unique()]
    gens += sorted(g for g in d.Genotype.unique() if g not in gens)
    fig = plt.figure(figsize=(19.0 * FIG_SCALE, 27.0 * FIG_SCALE * PUB_FIG_PAD * 0.58))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.22, 0.66, 0.66, 1.18], hspace=.42, wspace=.15,
                          left=.085, right=.985, top=.950, bottom=.042)
    pcols = ["plateau", "slope", "theta_crit"]
    for j, soil in enumerate(SOIL_ORDER):                       # (a), (b)
        ax = fig.add_subplot(gs[0, j])
        g = d[d.Soil == soil].copy()
        g["rel"] = rel_trans(g, fits)
        for gen in gens:
            gg = g[g.Genotype == gen]
            ax.scatter(gg.VWC * 100, gg.rel, s=20, alpha=.38, edgecolors="none", color=COL_GENO.get(gen, "#777777"), zorder=1)
        xs = np.linspace(g.VWC.min(), g.VWC.max(), 300)
        for gen in gens:                                        # solid over measured range, dotted beyond
            p = fits[(fits.Soil == soil) & (fits.Genotype == gen)][pcols].mean()
            col, gg = COL_GENO.get(gen, "#777777"), g[g.Genotype == gen]
            xo = np.linspace(gg.VWC.min(), gg.VWC.max(), 300)
            ax.plot(xs * 100, linear_plateau(xs, *p.values) / p.plateau, color=col, lw=1.3, alpha=.22, ls=(0, (1, 2)), zorder=2)
            ax.plot(xo * 100, linear_plateau(xo, *p.values) / p.plateau, color=col, lw=2.1, alpha=.75, zorder=3, label=gen)
        pm = fits[fits.Soil == soil][pcols].mean()
        ax.plot(xs * 100, linear_plateau(xs, *pm.values) / pm.plateau, color=COL_SOIL[soil], lw=5.0, zorder=4,
                label="Soil mean", solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=7.4, foreground="white"), pe.Normal()])
        ax.axvline(pm.theta_crit * 100, color=COL_ACC, ls="--", lw=2.2, zorder=2)
        ax.axhspan(LO, HI, color=COL_ACC, alpha=.18, lw=0, zorder=1)
        for v in (LO, HI):
            ax.axhline(v, color=COL_ACC, ls=":", lw=1.8, zorder=2, alpha=.75)
        ax.annotate(f"$\\mathbf{{\\theta_{{crit}}}}$ = {pm.theta_crit*100:.1f}%", xy=(pm.theta_crit * 100, HI + .14),
                    xytext=(-10, 0), textcoords="offset points", color=COL_ACC, fontsize=FS - 1, ha="right", va="bottom")
        ax.text(0.015, HI + .045, band_math(), transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=FS - 3, color=COL_ACC, style="italic")
        ax.invert_xaxis()
        ax.set_ylim(-.12, 1.34)
        ax.set_xlabel(r"$\theta$   (%, v/v)", fontsize=FS_AXIS)
        if j == 0:
            ax.set_ylabel(r"Tr / Tr$_{max}$", fontsize=FS_AXIS)
        _spines(ax)
        ax.yaxis.grid(True, ls=":", alpha=.35)
        _panel(ax, f"({'ab'[j]})  {soil}")
    fig.axes[1].legend(
        handles=[Line2D([], [], color=COL_GENO.get(g, "#777"), lw=3.0, label=g) for g in gens]
        + [Line2D([], [], color=COL_SOIL[s_], lw=5.0, label=f"{s_} mean") for s_ in SOIL_ORDER],
        loc="lower left", bbox_to_anchor=(0.005, 0.015), ncol=2, frameon=True, framealpha=.95, fontsize=FS_LEGEND - 6,
        handlelength=1.1, handletextpad=.35, labelspacing=.22, columnspacing=.7, borderaxespad=0)
    ymax = max(8.0, float((PERGEN.dg + PERGEN.se).max()) * 1.12)
    w = 0.80 / len(SCALES)
    off = (np.arange(len(SCALES)) - (len(SCALES) - 1) / 2) * w
    for j, thr in enumerate(thresholds):                        # (e), (f)
        ax = fig.add_subplot(gs[3, j])
        x = np.arange(len(gens))
        for i, (key, nm) in enumerate(SCALES):
            sub = PERGEN[(PERGEN.Threshold == thr) & (PERGEN.Scale == nm)].set_index("Genotype").loc[gens]
            ax.bar(x + off[i], sub.dg, w, yerr=sub.se, facecolor=COL_SCALE[nm], edgecolor="white", hatch=HATCH,
                   linewidth=BAR_EDGE_W, label=_dsc(nm), zorder=3,
                   error_kw=dict(ecolor="black", elinewidth=ELW, capsize=CAPSIZE, capthick=ELW))
            for q, (v, e, sg) in enumerate(zip(sub.dg, sub.se, sub.sig)):
                ax.text(q + off[i], v + e + ymax * .012, sg, ha="center", va="bottom", fontsize=FS - 6, rotation=90,
                        color=COL_ACC if sg == "ns" else "0.25", fontweight="bold" if sg == "ns" else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=18, ha="right", fontsize=FS_AXIS - 6)
        ax.set_ylim(0, ymax)
        if j == 0:
            ax.set_ylabel("Soil separation\n(pooled within-soil SD)", fontsize=FS_AXIS)
        _spines(ax)
        ax.yaxis.grid(True, ls=":", alpha=.35)
        _panel(ax, f"({'ef'[j]})  {_dthr_b(thr)}")
        if j == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=1, frameon=True, framealpha=.95,
                      fontsize=FS_LEGEND - 2, handlelength=1.2, handletextpad=.4, labelspacing=.28, borderaxespad=0)
    for j, thr in enumerate(thresholds):                        # (c), (d)
        _dumbbell_panel(fig.add_subplot(gs[1 + j, :]), thr, T, Y, CI, gens, letter="cd"[j], show_legend=(j == 0),
                        show_xlabel=(j == len(thresholds) - 1))
    save(fig, out)

def figure1f(PAIRS, WILC, out):
    """Paired ordering test: bootstrap CI of the difference in |d| + Wilcoxon p."""
    fig = plt.figure(figsize=(19.0 * FIG_SCALE, 27.0 * FIG_SCALE * PUB_FIG_PAD * 0.20))
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=.085, right=.985, top=.90, bottom=.20)[0, 0])
    yy, tk, tl, heads = 0, [], [], []
    tr = ax.get_yaxis_transform()
    for thr in PAIRS.Threshold.unique():
        heads.append((yy, thr))
        for k1, k2 in PAIR_LIST:
            cmp_ = f"{NAMES[k1]} vs {NAMES[k2]}"
            rb = PAIRS[(PAIRS.Threshold == thr) & (PAIRS.Comparison == cmp_)].iloc[0]
            rw = WILC[(WILC.Threshold == thr) & (WILC.Comparison == cmp_)].iloc[0]
            col = COL_CI if rb.hi < 0 else COL_ACC
            ax.plot([rb.lo, rb.hi], [yy, yy], color=col, lw=4.0, solid_capstyle="round", zorder=3)
            ax.scatter([rb["diff"]], [yy], s=190, color=col, edgecolor="black", lw=SPINE_W, zorder=5)
            pb = ("< " + f"{1/N_BOOT:.4f}") if rb.p_boot <= 1 / N_BOOT else f"= {rb.p_boot:.4f}"
            if rb.hi >= 0:
                ax.text(.735, yy, "n.s.", transform=tr, va="center", ha="right", fontsize=FS - 5, color=col, fontweight="bold")
            ax.text(.755, yy, f"$p_{{Wilcoxon}}$ = {rw.p_wilcoxon:.3f}", transform=tr, va="center", fontsize=FS - 5, color=col)
            ax.text(.890, yy, f"$p_{{boot}}$ {pb}", transform=tr, va="center", fontsize=FS - 5, color=col)
            tk.append(yy)
            tl.append(f"{_dsc(NAMES[k1])} vs {_dsc(NAMES[k2])}")
            yy -= 1
        yy -= 1.05
    ax.axvline(0, color=COL_ACC, lw=2.4, zorder=2)
    ax.set_yticks(tk)
    ax.set_yticklabels(tl, fontsize=FS_AXIS - 5)
    for y0, thr in heads:
        ax.text(-0.010, y0 + .95, _dthr_b(thr), fontsize=FS_AXIS - 4, fontweight="bold", va="center", ha="left",
                color="#222222", transform=tr)
    lo_x = float(PAIRS.lo.min()) - 0.30
    hi_x = max(float(PAIRS.hi.max()) + 0.30, 0.5)
    ax.set_xlim(lo_x, lo_x + (hi_x - lo_x) / 0.62)          # right third reserved for the p values
    ax.set_ylim(min(tk) - .9, max(tk) + 1.5)
    ax.set_xlabel("Paired difference in soil separation, SD units  "
                  "(negative = first descriptor brings the soils closer)", fontsize=FS_AXIS)
    _spines(ax)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, ls=":", alpha=.35)
    save(fig, out)

def _collapse_panel(ax, key, short, lab, COL, keep, letter, show_ylabel, highlight=False):
    """Tr/Trmax vs one descriptor: points + soil fits, shaded gap, arrow at the maximum gap."""
    sub, pc, ps, sgn, (lo, hi) = keep[key]
    r = COL[COL.key == key].iloc[0]
    for soil, g in sub.groupby("Soil"):
        ax.scatter(g[key], g.rel, s=42, color=COL_SOIL[soil], alpha=.55, edgecolors="none", zorder=2, label=soil)
    for soil, g in sub.groupby("Soil"):
        xa = sgn * g[key].values
        xx = np.linspace(xa.min(), xa.max(), 300)
        ax.plot(sgn * xx, linear_plateau(xx, *ps[soil]), color=COL_SOIL[soil], lw=4.4, zorder=5)
    xg = np.linspace(lo, hi, 400)
    ca = np.clip(linear_plateau(sgn * xg, *ps["Sandy Loam"]), 0.0, 1.05)
    cb = np.clip(linear_plateau(sgn * xg, *ps["Silt Loam"]), 0.0, 1.05)
    ax.fill_between(xg, ca, cb, color="0.45", alpha=.16, lw=0, zorder=1)
    xm = float(xg[int(np.argmax(np.abs(ca - cb)))])
    ya = float(np.clip(linear_plateau(sgn * xm, *ps["Sandy Loam"]), 0.0, 1.05))
    yb = float(np.clip(linear_plateau(sgn * xm, *ps["Silt Loam"]), 0.0, 1.05))
    col_arrow = COL_CI if highlight else "#333333"
    ax.annotate("", xy=(xm, ya), xytext=(xm, yb),
                arrowprops=dict(arrowstyle="<->", lw=2.6, color=col_arrow, shrinkA=0, shrinkB=0), zorder=7)
    ax.text(xm, (ya + yb) / 2, f"  {r.sep_max:.2f}", color=col_arrow, fontsize=FS - 2, fontweight="bold",
            va="center", ha="left", zorder=8, bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=.85))
    if sgn < 0:
        ax.invert_xaxis()
    ax.axhline(1, color="0.45", ls=":", lw=1.8)
    ax.set_ylim(-.16, 1.60)
    ax.set_xlabel(lab, fontsize=FS_AXIS)
    if show_ylabel:
        ax.set_ylabel(r"Tr / Tr$_{max}$", fontsize=FS_AXIS)
    col_txt = COL_CI if highlight else "0.15"
    ax.text(.035, .885, f"soils overlap over {r.overlap_pct_of_union:.0f}% of their combined range",
            transform=ax.transAxes, va="top", fontsize=FS_LEGEND - 1, color=col_txt)
    ax.text(.035, .800, f"max gap = {r.sep_max:.2f}", transform=ax.transAxes, va="top", fontsize=FS,
            fontweight="bold", color=col_txt)
    _spines(ax)
    ax.grid(True, ls=":", alpha=.32)
    _panel(ax, f"{letter}  {short}".strip())

def soil_legend():
    return ([Line2D([], [], marker="o", ls="", ms=13, color=COL_SOIL[s_], label=s_) for s_ in SOIL_ORDER]
            + [Patch(facecolor="0.45", alpha=.16, label="gap between soil curves")])

def figure2(COL, keep, out):
    panels = [("VWC", r"$\theta$", r"$\theta$   (cm$^3$ cm$^{-3}$)"),
              ("pF", r"$h$", r"$h$ as pF = log$_{10}$|h|   (h in cm)"),
              ("logK", r"$K(h)$", r"log$_{10}$ $K(h)$   (cm d$^{-1}$)"),
              ("logphi", r"$\Phi(h)$", r"log$_{10}$ $\Phi(h)$   (cm$^2$ d$^{-1}$)")]
    fig, axes = plt.subplots(2, 2, figsize=(17.0 * FIG_SCALE * 0.92, 15.0 * FIG_SCALE * PUB_FIG_PAD * 0.86))
    axes = axes.ravel()
    for j, (key, short, lab) in enumerate(panels):
        _collapse_panel(axes[j], key, short, lab, COL, keep, f"({'abcd'[j]})", show_ylabel=(j % 2 == 0),
                        highlight=(key == "logphi"))
    axes[0].legend(handles=soil_legend(), loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=True, framealpha=.92,
                   fontsize=FS_LEGEND - 2, handlelength=1.4, handletextpad=.45, labelspacing=.30, borderaxespad=0,
                   markerscale=1.6)
    fig.tight_layout(pad=1.4, w_pad=2.2)
    save(fig, out)


# ================================ TABLE NOTES ================================
S4_TITLE = "Table S4. Independence of the matric flux potential from the numerical integration scheme"
S4C_TITLE = "Table S4c. Sensitivity of the reported ordering to the dry reference limit of the matric flux potential"
S4_NOTE_HTML = ("&Phi; (cm<sup>2</sup> d<sup>-1</sup>) evaluated at the mean threshold suction of each soil, held "
                "fixed across rows so every value is a like-for-like recomputation. Every scheme integrates the same "
                "tabulated PDI conductivity function between the same limits and differs only in quadrature; the "
                "last column is the largest relative deviation from the reference scheme across both soils and both "
                "thresholds. All deviations are of order 10<sup>-3</sup>&nbsp;% or smaller, i.e. &Phi; is set by the "
                "conductivity function and not by the numerical treatment of the integral. The choice of dry "
                "reference limit is a separate matter and is reported in Table S4b.")
S4B_NOTE = ("&Phi; is defined relative to a dry reference, so moving that reference changes &Phi; by definition, "
            "not by numerical error; this is a different question from Table S4. Beyond the fitted range the "
            "conductivity is extrapolated log-linearly at its terminal slope. The integral converges because the "
            "film term decays as h<sup>-a</sup> with a &gt; 1, so extending the limit to pF 8, 10 or 12 adds a "
            "vanishing tail; pF 4.2, the permanent-wilting point, instead truncates it. The tail removed is "
            "&int;K dh between the two limits, which does not depend on the threshold and so enters as a "
            "constant offset per soil -- which is why the percentage change is largest where &Phi; is smallest. "
            "Because the analysis uses log<sub>10</sub>&Phi; and compares soils, the quantity that bears on the "
            "result is not this percentage but the change in the soil contrast, given in Table S4c.")
S4C_NOTE = ("The full analysis re-run with &Phi; referenced to each dry limit in turn; the pF %.1f row is the "
            "one reported in the paper. Raising the limit only lengthens the conductivity tail, so &theta;, h "
            "and K(h) cannot move and only |d| for &Phi;(h) responds. Lowering it to pF 4.2 truncates the "
            "integral at the permanent-wilting point and also raises the dry edge of the domain, so plants drier "
            "than &theta;(pF 4.2) leave the analysis altogether: the n column reports how many threshold values "
            "survive on the &Phi; scale, and any row where it falls is a different sample, not merely a "
            "different integral. order_holds is TRUE where |d| for &Phi;(h) &lt; K(h) &lt; h; supply_below_state "
            "is TRUE where both supply variables separate the soils less than both state variables." % PF_DRY)
S2_NOTE = ("Separation is the difference between soil means for that genotype, divided by the pooled "
           "within-soil standard deviation of the corresponding scale; SE is its standard error. Significance "
           "is a Welch t-test of that genotype's sandy loam against its silt loam plants, Holm-corrected across "
           "the six genotypes within each threshold and scale. *** P < 0.001, ** P < 0.01, * P < 0.05, ns not "
           "significant. These are the values plotted in Fig. 1e,f.")
S4_NOTE_PNG = ("Phi (cm2 d-1) evaluated at the mean threshold suction of each soil, held fixed across rows so every "
               "value is a like-for-like recomputation; these are not the threshold values reported in Table 1, "
               "which average log10 Phi across plants. Every scheme integrates the same tabulated PDI conductivity "
               "function between the same limits and differs only in quadrature. All deviations are of order 1e-3 % "
               "or smaller: Phi is set by the conductivity function, not by the numerical treatment of the "
               "integral. The choice of dry reference limit is a separate matter, reported in Table S4b.")
S1_NOTE = ("All sandy x silt combinations of the replicate PDI fits, at both thresholds. order_holds is TRUE "
           "where Φ(h) < K(h) < h; supply_below_state is TRUE where both supply variables separate the soils "
           "less than both state variables. d_θ is constant across pairings because water-content thresholds do "
           "not depend on the hydraulic fits.")


# ================================ MAIN ================================
LINES = []

def log(msg=""):
    print(msg)
    LINES.append(str(msg))

def main():
    if not DATA_DIR.exists():
        sys.exit(f"DATA_DIR does not exist or is not reachable:\n  {DATA_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 78)
    log("Transpiration thresholds:  \u03a6(h)  <  K(h)  <  h")
    log("=" * 78)
    log(f"data folder: {DATA_DIR}")

    log("\n[1] HYPROP PDI fits")
    sets = load_parameter_sets(DATA_DIR)
    params = {s: pooled_parameters(s, DATA_DIR) for s in sorted({p["soil"] for p in sets.values()})}
    for s, p in params.items():
        g = lambda k: p.get(k, np.nan)
        log(f"  {s:11s} [{p['source']}]: th_r={g('th_r'):.4f} th_s={g('th_s'):.4f} alpha={g('alpha'):.4f} "
            f"n={g('n'):.4f} Ks={g('Ks'):.1f} tau={g('tau'):+.4f} omega={g('omega'):.3e} a={g('a'):+.4f}")
        log(f"              theta spans {p['th_dry']:.4f}-{p['th_wet']:.4f} over pF {PF_MIN:.1f} to {PF_DRY:.1f}")
    validate_model({**sets, **{f'pooled {k}': v for k, v in params.items()}}, log)
    log("\n  FIG_CURVES = 'pooled': Fig 1 / Fig 2 / Table 1 use one joint fit per soil")

    log("\n[2] plant data")
    d = load_plant_data(PLANT_DIR)
    log("\n[3] linear-plateau fits in theta space")
    fits = fit_theta_space(d)

    log("\n[4] thresholds converted onto pF, log K, log phi  (PDI curves)")
    T = thresholds_on_scales(fits, params)
    Y = yardstick(T)
    log(tostr(Y[["Threshold", "Scale", "d", "pooled_SD", "n_sandy", "n_silt"]], ".4f"))

    log("\n[5] bootstrap and paired tests")
    CI, PAIRS = bootstrap_ordering(T)
    PERGEN, WILC = wilcoxon_pairs(T, Y)
    log(tostr(CI[["Threshold", "Scale", "d", "lo", "hi"]], ".3f"))
    log(tostr(PAIRS.merge(WILC, on=["Threshold", "Comparison"])
              [["Threshold", "Comparison", "diff", "lo", "hi", "p_boot", "p_wilcoxon"]], ".4f"))

    log("\n[6] native-space collapse test")
    COL, keep = collapse_test(observations_on_scales(d, fits, params))
    log(tostr(COL[["Scale", "sep_mean", "sep_max", "overlap_width", "breakpoint_gap", "R2_common", "R2_separate", "dR2"]], ".4f"))
    log("  sep_mean = mean vertical gap between the two soils' fitted curves,")
    log("  in units of Tr/Tr_max, over the range where both soils have data.")

    log("\n[7] all HYPROP curve pairings")
    n_sa = sum(1 for p in sets.values() if p["soil"] == "Sandy Loam")
    n_si = sum(1 for p in sets.values() if p["soil"] == "Silt Loam")
    log(f"  {n_sa} sandy x {n_si} silt = {n_sa*n_si} curve pairings")
    PAIR6 = curve_pairings(fits, sets)
    log(tostr(PAIR6, ".2f"))
    log(f"  both supply variables below both state variables in "
        f"{PAIR6.supply_below_state.sum()}/{len(PAIR6)} pairing x threshold combinations")
    log(f"  order \u03a6(h) < K(h) < h holds in {PAIR6.order_holds.sum()}/{len(PAIR6)} pairing x threshold combinations")

    log("\n[8] Table 1  raw values, fold-difference, span-normalised gap")
    TAB1 = table_raw(T, params)
    log(tostr(TAB1, ".4g"))

    log("\n[9] figures")
    figure1(d, fits, T, Y, CI, PERGEN, OUT_DIR / "Fig1_soil_separation.png")
    figure1f(PAIRS, WILC, OUT_DIR / "Fig1f_ordering_test.png")
    figure2(COL, keep, OUT_DIR / "Fig2_collapse_test.png")

    log("\n[10] integration audit -> Table S4")
    AUD, CONV = integration_audit(params, fits, log)
    DRY = dry_limit_sensitivity(params, fits, log)
    for df, name in ((TAB1, "Table1_raw_thresholds"), (AUD, "TableS4_integration_audit"),
                     (CONV, "TableS4b_upper_limit_convergence"), (DRY, "TableS4c_dry_limit_sensitivity")):
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    audfmt = {c: "{:.6f}" for c in AUD.columns if c != "Scheme"} | {"Max deviation (%)": "{:.2e}"}
    write_html_table(AUD, OUT_DIR / "TableS4_integration_audit.html", S4_TITLE, note=S4_NOTE_HTML,
                     highlight=("Max deviation (%)",), floatfmt=audfmt)
    chg = "Change vs pF %.1f (%%)" % PF_DRY
    tail = "Tail removed (cm<sup>2</sup> d<sup>-1</sup>)"
    write_html_table(CONV.rename(columns={"upper_limit_pF": "Upper limit (pF)", "phi": "&Phi;", "tail_removed": tail,
                                          "change_pct": chg, "film_exponent": "Film exponent a"}),
                     OUT_DIR / "TableS4b_upper_limit_convergence.html",
                     "Table S4b. Effect of the dry reference limit on the matric flux potential", note=S4B_NOTE,
                     floatfmt={"&Phi;": "{:.6f}", tail: "{:.6f}", chg: "{:.4f}", "Film exponent a": "{:.3f}",
                               "Upper limit (pF)": "{:.1f}"})
    s4cfmt = {c: "{:.3f}" for c in DRY.columns if DRY[c].dtype.kind == "f"}
    s4cfmt["Dry limit (pF)"] = "{:.1f}"
    write_html_table(DRY, OUT_DIR / "TableS4c_dry_limit_sensitivity.html", S4C_TITLE, note=S4C_NOTE, floatfmt=s4cfmt,
                     highlight=("order_holds", "supply_below_state"))
    png_table(DRY, OUT_DIR / "TableS4c_dry_limit_sensitivity.png", S4C_TITLE,
              note=re.sub(r"<[^>]+>", "", S4C_NOTE).replace("&Phi;", "Phi").replace("&theta;", "theta")
              .replace("&lt;", "<").replace("&gt;", ">"),
              floatfmt=s4cfmt, red_cols=("order_holds", "supply_below_state"))
    COL.drop(columns=["sgn"]).to_csv(OUT_DIR / "TableS3_curve_separation.csv", index=False)
    PAIR6.to_csv(OUT_DIR / "TableS1_curve_pairings.csv", index=False)
    T.to_csv(OUT_DIR / "thresholds_per_plant.csv", index=False)
    fits.to_csv(OUT_DIR / "linear_plateau_fits.csv", index=False)

    log("\\n[11] supplementary tables as PNG")
    GEN = per_genotype_table(PERGEN, log)
    GEN.to_csv(OUT_DIR / "TableS2_per_genotype_separation.csv", index=False)
    png_table(GEN, OUT_DIR / "TableS2_per_genotype_separation.png",
              "Table S2. Separation between soils for each genotype, on each hydraulic scale", note=S2_NOTE,
              floatfmt={f"{_txt(nm)} {x}": "{:.2f}" for _, nm in SCALES for x in ("sep.", "SE")},
              red_cols=tuple(f"{_txt(nm)} sig." for _, nm in SCALES))
    png_table(AUD, OUT_DIR / "TableS4_integration_audit.png", S4_TITLE, note=S4_NOTE_PNG, floatfmt=audfmt,
              red_cols=("Max deviation (%)",))
    png_table(PAIR6, OUT_DIR / "TableS1_curve_pairings.png",
              "Table S1. Soil separation for every pairing of replicate HYPROP fits", note=S1_NOTE,
              floatfmt={c: "{:.2f}" for c in PAIR6.columns if PAIR6[c].dtype.kind == "f"})
    (OUT_DIR / "stats_summary.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()