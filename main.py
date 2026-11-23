
import os, runpy
import numpy as np
import pandas as pd
import matplotlib as _mpl

# ================================ SETTINGS ================================
BASE = os.path.dirname(os.path.abspath(__file__))        # repository folder
CODE_DIR = os.path.join(BASE, "Input_main")              # code1, code2 and their input files
_find = lambda *names: next((os.path.join(CODE_DIR, n) for n in names if os.path.isfile(os.path.join(CODE_DIR, n))),
                            os.path.join(CODE_DIR, names[0]))
PATH_CODE1 = _find("Code1.py", "code1.py")
PATH_CODE2 = _find("Code2.py", "code2.py")
OUT_BASE = os.path.join(BASE, "Output", "main")
OUT_FINAL = os.path.join(OUT_BASE, "FinalResults")       # Excel with df_code1 / df_code2
OUT_FOREST = os.path.join(OUT_BASE, "Plot5by2give")      # above-ground forest plots
OUT_BELOW = os.path.join(OUT_BASE, "5by2_below")         # below-ground forest plots
OUT_SOIL = os.path.join(OUT_BASE, "Soil_effect")         # soil-effect figures
OUT_M5VSM3 = os.path.join(OUT_BASE, "M5vsM3")            # LowLate vs LowConst + Fig S7
OUT_DRYDOWN = os.path.join(OUT_BASE, "drydown")
for _d in (OUT_BASE, OUT_FINAL, OUT_FOREST, OUT_BELOW, OUT_SOIL, OUT_M5VSM3, OUT_DRYDOWN):
    os.makedirs(_d, exist_ok=True)
# publication fonts (every figure uses these)
PUB_FS_BUMP, PUB_FIG_GROWTH = 2, 1.0
PUB_FS_TICK, PUB_FS_AXIS, PUB_FS_PANEL, PUB_FS_LEGEND = 21 + PUB_FS_BUMP, 24 + PUB_FS_BUMP, 24 + PUB_FS_BUMP, 19 + PUB_FS_BUMP
PUB_FIG_PAD = 1.22
for _k in ("axes.spines.top", "axes.spines.right", "axes.spines.left", "axes.spines.bottom"):
    _mpl.rcParams[_k] = True                              # every panel framed on all 4 sides
_mpl.rcParams["axes.edgecolor"] = "black"

# ================================ [LOADER] ================================
df_code1 = runpy.run_path(PATH_CODE1)["merged"]           # code1 also sets its font defaults
df_code2 = runpy.run_path(PATH_CODE2)["combined_df"]
c1 = df_code1.copy()
c1['_Date'] = pd.to_datetime(c1['Date'], errors='coerce').dt.normalize()
in_nov = (c1['_Date'].dt.month == 11) & (c1['_Date'].dt.day.isin({28, 29, 30}))
window_rows_28_30 = c1.loc[in_nov & (c1['_Date'].dt.year == int(c1.loc[in_nov, '_Date'].dt.year.max()))].copy()
present_days_28_30 = sorted(window_rows_28_30['_Date'].dt.day.unique().tolist())
with pd.ExcelWriter(os.path.join(OUT_FINAL, "df_code1_and_code2_outputs.xlsx"), engine='openpyxl') as writer:
    df_code1.to_excel(writer, sheet_name='df_code1_full', index=False)
    window_rows_28_30.drop(columns=['_Date'], errors='ignore').to_excel(writer, sheet_name='df_code1_Nov28_30', index=False)
    df_code2.to_excel(writer, sheet_name='df_code2_full', index=False)
    df_code2[df_code2['Cam'] == 'Cam2'].to_excel(writer, sheet_name='df_code2_Cam2', index=False)
    for sheet in writer.sheets.values():                  # column width = longest entry (max 45)
        for col_cells in sheet.columns:
            width = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            sheet.column_dimensions[col_cells[0].column_letter].width = min(width + 3, 45)


# ================================ SHARED STATISTICS + TABLE PICTURE ================================
import re
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from scipy import stats as _scistats

TRANSFORMS = ("none", "log1p", "sqrt", "boxcox", "yeojohnson")
GENO_ORDER = ['Alpenfex', 'Braunes Schindelmeiser', 'Pfarrkirchner', 'Benedictio KWS', 'Luigi CS', 'P8888']
SOIL_COLORS = {"Silt Loam": "#8B5A2B", "Sandy Loam": "#C9A97C"}
GENO_COLORS = {'Alpenfex': '#2E7D32', 'Braunes Schindelmeiser': '#66A861', 'Pfarrkirchner': '#A5D6A7',   # landraces green
               'P8888': '#5E3C99', 'Luigi CS': '#8E71C1', 'Benedictio KWS': '#D6C6EA'}                   # hybrids purple

def _holm_adjust(pvals):
    p = np.asarray(pvals, float)
    if p.size == 0:
        return p
    running_max, adj = 0.0, np.empty_like(p)
    for i, idx in enumerate(np.argsort(p)):
        running_max = max(running_max, (p.size - i) * p[idx])
        adj[idx] = np.clip(running_max, 0, 1)
    return adj

def _stars(p):
    if p is None or not np.isfinite(p): return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""

def _fmt_p(x):
    if x is None or not np.isfinite(x): return ""
    return f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"

def _fmt_f(x, nd=3):
    return f"{x:.{nd}f}" if (x is not None and np.isfinite(x)) else ""

def _fmt_i(x):
    try: return str(int(x))
    except: return ""

def _partial_eta_sq(ss_effect, ss_error):
    if not all(map(np.isfinite, [ss_effect, ss_error])) or ss_error <= 0:
        return np.nan
    return float(ss_effect / (ss_effect + ss_error))

def _residual_diagnostics(fit, alpha_norm=0.05, alpha_hetero=0.05):
    """Shapiro (JB if n>5000) on residuals + Breusch-Pagan; assumptions met if both p >= alpha."""
    resid = np.asarray(fit.resid, float)
    resid = resid[np.isfinite(resid)]
    n = resid.size
    p_norm = (float(_scistats.shapiro(resid).pvalue) if n <= 5000 else float(jarque_bera(resid)[1])) if n >= 3 else np.nan
    try: p_het = float(het_breuschpagan(fit.resid, fit.model.exog)[1])
    except: p_het = np.nan
    return {"p_norm": p_norm, "p_hetero": p_het,
            "assumptions_met": bool(np.isfinite(p_norm) and p_norm >= alpha_norm and np.isfinite(p_het) and p_het >= alpha_hetero)}

def _apply_transform(y, name, short=False):
    """Transformed y (None if not possible) + {'transform', 'details'}  (short = wording of the later cells)."""
    y = np.asarray(y, float)
    ok = np.isfinite(y); y_ok = y[ok]
    if y_ok.size == 0: return y, {"transform": name, "details": "all_nan"}
    if name == "none": return y, {"transform": "none", "details": ""}
    if name in ("log1p", "sqrt"):
        if y_ok.min() < 0: return None, {"transform": name, "details": "min<0"}
        return np.where(ok, (np.log1p if name == "log1p" else np.sqrt)(y), np.nan), {"transform": name, "details": ""}
    if name == "boxcox":
        shift = max(0.0, 1.0 - y_ok.min())
        y_pos = y_ok + shift
        if y_pos.min() <= 0: return None, {"transform": "boxcox", "details": "no_pos" if short else "cannot_make_positive"}
        try:
            yt, lam = _scistats.boxcox(y_pos)
            out = np.full_like(y, np.nan, float); out[ok] = yt
            return out, {"transform": "boxcox", "details": f"sh={shift:.3g};lm={lam:.3g}" if short else f"shift={shift:.4g};lambda={lam:.4g}"}
        except Exception as e: return None, {"transform": "boxcox", "details": str(e) if short else f"failed:{type(e).__name__}"}
    if name == "yeojohnson":
        try:
            yt, lam = _scistats.yeojohnson(y_ok)
            out = np.full_like(y, np.nan, float); out[ok] = yt
            return out, {"transform": "yeojohnson", "details": f"lm={lam:.3g}" if short else f"lambda={lam:.4g}"}
        except Exception as e: return None, {"transform": "yeojohnson", "details": str(e) if short else f"failed:{type(e).__name__}"}
    return None, {"transform": name, "details": "unknown"}

def _fit_choose_transform(tmp, formula, transform_try_order=TRANSFORMS, alpha_norm=0.05, alpha_hetero=0.05):
    """OLS on the first transform meeting the assumptions, else the one with the highest p_norm + p_hetero."""
    y = tmp["Value"].to_numpy(float)
    best = None
    for tname in transform_try_order:
        yT, meta = _apply_transform(y, tname)
        if yT is None: continue
        work = tmp.copy(); work["Y"] = yT
        work = work.dropna(subset=["Y"])
        if work.shape[0] < 4: continue
        try:
            fit = smf.ols(formula, data=work).fit()
            diag = _residual_diagnostics(fit, alpha_norm, alpha_hetero)
            score = sum(v for v in [diag["p_norm"], diag["p_hetero"]] if np.isfinite(v))
            if diag["assumptions_met"]:
                return fit, tname, diag, meta, work
            if best is None or score > best["score"]:
                best = dict(fit=fit, tname=tname, diag=diag, meta=meta, work=work, score=score)
        except: continue
    return best["fit"], best["tname"], best["diag"], best["meta"], best["work"]

def _fit_choose(tmp, formula, tto=TRANSFORMS, an=0.05, ah=0.05, min_rows=6):
    """As _fit_choose_transform, but drops rows missing the model factors and needs >= min_rows rows (later cells)."""
    y = tmp["Value"].to_numpy(float); best = None
    for tn in tto:
        yT, meta = _apply_transform(y, tn, short=True)
        if yT is None: continue
        w = tmp.copy(); w["Y"] = yT
        w = w.dropna(subset=["Y"] + [c for c in ["Group", "Treatment", "Genotype"] if c in formula])
        if len(w) < min_rows: continue
        try:
            f = smf.ols(formula, data=w).fit(); d = _residual_diagnostics(f, an, ah)
            sc = sum(v for v in [d["p_norm"], d["p_hetero"]] if np.isfinite(v))
            if d["assumptions_met"]: return f, tn, d, meta, w
            if best is None or sc > best["sc"]: best = dict(f=f, tn=tn, d=d, meta=meta, w=w, sc=sc)
        except: continue
    return best["f"], best["tn"], best["d"], best["meta"], best["w"]

def _robust_term_pvalues_hc3(fit):
    """{term: HC3 Wald p}."""
    try:
        wt = fit.get_robustcov_results(cov_type="HC3").wald_test_terms(skip_single=False)
        if hasattr(wt, "pvalues") and hasattr(wt, "terms"):
            pvals, terms = np.asarray(wt.pvalues).ravel(), list(wt.terms)
            if len(terms) == len(pvals):
                return {str(t): float(p) for t, p in zip(terms, pvals)}
        tbl = getattr(wt, "table", None)
        if tbl is not None:
            pcol = next((c for c in ["P>F", "PR(>F)", "Pr>F", "Pr(>F)", "pvalue", "P>chi2", "PR(>chi2)", "Pr>chi2", "Pr(>chi2)",
                                     "P>Chi2", "PR(>Chi2)", "Pr>Chi2", "Pr(>Chi2)"] if c in tbl.columns), None)
            if pcol:
                return {str(t): float(tbl.loc[t, pcol]) for t in tbl.index}
    except Exception:
        pass
    return {}

def _lookup_term_p(pmap, target):
    if not pmap: return np.nan
    tl = target.lower()
    for k, v in pmap.items():
        if tl == str(k).lower() or tl in str(k).lower():
            return float(v)
    return np.nan

def _anova_p_for_term(fit, term, anova_type=2, robust_fallback=True, diag=None):
    """(p, method): ANOVA p if assumptions met, else HC3 robust p."""
    if diag is None: diag = _residual_diagnostics(fit)
    try:
        aov = anova_lm(fit, typ=anova_type)
        classical_p = float(aov.loc[term, "PR(>F)"]) if term in aov.index else np.nan
    except Exception:
        classical_p = np.nan
    if diag["assumptions_met"] or not robust_fallback:
        return classical_p, ("standard_anova" if diag["assumptions_met"] else "standard_anova_assump_failed")
    try:
        p = _lookup_term_p(_robust_term_pvalues_hc3(fit), term)
        if np.isfinite(p):
            return p, "robust_hc3"
    except Exception:
        pass
    return classical_p, "standard_anova_assump_failed"

def _effect_size_from_aov(fit, term, anova_type=2):
    """Partial eta squared of one term."""
    try:
        aov = anova_lm(fit, typ=anova_type)
        ss_e = float(aov.loc["Residual", "sum_sq"]) if "Residual" in aov.index else np.nan
        return _partial_eta_sq(float(aov.loc[term, "sum_sq"]), ss_e) if term in aov.index else np.nan
    except: return np.nan

def _cld_letters_from_pairwise(groups, pair_pvals_adj, alpha=0.05):
    """Compact letter display: groups sharing a letter do not differ (insert-and-absorb on the n.s. graph)."""
    k = len(groups)
    ns = np.eye(k, dtype=bool)
    uncovered = set()
    for (i, j), p in pair_pvals_adj.items():
        ns[i, j] = ns[j, i] = (p >= alpha)
        if p >= alpha: uncovered.add((i, j))
        else: ns[i, j] = ns[j, i] = False
    letters = {g: "" for g in groups}
    alphabet = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    li = 0
    while uncovered:
        counts = {}
        for a, b in uncovered:
            counts[a] = counts.get(a, 0) + 1; counts[b] = counts.get(b, 0) + 1
        clique = [max(counts, key=counts.get)]; changed = True
        while changed:
            changed = False
            cands = [(c, sum((tuple(sorted((c, u))) in uncovered) for u in clique))
                     for c in range(k) if c not in clique and all(ns[c, m] for m in clique)]
            if cands:
                clique.append(max(cands, key=lambda x: x[1])[0]); changed = True
        for n in clique: letters[groups[n]] += alphabet[li % len(alphabet)]
        for a, b in combinations(clique, 2): uncovered.discard(tuple(sorted((a, b))))
        li += 1
    for g in groups:
        if not letters[g]: letters[g] = alphabet[li % len(alphabet)]; li += 1
    return letters

def _wrap_words(text, max_chars):
    if len(text) <= max_chars: return [text]
    lines, cur = [], ""
    for w in text.split():
        if not cur: cur = w
        elif len(cur) + 1 + len(w) <= max_chars: cur += " " + w
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines or [text]

def save_df_as_table_png(df, out_png, title, footnote=None, font_size=11 + PUB_FS_BUMP, id_cols=None, dpi=200,
                         col_width_chars=None, wide={"Infer": 22, "Note": 28}, none_vals=(["none"], [""]), **_):
    """Table picture: dark header, zebra rows, id columns shaded/left-aligned, 'sig' columns red (unless 'none')."""
    def _clean(v):
        if v is None: return ""
        try:
            if np.isnan(float(v)): return ""
        except Exception: pass
        return str(v).strip()
    cols, nrows = list(df.columns), df.shape[0]
    ncols = len(cols)
    if id_cols is None: id_cols = cols[:2]
    wrap_w = [20 if c in id_cols else 30 if "sig" in c.lower() else wide.get(c, 14) for c in cols]
    for c, w in (col_width_chars or {}).items():
        if c in cols: wrap_w[cols.index(c)] = w
    hdr_lines = [_wrap_words(str(c), wrap_w[j]) for j, c in enumerate(cols)]
    body_lines = [[_wrap_words(_clean(v), wrap_w[j]) for j, v in enumerate(row)] for row in df.itertuples(index=False)]
    CHAR_W, LINE_H, PAD_H, PAD_W = font_size * 0.58 / 72, font_size * 1.50 / 72, font_size * 0.90 / 72, font_size * 0.80 / 72
    col_w_in = np.array([float(max(max(len(l) for l in hdr_lines[j]), *(max(len(l) for l in body_lines[i][j]) for i in range(nrows))))
                         * CHAR_W + 2 * PAD_W for j in range(ncols)])
    hdr_h = max(len(hdr_lines[j]) for j in range(ncols)) * LINE_H + PAD_H
    row_hs = np.array([max(len(body_lines[i][j]) for j in range(ncols)) * LINE_H + PAD_H for i in range(nrows)])
    MARGIN, TITLE_H, FOOT_H = 0.15, 0.40, (0.32 if footnote else 0.05)
    fig_w_in = float(col_w_in.sum()) + 2 * MARGIN
    fig_h_in = TITLE_H + hdr_h + float(row_hs.sum()) + FOOT_H + 2 * MARGIN
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w_in); ax.set_ylim(0, fig_h_in); ax.axis("off")
    ax.text(fig_w_in / 2, fig_h_in - MARGIN - TITLE_H / 2, title, ha="center", va="center", fontsize=font_size + 1.5,
            fontweight="bold", color="#1A1A2E")
    x_starts = np.concatenate([[MARGIN], MARGIN + np.cumsum(col_w_in)[:-1]])
    y_top = fig_h_in - MARGIN - TITLE_H

    def _cell(x, y_top_cell, w, h, lines, bg, fg, bold=False, ha="center"):
        ax.add_patch(mpatches.Rectangle((x, y_top_cell - h), w, h, linewidth=0.5, edgecolor="#BDC3C7", facecolor=bg, clip_on=False))
        for k, line in enumerate(lines):
            ax.text(x + PAD_W if ha == "left" else x + w / 2, y_top_cell - h / 2 + ((len(lines) - 1) / 2 - k) * LINE_H, line,
                    ha=ha, va="center", clip_on=False, fontproperties=FontProperties(size=font_size, weight="bold" if bold else "normal"),
                    color=fg)
    for j in range(ncols):
        _cell(x_starts[j], y_top, col_w_in[j], hdr_h, hdr_lines[j], "#2C3E50", "white", bold=True)
    y_cur = y_top - hdr_h
    for i in range(nrows):
        for j in range(ncols):
            is_id, val = cols[j] in id_cols, body_lines[i][j]
            sig_hit = "sig" in cols[j].lower() and val not in none_vals
            _cell(x_starts[j], y_cur, col_w_in[j], row_hs[i], val, "#EBF2FB" if is_id else ("#F4F6F9" if i % 2 == 0 else "white"),
                  "#C0392B" if sig_hit else "#2C3E50", bold=is_id, ha="left" if is_id else "center")
        y_cur -= row_hs[i]
    total_h = hdr_h + float(row_hs.sum())
    ax.add_patch(mpatches.Rectangle((MARGIN, y_top - total_h), col_w_in.sum(), total_h, linewidth=1.4, edgecolor="#2C3E50",
                                    facecolor="none", clip_on=False))
    if footnote:
        ax.text(MARGIN, FOOT_H / 2, footnote, ha="left", va="center", fontsize=max(7, font_size - 2), color="#555555",
                style="italic", clip_on=False)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return out_png

def _inset_title(ax, label, title_str, fs_panel, fs_title, *_, align='left'):
    """Bold panel letter + bold title in the top corner of the axes."""
    x, dx, ha = (0.98, -0.075, 'right') if align == 'right' else (0.02, 0.055, 'left')
    if label:
        ax.text(x, 0.97, label, transform=ax.transAxes, ha=ha, va='top', fontsize=fs_panel, fontweight='bold', clip_on=False, zorder=10)
    if title_str:
        ax.text(x + dx, 0.97, title_str, transform=ax.transAxes, ha=ha, va='top', fontsize=fs_title, fontweight='bold',
                clip_on=False, zorder=10)

def _contrast_p(test_fit, design, rowA, rowB, dmatrix=True):
    """p of the model contrast rowA - rowB (dicts of factor levels) from `test_fit` (OLS or HC3 results)."""
    mk = lambda r: pd.DataFrame({**{k: [v] for k, v in r.items()}, "Y": [0.0]})
    if dmatrix:
        dA, dB = (patsy.dmatrix(design, mk(r), return_type="matrix") for r in (rowA, rowB))
    else:
        dA, dB = (patsy.build_design_matrices([design], mk(r))[0] for r in (rowA, rowB))
    return float(test_fit.t_test(np.asarray(dA - dB, float).ravel()).pvalue)

# short names used by the later cells (same functions)
_holm, _diag, _p4t, _cld_pw, _eta2p, _fp, _ff, _fi = (_holm_adjust, _residual_diagnostics, _anova_p_for_term,
                                                     _cld_letters_from_pairwise, _partial_eta_sq, _fmt_p, _fmt_f, _fmt_i)


# ================================ [FOREST-ABOVE] 4x2 forest plots + Tables A1-A4 ================================
# ratios IntConst/OptConst (triangle) and LowConst/OptConst (square) per genotype and soil;
# stars = genotype simple effect, letters = genotype CLD (capitals IntConst, lowercase LowConst), both from Y ~ Group*Genotype
TREATS_A = ["Optimum", "Moderate Drought", "Severe Drought"]
SOILS = ["Silt Loam", "Sandy Loam"]

def canon_soil(s):
    if pd.isna(s): return np.nan
    x = str(s).strip().lower()
    return "Silt Loam" if "silt" in x else "Sandy Loam" if "sand" in x else str(s).strip()

def treat_from_raw(raw):
    if pd.isna(raw): return None
    s = str(raw).strip().lower()
    if s in ("no drought", "control") or any(k in s for k in ("optimum", "optimal", "well")): return "Optimum"
    if "moderate" in s or s == "m2": return "Moderate Drought"
    if "severe" in s or s == "m3": return "Severe Drought"
    return None

def _pair_data(rep_df, vcol, soil, treatA, treatB, labA, labB):
    """Rows of one soil and two treatments as Group / Genotype / Value."""
    sub = rep_df[(rep_df["Soil"] == soil) & (rep_df["Treat"].isin([treatA, treatB]))].copy()
    sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
    sub = sub.dropna(subset=["Genotype", "Treat", vcol])
    tmp = pd.DataFrame({"Group": sub["Treat"].map({treatA: labA, treatB: labB}).astype(str),
                        "Genotype": sub["Genotype"].astype(str), "Value": sub[vcol].to_numpy(float)})
    return sub, tmp

def _cld_from_factorial(rep_df, vcol, soil, treatA, treatB, labA, labB):
    """CLD letters of genotypes within each treatment (Holm-adjusted contrasts of Y ~ C(Group) * C(Genotype))."""
    sub, tmp = _pair_data(rep_df, vcol, soil, treatA, treatB, labA, labB)
    empty = {treatA: {}, treatB: {}}
    if sub.empty or sub["Genotype"].nunique() < 2 or sub["Treat"].nunique() < 2:
        return empty
    try:
        fit, tname, diag, meta, work = _fit_choose_transform(tmp, "Y ~ C(Group) * C(Genotype)")
    except Exception:
        return empty
    groups = sorted(work["Genotype"].unique().tolist())
    if len(groups) < 2:
        return empty
    result = {}
    for treat_level, group_label in [(treatA, labA), (treatB, labB)]:
        raw_ps, pairs = [], []
        for i, j in combinations(range(len(groups)), 2):
            gi, gj = groups[i], groups[j]
            try:
                p = _contrast_p(fit, fit.model.data.design_info, {"Group": group_label, "Genotype": gi},
                                {"Group": group_label, "Genotype": gj})
                p = p if np.isfinite(p) else np.nan
            except Exception:                    # fallback: one-way ANOVA of the two genotypes
                try:
                    pw = work[(work["Group"] == group_label) & (work["Genotype"].isin([gi, gj]))].copy()
                    if pw["Genotype"].nunique() < 2 or pw.shape[0] < 4:
                        p = 1.0
                    else:
                        pf = smf.ols("Y ~ C(Genotype)", data=pw).fit()
                        p_try, _ = _anova_p_for_term(pf, "C(Genotype)", 2, True, _residual_diagnostics(pf))
                        p = float(p_try) if np.isfinite(p_try) else 1.0
                except Exception:
                    p = 1.0
            raw_ps.append(float(p) if np.isfinite(p) else 1.0)
            pairs.append((i, j))
        result[treat_level] = _cld_letters_from_pairwise(groups, dict(zip(pairs, _holm_adjust(raw_ps))))
    return result

def _within_genotype_pvals(rep_df, vcol, soil, treatA, treatB, labA, labB):
    """{genotype: p of treatA vs treatB within that genotype} from Y ~ C(Group) * C(Genotype) (HC3 if assumptions fail)."""
    sub, _ = _pair_data(rep_df, vcol, soil, treatA, treatB, labA, labB)
    if sub.empty:
        return {}
    geno_ok = sub.groupby("Genotype")["Treat"].nunique() >= 2
    valid = geno_ok[geno_ok].index.astype(str).tolist()
    if not valid:
        return {}
    _, tmp = _pair_data(sub[sub["Genotype"].astype(str).isin(valid)], vcol, soil, treatA, treatB, labA, labB)
    try:
        fit, tname, diag, meta, work = _fit_choose_transform(tmp, "Y ~ C(Group) * C(Genotype)")
    except Exception:
        return {}
    test_fit = fit
    if not diag.get("assumptions_met", False):
        try: test_fit = fit.get_robustcov_results(cov_type="HC3")
        except Exception: test_fit = fit
    pvals = {}
    for g in sorted(work["Genotype"].astype(str).unique().tolist()):
        if g in valid:
            try:
                p = _contrast_p(test_fit, fit.model.data.design_info, {"Group": labA, "Genotype": g}, {"Group": labB, "Genotype": g})
                pvals[g] = p if np.isfinite(p) else np.nan
            except Exception:
                pvals[g] = np.nan
    return pvals

def _anova_overall_equal_weight(rep_df, vcol, soil, treatA, treatB):
    """Genotype means per treatment, blocked ANOVA Y ~ C(Genotype) + C(Treat2)."""
    sub = rep_df[(rep_df["Soil"] == soil) & (rep_df["Treat"].isin([treatA, treatB]))].copy()
    if sub.empty: return np.nan, "no_data", False, "", "", np.nan, np.nan, np.nan, 0, np.nan, np.nan
    sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
    sub = sub.dropna(subset=["Genotype", "Treat", vcol])
    gm = sub.groupby(["Genotype", "Treat"], as_index=False)[vcol].mean()
    piv = gm.pivot(index="Genotype", columns="Treat", values=vcol)
    if treatA not in piv.columns or treatB not in piv.columns:
        return np.nan, "missing_treat", False, "", "", np.nan, np.nan, np.nan, 0, np.nan, np.nan
    keep = piv.dropna(subset=[treatA, treatB]).index
    if len(keep) < 2:
        return np.nan, "too_few", False, "", "", np.nan, np.nan, np.nan, int(len(keep)), np.nan, np.nan
    meanA, meanB = float(np.nanmean(piv.loc[keep, treatA])), float(np.nanmean(piv.loc[keep, treatB]))
    work = gm[gm["Genotype"].isin(keep)].rename(columns={vcol: "Value"}).copy()
    work["Genotype"] = work["Genotype"].astype(str)
    work["Treat2"] = work["Treat"].map({treatA: "A", treatB: "B"}).astype("category")
    work = work.dropna(subset=["Treat2", "Value"])
    if work.shape[0] < 4:
        return np.nan, "insufficient_rows", False, "", "", np.nan, np.nan, np.nan, int(len(keep)), meanA, meanB
    try:
        fit, tname, diag, meta, _ = _fit_choose_transform(work, "Y ~ C(Genotype) + C(Treat2)")
        p, method = _anova_p_for_term(fit, "C(Treat2)", 2, True, diag)
        return (float(p) if np.isfinite(p) else np.nan, method, diag["assumptions_met"], tname, meta.get("details", ""),
                diag["p_norm"], diag["p_hetero"], _effect_size_from_aov(fit, "C(Treat2)"), int(len(keep)), meanA, meanB)
    except: return np.nan, "fit_failed", False, "", "", np.nan, np.nan, np.nan, int(len(keep)), meanA, meanB

def factorial_two_way(rep_df, vcol, soil, treatA, treatB, labA, labB):
    """Y ~ C(Group) * C(Genotype) within one soil: p for treatment, genotype and interaction."""
    sub, tmp = _pair_data(rep_df, vcol, soil, treatA, treatB, labA, labB)
    if sub.empty or sub["Genotype"].nunique() < 2 or sub["Treat"].nunique() < 2 or sub[vcol].count() < 6:
        return {"N": int(sub.shape[0]), "Transform": "", "Assump_OK": "", "p_norm": np.nan, "p_hetero": np.nan, "Method": "",
                "P_Group": np.nan, "P_Genotype": np.nan, "P_Group×Geno": np.nan}
    fit, tname, diag, _, work = _fit_choose_transform(tmp, "Y ~ C(Group) * C(Genotype)")
    p_g, _ = _anova_p_for_term(fit, "C(Group)", 2, True, diag)
    p_ge, _ = _anova_p_for_term(fit, "C(Genotype)", 2, True, diag)
    p_i, _ = _anova_p_for_term(fit, "C(Group):C(Genotype)", 2, True, diag)
    if not np.isfinite(p_i):
        p_i, _ = _anova_p_for_term(fit, "C(Genotype):C(Group)", 2, True, diag)
    fin = lambda p: float(p) if np.isfinite(p) else np.nan
    return {"N": int(work.shape[0]), "Transform": tname, "Assump_OK": "yes" if diag["assumptions_met"] else "no",
            "p_norm": diag["p_norm"], "p_hetero": diag["p_hetero"], "Method": "standard_anova" if diag["assumptions_met"] else "robust_hc3",
            "P_Group": fin(p_g), "P_Genotype": fin(p_ge), "P_Group×Geno": fin(p_i)}

def ratio_center_se(ratios):
    r = np.asarray(ratios, float); r = r[np.isfinite(r)]
    if r.size == 0: return np.nan, np.nan
    return float(np.nanmean(r)), (float(np.nanstd(r, ddof=1)) if r.size > 1 else 0.0) / np.sqrt(max(r.size, 1))

safe_ratio = lambda n, d: float(n / d) if (np.isfinite(n) and np.isfinite(d) and d != 0) else np.nan
fin_diff = lambda a, b: (a - b) if all(map(np.isfinite, [a, b])) else np.nan

def plot_area_avg2830_and_tradeoff_anova(window_rows_M2, df_code2, out_dir):
    FS_ALL, FS_AXIS, FS_PANEL, FS_LEGEND = PUB_FS_TICK, PUB_FS_AXIS, PUB_FS_PANEL, PUB_FS_LEGEND
    fig_scale, tag, star_dy = 1.30 * PUB_FIG_GROWTH, "AvgD28_30", 0.115
    latest_year = datetime.now().year
    # ---- replicate means: window rows (transpiration, area) and Cam2 (Psi_x, Ksp) ----
    wr = window_rows_M2.copy()
    wr["Soil"] = wr["Soil"].map(canon_soil)
    wr["Genotype"] = wr["Genotype"].astype(str).str.strip()
    wr["Rep"] = wr["Replicate"].astype(str).str.strip()
    wr["Treat"] = wr["Drought Category"].map(treat_from_raw)
    for c, src in (("_Trans", "Trans_for_norm"), ("_NormTrans", "NormTrans"), ("_Area", "Area_filled")):
        wr[c] = pd.to_numeric(wr[src], errors="coerce")
    wr = wr[(wr["Genotype"] != "Unknown") & (wr["Treat"].isin(TREATS_A)) & (wr["Soil"].isin(SOILS))]
    wr_rep = (wr.groupby(["Soil", "Genotype", "Treat", "Rep"], as_index=False)
              .agg(Trans=("_Trans", "mean"), NormTrans=("_NormTrans", "mean"), Area=("_Area", "mean")))
    df2 = df_code2[df_code2["Cam"].astype(str).str.strip() == "Cam2"].copy()
    rep_col = next(c for c in ["Replicate", "Rep"] if c in df2.columns)
    drt_col = next(c for c in ["DroughtLevel", "Drought Category", "Drought Treatment"] if c in df2.columns)
    hx_col = next(c for c in ("hXylem_Ψx", "hXylem_Psi_x", "hXylem_PsiX", "hXylem_Psi", "hXylem_psi_x") if c in df2.columns)
    df2["Soil"] = df2["Soil"].map(canon_soil)
    df2["Genotype"] = df2["Genotype"].astype(str).str.strip()
    df2["Rep"] = df2[rep_col].astype(str).str.strip()
    df2["Treat"] = df2[drt_col].map(treat_from_raw)
    df2["_Ksp"] = pd.to_numeric(df2["Ksp"], errors="coerce")
    df2["_Psi_plot"] = -pd.to_numeric(df2[hx_col], errors="coerce")
    df2 = df2[(df2["Genotype"] != "Unknown") & (df2["Treat"].isin(TREATS_A)) & (df2["Soil"].isin(SOILS))]
    df2_rep = df2.groupby(["Soil", "Genotype", "Treat", "Rep"], as_index=False).agg(Ksp=("_Ksp", "mean"), PsiPlot=("_Psi_plot", "mean"))
    universe = sorted(pd.unique(pd.concat([wr_rep["Genotype"], df2_rep["Genotype"]]).dropna()))
    geno_order = [g for g in GENO_ORDER if g in universe] + [g for g in universe if g not in GENO_ORDER]
    comps = [("Int vs Opt", "Moderate Drought", "Optimum", "IntConst", "OptConst"),
             ("Low vs Opt", "Severe Drought", "Optimum", "LowConst", "OptConst"),
             ("Low vs Int", "Severe Drought", "Moderate Drought", "LowConst", "IntConst")]
    metrics = [("Transpiration", wr_rep, "Trans"), ("Shoot Area", wr_rep, "Area"), ("hx,md", df2_rep, "PsiPlot"),
               ("Normalised Transpiration", wr_rep, "NormTrans"), ("Ksp", df2_rep, "Ksp")]

    def sig_effects(p_g, p_ge, p_i):
        parts = [f"{n} (p={_fmt_p(p)})" for n, p in (("Group", p_g), ("Genotype", p_ge), ("Group×Genotype", p_i)) if np.isfinite(p) and p < 0.05]
        return "; ".join(parts) if parts else "none"
    anova_rows, fold_rows, overall_rows = [], [], []
    for name, rep_df, vcol in metrics:
        for soil in SOILS:
            for comp, tA, tB, lA, lB in comps:
                res = factorial_two_way(rep_df, vcol, soil, tA, tB, lA, lB)
                anova_rows.append({"Parameter": name, "Soil": soil, "Comp": comp, **res,
                                   "Sig_effects": sig_effects(res["P_Group"], res["P_Genotype"], res["P_Group×Geno"])})

    def build_plot_tables(rep_df, vcol, metric_name):
        """Per soil: genotype ratio vs OptConst (mean +- SE), stars and letters; also adds the fold-change rows."""
        out = {}
        for soil in SOILS:
            rep_s = rep_df[rep_df["Soil"] == soil].copy()
            present = [g for g in geno_order if (rep_s["Genotype"] == g).any()]
            if rep_s.empty or not present:
                out[soil] = pd.DataFrame(columns=["Genotype", "r_mod", "se_mod", "r_sev", "se_sev", "p_mod", "p_sev", "cld_mod", "cld_sev"])
                continue
            cld_mod = {g: str(v).upper() for g, v in _cld_from_factorial(rep_s, vcol, soil, "Moderate Drought", "Optimum", "IntConst",
                                                                        "OptConst").get("Moderate Drought", {}).items()}
            cld_sev = _cld_from_factorial(rep_s, vcol, soil, "Severe Drought", "Optimum", "LowConst", "OptConst").get("Severe Drought", {})
            p_mod = _within_genotype_pvals(rep_s, vcol, soil, "Moderate Drought", "Optimum", "IntConst", "OptConst")
            p_sev = _within_genotype_pvals(rep_s, vcol, soil, "Severe Drought", "Optimum", "LowConst", "OptConst")
            rows = []
            for g in present:
                v = {t: pd.to_numeric(rep_s[(rep_s["Genotype"] == g) & (rep_s["Treat"] == t)][vcol], errors="coerce").dropna().to_numpy(float)
                     for t in TREATS_A}
                m = {t: float(np.nanmean(v[t])) if v[t].size else np.nan for t in TREATS_A}
                opt = m["Optimum"]
                ratios = {t: v[t] / opt if (np.isfinite(opt) and opt != 0 and v[t].size) else np.array([], float) for t in TREATS_A[1:]}
                r_mod, se_mod = ratio_center_se(ratios["Moderate Drought"])
                r_sev, se_sev = ratio_center_se(ratios["Severe Drought"])
                fold_rows.append({"Parameter": metric_name, "Soil": soil, "Genotype": g, "Mean_Optimum": opt,
                                  "Mean_Moderate": m["Moderate Drought"], "Mean_Severe": m["Severe Drought"],
                                  "Ratio_M2_over_M1": safe_ratio(m["Moderate Drought"], opt), "Ratio_M3_over_M1": safe_ratio(m["Severe Drought"], opt),
                                  "Delta_M2_minus_M1": fin_diff(m["Moderate Drought"], opt), "Delta_M3_minus_M1": fin_diff(m["Severe Drought"], opt)})
                rows.append({"Genotype": g, "r_mod": r_mod, "se_mod": se_mod, "r_sev": r_sev, "se_sev": se_sev,
                             "p_mod": float(p_mod.get(g, np.nan)), "p_sev": float(p_sev.get(g, np.nan)),
                             "cld_mod": cld_mod.get(g, ""), "cld_sev": cld_sev.get(g, "")})
            piv = rep_s.groupby(["Genotype", "Treat"])[vcol].mean().reset_index().pivot(index="Genotype", columns="Treat", values=vcol)
            if not piv.empty:
                o = {t: float(np.nanmean(piv[t].to_numpy())) if t in piv.columns else np.nan for t in TREATS_A}
                fold_rows.append({"Parameter": metric_name, "Soil": soil, "Genotype": "All genotypes (mean of genotype means)",
                                  "Mean_Optimum": o["Optimum"], "Mean_Moderate": o["Moderate Drought"], "Mean_Severe": o["Severe Drought"],
                                  "Ratio_M2_over_M1": safe_ratio(o["Moderate Drought"], o["Optimum"]),
                                  "Ratio_M3_over_M1": safe_ratio(o["Severe Drought"], o["Optimum"]),
                                  "Delta_M2_minus_M1": fin_diff(o["Moderate Drought"], o["Optimum"]),
                                  "Delta_M3_minus_M1": fin_diff(o["Severe Drought"], o["Optimum"])})
            out[soil] = pd.DataFrame(rows)
        return out
    ratios = {name: build_plot_tables(rep_df, vcol, label) for name, rep_df, vcol, label in
              [("trans", wr_rep, "Trans", "Transpiration"), ("area", wr_rep, "Area", "Shoot Area"),
               ("psix", df2_rep, "PsiPlot", "Xylem Water Potential (−hx,md)"), ("nt", wr_rep, "NormTrans", "Normalised Transpiration"),
               ("ksp", df2_rep, "Ksp", "Ksp")]}
    for name, rep_df, vcol in metrics:
        for soil in SOILS:
            for comp, tA, tB, *_ in comps:
                p, method, ass, tname, tdet, pn, ph, eta, nG, mA, mB = _anova_overall_equal_weight(rep_df, vcol, soil, tA, tB)
                overall_rows.append({"Parameter": name, "Soil": soil, "Comparison": comp, "Treatment_A": tA, "Treatment_B": tB,
                                     "n_genotypes": nG, "Mean_A": mA, "Mean_B": mB, "Difference_A_minus_B": fin_diff(mA, mB),
                                     "Transform_used": tname, "Transform_details": tdet, "resid_p_normality": pn, "resid_p_hetero_BP": ph,
                                     "Assumptions_met": ass, "Inference_method": method, "p_value": p, "eta2p": eta, "Stars": _stars(p),
                                     "Note": "assumptions failed; HC3 robust p-values" if method == "robust_hc3" else ""})

    # ---- forest plot 4 x 2 ----
    panels = [("Transpiration", ratios["trans"]), ("Shoot Area", ratios["area"]), ("hx,md", ratios["psix"]),
              ("Normalised Transpiration", ratios["nt"])]
    n_rows = len(panels)
    rc = {"font.size": FS_ALL, "axes.titlesize": FS_AXIS, "axes.labelsize": FS_AXIS, "xtick.labelsize": FS_ALL,
          "ytick.labelsize": FS_AXIS, "legend.fontsize": FS_LEGEND, "figure.titlesize": FS_AXIS}
    ekw = dict(capsize=6, linestyle="none", ecolor="black", elinewidth=1.6, capthick=1.6, zorder=3)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_rows, 2, figsize=(20.0 * fig_scale * 1.10, (4.2 * n_rows + 1.6) * fig_scale * PUB_FIG_PAD),
                                 constrained_layout=False)
        for r, (title, ratio_dict) in enumerate(panels):
            # x-limits from the data (+-SE), 1.0 always visible, extra room on the side where letters go
            lo_d, hi_d, need_left, need_right = [], [], False, False
            for soil in SOILS:
                dfp = ratio_dict.get(soil, pd.DataFrame())
                if dfp is None or dfp.empty: continue
                for rr, ss in [(dfp["r_mod"].to_numpy(float), dfp["se_mod"].to_numpy(float)),
                               (dfp["r_sev"].to_numpy(float), dfp["se_sev"].to_numpy(float))]:
                    se, fin = np.nan_to_num(ss, nan=0.0), np.isfinite(rr)
                    if not fin.any(): continue
                    lo_d.append(float(np.min((rr - se)[fin]))); hi_d.append(float(np.max((rr + se)[fin])))
                    need_left |= bool((rr[fin] < 1.0).any()); need_right |= bool((rr[fin] >= 1.0).any())
            d_lo, d_hi = min(min(lo_d) if lo_d else 0.5, 1.0), max(max(hi_d) if hi_d else 1.5, 1.0)
            span = max(d_hi - d_lo, 1e-6)
            lo, hi = d_lo - 0.06 * span, d_hi + 0.06 * span
            room = 0.13 * (hi - lo)
            xlim = (lo - room if need_left else lo, hi + room if need_right else hi)
            margin = 0.015 * (xlim[1] - xlim[0])
            for c, soil in enumerate(SOILS):
                ax, dfp = axes[r, c], ratio_dict.get(soil, pd.DataFrame())
                label = f"({chr(97 + r * 2 + c)})"
                ax.axvline(1.0, color="black", linewidth=1.2, zorder=0)
                if dfp is None or dfp.empty:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=FS_AXIS)
                    ax.set_yticks([]); ax.set_xlim(xlim)
                    ax.grid(axis="x", linestyle=":", alpha=0.5)
                    _inset_title(ax, label, f"{title} : {soil}", FS_PANEL, FS_AXIS)
                    continue
                genes = dfp["Genotype"].tolist()
                y = np.arange(len(genes))
                r_mod, se_mod = dfp["r_mod"].to_numpy(float), dfp["se_mod"].to_numpy(float)
                r_sev, se_sev = dfp["r_sev"].to_numpy(float), dfp["se_sev"].to_numpy(float)
                sc = SOIL_COLORS.get(soil, "#888888")
                ax.errorbar(r_mod, y + 0.15, xerr=se_mod, fmt="^", markersize=9, color=sc, label=r"Int$_{Const}$/Opt$_{Const}$", **ekw)
                ax.errorbar(r_sev, y - 0.15, xerr=se_sev, fmt="s", markersize=10, color=sc, label=r"Low$_{Const}$/Opt$_{Const}$", **ekw)
                for i in range(len(genes)):
                    for yi, ri, sei, p, key, dy, va in [(y[i] + 0.15, r_mod[i], se_mod[i], float(dfp.loc[i, "p_mod"]), "cld_mod", star_dy, "bottom"),
                                                        (y[i] - 0.15, r_sev[i], se_sev[i], float(dfp.loc[i, "p_sev"]), "cld_sev", -star_dy, "top")]:
                        if _stars(p) and np.isfinite(ri):
                            ax.text(ri, yi + dy, _stars(p), ha="center", va=va, fontsize=FS_AXIS, clip_on=True)
                        ltr, se_v = str(dfp.loc[i, key] or ""), (sei if np.isfinite(sei) else 0.0)
                        if ltr and np.isfinite(ri):         # letters on the far side from 1.0
                            ax.text((ri + se_v + margin) if ri >= 1.0 else (ri - se_v - margin), yi, ltr,
                                    ha="left" if ri >= 1.0 else "right", va="center", fontsize=FS_AXIS, clip_on=True)
                ax.set_ylim(-0.85, (len(genes) - 1) + 0.15 + star_dy + 0.35 + 1.45)
                ax.set_yticks(y)
                ax.set_yticklabels(genes if c == 0 else [], fontsize=FS_AXIS)
                ax.set_xlim(xlim)
                ax.grid(axis="x", linestyle=":", alpha=0.5)
                ax.tick_params(axis="x", labelsize=FS_ALL, width=1.2, length=5)
                ax.tick_params(axis="y", labelsize=FS_AXIS, width=1.2, length=5)
                if r == n_rows - 1:
                    ax.set_xlabel(r"Relative change from Opt$_{Const}$", fontsize=FS_AXIS)
                _inset_title(ax, label, f"{title} : {soil}", FS_PANEL, FS_AXIS)
                if r == 0 and c == 0:
                    ax.legend(handles=[Line2D([0], [0], marker="^", linestyle="none", markersize=9, color="black", label=r"Int$_{Const}$/Opt$_{Const}$"),
                                       Line2D([0], [0], marker="s", linestyle="none", markersize=10, color="black", label=r"Low$_{Const}$/Opt$_{Const}$")],
                              loc="lower right", fontsize=FS_LEGEND, frameon=True, framealpha=0.9)
                if r == 0 and c == 1:
                    ax.legend(handles=[Patch(facecolor=SOIL_COLORS[s], edgecolor="black", linewidth=0.8, label=s) for s in SOILS],
                              loc="upper right", fontsize=FS_LEGEND, frameon=True, framealpha=0.9, handlelength=1.2, handletextpad=0.4,
                              labelspacing=0.3)
        plt.tight_layout(rect=[0.02, 0.03, 0.995, 0.980], h_pad=2.6, w_pad=2.2)
        fig.canvas.draw()
        fig.suptitle(f"Forest Plots (4x2): IntConst/OptConst & LowConst/OptConst ({latest_year})", fontsize=FS_AXIS, y=0.998)
        plt.savefig(os.path.join(out_dir, f"code1_forest_4x2_ratios_mod_sev_vs_opt_ANOVA_{tag}_{latest_year}.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    # ---- Tables A1-A4 ----
    rename_par = lambda m: {"Ksp (soil–plant hydraulic conductance)": "Ksp", "Normalised Transpiration": "Trans_Norm",
                            "Xylem Water Potential (−hx,md)": "hx,md"}.get(str(m).strip(), str(m).strip())
    rename_geno = lambda g: "Braues.S" if str(g).strip() == "Braunes Schindelmeiser" else str(g).strip()
    treat_short = lambda t: {"Optimum": "OptConst", "Moderate Drought": "IntConst", "Severe Drought": "LowConst"}.get(str(t).strip(), str(t).strip())

    def panel_for(parameter, soil):
        p = str(parameter).strip()
        p = "hx,md" if ("Xylem Water Potential" in p or p == "hx,md") else p
        order = ["Transpiration", "Shoot Area", "hx,md", "Normalised Transpiration"]
        return f"({chr(97 + 2 * order.index(p) + (0 if str(soil).strip() == 'Silt Loam' else 1))})" if p in order else ""
    def fmt_cols(d, spec):
        for c, nd in spec:
            d[c] = pd.to_numeric(d[c], errors="coerce").apply(lambda x: _fmt_f(x, nd) if np.isfinite(x) else "")
    add_panel = lambda d: d.apply(lambda r: panel_for(r["Parameter"], r["Soil"]), axis=1)
    path = lambda n: os.path.join(out_dir, f"Table_{n}_{tag}_{latest_year}.png")
    anova_df, fold_df, overall_df = pd.DataFrame(anova_rows), pd.DataFrame(fold_rows), pd.DataFrame(overall_rows)

    d = anova_df.copy()
    for col in ["p_norm", "p_hetero", "P_Group", "P_Genotype", "P_Group×Geno"]:
        d[col] = d[col].apply(_fmt_p)
    d["N"] = d["N"].apply(_fmt_i)
    d["Panel"] = add_panel(d)
    d = d[["Panel", "Parameter", "Soil", "Comp", "N", "Transform", "Method", "p_norm", "p_hetero", "Assump_OK", "P_Group", "P_Genotype",
           "P_Group×Geno", "Sig_effects"]].rename(columns={"Transform": "Transf", "Method": "Infer", "p_norm": "p(norm)", "p_hetero": "p(BP)",
                                                         "Assump_OK": "Assump", "P_Group": "p(Treatment)", "P_Genotype": "p(Geno)",
                                                         "P_Group×Geno": "p(Int)", "Sig_effects": "Sig effects"})
    d["Parameter"] = d["Parameter"].apply(rename_par)
    save_df_as_table_png(d, path("A1_FactorialANOVA"),
                         title="Table A1. Two-way factorial ANOVA within soil (Int vs Opt; Low vs Opt; Low vs Int)",
                         footnote=("Per soil and comparison: Y ~ C(Group) * C(Genotype), replicate means. Assumptions: Shapiro/JB p≥0.05, "
                                   "BP p≥0.05. Fallback: HC3 robust Wald p-values (robust_fallback=True). CLD letters in plots = pairwise "
                                   "genotype contrasts from this model (capitals = IntConst, lowercase = LowConst)."),
                         font_size=13, id_cols=["Panel", "Parameter", "Soil", "Comp"])

    d2 = fold_df.copy()
    fmt_cols(d2, [("Mean_Optimum", 4), ("Mean_Moderate", 4), ("Mean_Severe", 4), ("Ratio_M2_over_M1", 3), ("Ratio_M3_over_M1", 3),
                  ("Delta_M2_minus_M1", 4), ("Delta_M3_minus_M1", 4)])
    d2["Panel"] = add_panel(d2)
    d2 = d2[["Panel", "Parameter", "Soil", "Genotype", "Mean_Optimum", "Mean_Moderate", "Mean_Severe", "Ratio_M2_over_M1", "Ratio_M3_over_M1",
             "Delta_M2_minus_M1", "Delta_M3_minus_M1"]].rename(columns={
        "Genotype": "Geno", "Mean_Optimum": "Mean OptE", "Mean_Moderate": "Mean IntE", "Mean_Severe": "Mean LowE", "Ratio_M2_over_M1": "Int/Opt",
        "Ratio_M3_over_M1": "Low/Opt", "Delta_M2_minus_M1": "Δ(Int−Opt)", "Delta_M3_minus_M1": "Δ(Low−Opt)"})
    d2["Parameter"], d2["Geno"] = d2["Parameter"].apply(rename_par), d2["Geno"].apply(rename_geno)
    save_df_as_table_png(d2, path("A2_FoldChanges"), title="Table A2. Genotype-wise means, ratios and deltas vs M1 (within soil)",
                         footnote="Means from replicate means. Overall = mean of genotype means. Ratios = IntConst/OptConst or LowConst/OptConst; Δ = Int−Opt or Low−Opt.",
                         font_size=13, id_cols=["Panel", "Parameter", "Soil", "Geno"])

    base = fold_df[~fold_df["Genotype"].astype(str).str.contains("All genotypes", na=False)].copy()
    for c in ("Mean_Moderate", "Mean_Severe"):
        base[c] = pd.to_numeric(base[c], errors="coerce")
    grp = base.groupby(["Parameter", "Soil"], as_index=False).agg(
        AllGenMean_M2=("Mean_Moderate", "mean"), AllGenMean_M3=("Mean_Severe", "mean"),
        nG_M2=("Mean_Moderate", lambda x: int(pd.Series(x).dropna().size)), nG_M3=("Mean_Severe", lambda x: int(pd.Series(x).dropna().size)))
    s3 = base.merge(grp, on=["Parameter", "Soil"], how="left")
    for mean, allm, ratio, pct in [("Mean_Moderate", "AllGenMean_M2", "Ratio_M2_vs_AllMean", "PctDiff_M2"),
                                   ("Mean_Severe", "AllGenMean_M3", "Ratio_M3_vs_AllMean", "PctDiff_M3")]:
        s3[ratio] = np.where((s3[allm] != 0) & s3[allm].apply(np.isfinite), s3[mean] / s3[allm], np.nan)
        s3[pct] = (s3[ratio] - 1.0) * 100.0
    direction = lambda r: "Increase" if r > 1 else "Decrease" if r < 1 else "No change" if np.isfinite(r) else ""
    s3["Dir_M2"], s3["Dir_M3"] = s3["Ratio_M2_vs_AllMean"].apply(direction), s3["Ratio_M3_vs_AllMean"].apply(direction)
    fmt_cols(s3, [("Mean_Moderate", 4), ("AllGenMean_M2", 4), ("Ratio_M2_vs_AllMean", 3), ("PctDiff_M2", 1), ("Mean_Severe", 4),
                  ("AllGenMean_M3", 4), ("Ratio_M3_vs_AllMean", 3), ("PctDiff_M3", 1)])
    s3["Panel"] = add_panel(s3)
    s3 = (s3[["Panel", "Parameter", "Soil", "Genotype", "Mean_Moderate", "AllGenMean_M2", "Ratio_M2_vs_AllMean", "PctDiff_M2", "Dir_M2",
              "Mean_Severe", "AllGenMean_M3", "Ratio_M3_vs_AllMean", "PctDiff_M3", "Dir_M3", "nG_M2", "nG_M3"]]
          .rename(columns={"Genotype": "Geno", "Mean_Moderate": "Mean IntE", "AllGenMean_M2": "AllMean IntE", "Ratio_M2_vs_AllMean": "Ratio Int",
                           "PctDiff_M2": "%Δ Int", "Mean_Severe": "Mean LowE", "AllGenMean_M3": "AllMean LowE", "Ratio_M3_vs_AllMean": "Ratio Low",
                           "PctDiff_M3": "%Δ Low"}).sort_values(["Parameter", "Soil", "Geno"]).reset_index(drop=True))
    s3["Parameter"], s3["Geno"] = s3["Parameter"].apply(rename_par), s3["Geno"].apply(rename_geno)
    save_df_as_table_png(s3, path("A3_Genotype_vs_AllMean"),
                         title="Table A3. Genotype deviation vs mean of genotype means (within soil), M2 and M3",
                         footnote="AllMean = equal-weight mean of genotype means. Ratio = Genotype/AllMean; %Δ = (Ratio−1)×100.",
                         font_size=13, id_cols=["Panel", "Parameter", "Soil", "Geno"])

    d4 = overall_df.copy()
    d4["n_genotypes"] = d4["n_genotypes"].apply(_fmt_i)
    for c in ("resid_p_normality", "resid_p_hetero_BP", "p_value"):
        d4[c] = d4[c].apply(_fmt_p)
    d4["Assumptions_met"] = d4["Assumptions_met"].apply(lambda v: "Yes" if bool(v) else "No")
    d4["eta2p"] = d4["eta2p"].apply(lambda x: _fmt_f(x, 3))
    fmt_cols(d4, [("Mean_A", 4), ("Mean_B", 4), ("Difference_A_minus_B", 4)])
    d4["Panel"] = add_panel(d4)
    d4 = (d4[["Panel", "Parameter", "Soil", "Comparison", "Treatment_A", "Treatment_B", "n_genotypes", "Mean_A", "Mean_B",
              "Difference_A_minus_B", "Transform_used", "Inference_method", "resid_p_normality", "resid_p_hetero_BP", "Assumptions_met",
              "p_value", "Stars", "eta2p"]]
          .rename(columns={"Comparison": "Comp", "Treatment_A": "Trt A", "Treatment_B": "Trt B", "n_genotypes": "nG", "Mean_A": "Mean A",
                           "Mean_B": "Mean B", "Difference_A_minus_B": "Diff", "Transform_used": "Transf", "Inference_method": "Infer",
                           "resid_p_normality": "p(norm)", "resid_p_hetero_BP": "p(BP)", "Assumptions_met": "Assump", "p_value": "p",
                           "eta2p": "η²p"}).sort_values(["Parameter", "Soil", "Comp"]).reset_index(drop=True))
    d4["Parameter"], d4["Comp"] = d4["Parameter"].apply(rename_par), d4["Comp"].astype(str)
    d4["Trt A"], d4["Trt B"] = d4["Trt A"].apply(treat_short), d4["Trt B"].apply(treat_short)
    save_df_as_table_png(d4, path("A4_Overall_EqualWeightGenotypes"),
                         title="Table A4. Overall drought comparisons (equal-weight genotype means), per soil",
                         footnote="Equal-weight genotypes: genotype means per treatment, blocked ANOVA Y ~ C(Genotype) + C(Treat2) testing C(Treat2).",
                         font_size=13, id_cols=["Panel", "Parameter", "Soil", "Comp"])

plot_area_avg2830_and_tradeoff_anova(window_rows_28_30, df_code2, OUT_FOREST)


# ================================ [FOREST-BELOW] root-trait forest plots (Cam2) + Tables B1-B4 ================================
SOIL_SILT, SOIL_SAND = 'Silt Loam', 'Sandy Loam'
DL_OPT, DL_INT, DL_CONT = 'Optimum', 'Moderate Drought', 'Severe Drought'      # OptConst, IntConst, LowConst
B_METRICS = ('trl', 'rdmean', 'srm_upscaled')
B_LABEL = {'trl': 'Total Root Length', 'rdmean': 'Root Diameter', 'srm_upscaled': 'Rhizosheath mass'}   # figure / fold tables
B_ANOVA_LABEL = {"trl": "TRL", "rdmean": "rd", "srm_upscaled": "Rhizosheath"}                          # ANOVA tables
B_COMPS = [("Int vs Opt", DL_INT, DL_OPT, "IntConst", "OptConst"), ("Low vs Opt", DL_CONT, DL_OPT, "LowConst", "OptConst"),
           ("Low vs Int", DL_CONT, DL_INT, "LowConst", "IntConst")]

def _canon_drought(s):
    if pd.isna(s): return None
    x = str(s).strip().lower()
    if any(k in x for k in ("optimum", "optimal", "no drought", "control", "well")) or x in {"m1", "opt", "nd", "ww"}: return DL_OPT
    if "moderate" in x or x == "m2": return DL_INT
    if "severe" in x or x == "m3": return DL_CONT
    return str(s).strip() if str(s).strip() in {DL_OPT, DL_INT, DL_CONT} else None

def _se_array(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else 0.0 if x.size == 1 else float(np.nanstd(x, ddof=1) / np.sqrt(x.size))

def _factorial_b(rep, col, soil, dA, dB, lA, lB):
    """Y ~ C(Group) * C(Genotype) on replicate means of one soil; (result row, fit bundle or None)."""
    sub, tmp = _pair_data(rep, col, soil, dA, dB, lA, lB)
    res = {"N": int(sub.shape[0]), "Transform": "", "Assump_OK": "", "p_norm": np.nan, "p_hetero": np.nan, "Method": "",
           "P_Group": np.nan, "P_Genotype": np.nan, "P_Group×Geno": np.nan}
    if sub.empty or sub["Genotype"].nunique() < 2 or sub["Treat"].nunique() < 2 or sub[col].count() < 6:
        return res, None
    try:
        fit, tname, diag, meta, work = _fit_choose_transform(tmp, "Y ~ C(Group) * C(Genotype)")
    except Exception:
        return res, None
    p = {t: _anova_p_for_term(fit, t, 2, True, diag)[0] for t in ("C(Group)", "C(Genotype)", "C(Group):C(Genotype)")}
    if not np.isfinite(p["C(Group):C(Genotype)"]):
        p["C(Group):C(Genotype)"] = _anova_p_for_term(fit, "C(Genotype):C(Group)", 2, True, diag)[0]
    fin = lambda v: float(v) if np.isfinite(v) else np.nan
    res.update({"N": int(work.shape[0]), "Transform": tname, "Assump_OK": "yes" if diag["assumptions_met"] else "no",
                "p_norm": diag["p_norm"], "p_hetero": diag["p_hetero"], "Method": "standard_anova" if diag["assumptions_met"] else "robust_hc3",
                "P_Group": fin(p["C(Group)"]), "P_Genotype": fin(p["C(Genotype)"]), "P_Group×Geno": fin(p["C(Group):C(Genotype)"])})
    return res, {"fit": fit, "diag": diag, "work": work, "labelA": lA, "labelB": lB}

def _within_genotype_pvals_b(bundle, genos):
    """{genotype: p of Group A vs B within genotype} from the fitted factorial model (HC3 if assumptions fail)."""
    fit, work = bundle["fit"], bundle["work"]
    present = set(sorted(pd.unique(work["Genotype"].astype(str))))
    rob = None
    if not bundle["diag"].get("assumptions_met", False):
        try: rob = fit.get_robustcov_results(cov_type="HC3")
        except Exception: rob = None
    out = {}
    for g in [g for g in genos if str(g) in present]:
        try:
            p = _contrast_p(rob if rob else fit, fit.model.data.design_info, {"Group": bundle["labelA"], "Genotype": str(g)},
                            {"Group": bundle["labelB"], "Genotype": str(g)}, dmatrix=False)
            out[str(g)] = p if np.isfinite(p) else np.nan
        except Exception:
            out[str(g)] = np.nan
    return out

def _cld_b(rep, col, soil, dA, dB, lA, lB):
    """CLD letters of genotypes within level dA (contrasts of Y ~ C(Group) * C(Genotype), HC3 if assumptions fail, Holm)."""
    sub, tmp = _pair_data(rep, col, soil, dA, dB, lA, lB)
    if sub.empty or sub["Genotype"].nunique() < 2 or sub["Treat"].nunique() < 2:
        return {}
    try:
        fit, tname, diag, meta, work = _fit_choose_transform(tmp, "Y ~ C(Group) * C(Genotype)")
    except Exception:
        return {}
    groups = sorted(work["Genotype"].unique().tolist())
    if len(groups) < 2:
        return {}
    raw, pairs = [], []
    for i, j in combinations(range(len(groups)), 2):
        gi, gj = groups[i], groups[j]
        try:
            tf = fit.get_robustcov_results(cov_type="HC3") if not diag.get("assumptions_met", False) else fit
            p = _contrast_p(tf, fit.model.data.design_info, {"Group": lA, "Genotype": gi}, {"Group": lA, "Genotype": gj}, dmatrix=False)
            p = p if np.isfinite(p) else np.nan
        except Exception:
            try:
                pw = work[(work["Group"] == lA) & (work["Genotype"].isin([gi, gj]))].copy()
                if pw["Genotype"].nunique() < 2 or pw.shape[0] < 4: p = 1.0
                else:
                    pf = smf.ols("Y ~ C(Genotype)", data=pw).fit()
                    p_try, _ = _anova_p_for_term(pf, "C(Genotype)", 2, True, _residual_diagnostics(pf))
                    p = float(p_try) if np.isfinite(p_try) else 1.0
            except Exception: p = 1.0
        raw.append(float(p) if np.isfinite(p) else 1.0)
        pairs.append((i, j))
    return _cld_letters_from_pairwise(groups, dict(zip(pairs, _holm_adjust(raw))))

def _overall_blocked_anova_rows(rep, soil, col, label):
    """Equal-weight genotype means, Y ~ C(Genotype) + C(Treat2) for Int vs Opt, Low vs Opt, Low vs Int."""
    rows, sub = [], rep[rep["Soil"] == soil].copy()
    if sub.empty: return rows
    piv = sub.groupby(["Genotype", "Treat"], as_index=False)[col].mean().pivot(index="Genotype", columns="Treat", values=col)
    base = {"Parameter": label, "Soil": soil}
    empty = {"Transf": "", "p(norm)": np.nan, "p(BP)": np.nan, "Assump": "No", "p": np.nan, "η²p": np.nan, "Transf details": ""}
    for A, B, comp in ((DL_INT, DL_OPT, "Int vs Opt"), (DL_CONT, DL_OPT, "Low vs Opt"), (DL_CONT, DL_INT, "Low vs Int")):
        if A not in piv.columns or B not in piv.columns:
            rows.append({**base, "Comp": comp, "nG": 0, "Mean A": np.nan, "Mean B": np.nan, "Diff": np.nan, **empty,
                         "Infer": "missing_levels", "Note": "missing drought level(s)"}); continue
        both = piv[[A, B]].dropna(); nG = int(both.shape[0])
        mA = float(np.nanmean(both[A].to_numpy())) if nG else np.nan
        mB = float(np.nanmean(both[B].to_numpy())) if nG else np.nan
        row = {**base, "Comp": comp, "nG": nG, "Mean A": mA, "Mean B": mB, "Diff": fin_diff(mA, mB)}
        if nG < 2:
            rows.append({**row, **empty, "Infer": "too_few_genotypes", "Note": "need ≥2 genotypes"}); continue
        tmp = pd.DataFrame({"Genotype": np.repeat(both.index.astype(str).to_numpy(), 2), "Treat2": np.tile(["A", "B"], nG),
                            "Value": np.concatenate([both[A].to_numpy(float), both[B].to_numpy(float)])})
        try:
            fit, tname, diag, meta, _ = _fit_choose_transform(tmp, "Y ~ C(Genotype) + C(Treat2)")
            pval, method = _anova_p_for_term(fit, "C(Treat2)", 2, True, diag)
            rows.append({**row, "Transf": tname, "Infer": method, "p(norm)": diag.get("p_norm", np.nan), "p(BP)": diag.get("p_hetero", np.nan),
                         "Assump": "Yes" if diag.get("assumptions_met", False) else "No", "p": float(pval) if np.isfinite(pval) else np.nan,
                         "η²p": _effect_size_from_aov(fit, "C(Treat2)"), "Note": "assumptions failed; HC3 robust p-values" if method == "robust_hc3" else "",
                         "Transf details": meta.get("details", "")})
        except Exception as e:
            rows.append({**row, **empty, "Infer": "fit_failed", "Note": f"fit_failed:{type(e).__name__}"})
    return rows

def plot_forest_6subplots_cam2(df_code2, outdir=OUT_BELOW, metrics=B_METRICS):
    FS_ALL, FS_AXIS, FS_PANEL, FS_LEGEND, FIG_SCALE = PUB_FS_TICK, PUB_FS_AXIS, PUB_FS_PANEL, PUB_FS_LEGEND, 1.30 * PUB_FIG_GROWTH
    sub = df_code2[df_code2['Cam'] == 'Cam2'].copy()
    sub = sub[sub['Soil'].isin([SOIL_SILT, SOIL_SAND])].copy()
    sub['DroughtLevel'] = sub['DroughtLevel'].map(_canon_drought)
    sub = sub[sub['DroughtLevel'].isin([DL_OPT, DL_CONT, DL_INT]) & sub['Genotype'].isin(GENO_ORDER)].copy()
    genos = [g for g in GENO_ORDER if g in sub['Genotype'].unique()]
    panel_of = {n: i for i, c in enumerate(metrics) for n in (c, B_LABEL.get(c, c), B_ANOVA_LABEL.get(c, c))}
    panel_for = lambda par, soil: (f"({chr(97 + 2 * panel_of[str(par).strip()] + (0 if str(soil).strip() == 'Silt Loam' else 1))})"
                                   if str(par).strip() in panel_of else "")
    fac_rows, fold_rows, s4_rows, info = [], [], [], {}
    for col in metrics:
        rep = sub.dropna(subset=[col, 'Soil', 'Genotype', 'DroughtLevel', 'Rep']).copy()        # replicate means
        rep[col] = pd.to_numeric(rep[col], errors='coerce'); rep = rep.dropna(subset=[col])
        rep = rep.groupby(['Soil', 'Genotype', 'DroughtLevel', 'Rep'], as_index=False).agg(**{col: (col, 'mean')}).rename(columns={'DroughtLevel': 'Treat'})
        info[col] = {}
        for soil in [SOIL_SILT, SOIL_SAND]:
            bundles = {}
            for comp, dA, dB, lA, lB in B_COMPS:
                res, bundle = _factorial_b(rep, col, soil, dA, dB, lA, lB)
                fac_rows.append({"Parameter": B_ANOVA_LABEL[col], "Soil": soil, "Comp": comp, **res,
                                 "Sig_effects": "; ".join(n for n, p in (("Group", res["P_Group"]), ("Genotype", res["P_Genotype"]),
                                                                         ("Group×Genotype", res["P_Group×Geno"])) if np.isfinite(p) and p < 0.05) or "none"})
                if bundle is not None: bundles[comp] = bundle
            stars = {c: _within_genotype_pvals_b(bundles[c], genos) if c in bundles else {str(g): np.nan for g in genos} for c in ("Int vs Opt", "Low vs Opt")}
            cld = {DL_INT: _cld_b(rep, col, soil, DL_INT, DL_OPT, "IntConst", "OptConst"),
                   DL_CONT: _cld_b(rep, col, soil, DL_CONT, DL_OPT, "LowConst", "OptConst")}
            s = rep[rep['Soil'] == soil]
            genes, pts = [], {DL_INT: ([], []), DL_CONT: ([], [])}
            for g in genos:                                   # ratio of each replicate to the genotype's OptConst mean
                vals = {d: pd.to_numeric(s[(s['Genotype'] == g) & (s['Treat'] == d)][col], errors='coerce').dropna().to_numpy(float)
                        for d in (DL_OPT, DL_INT, DL_CONT)}
                m = {d: float(np.nanmean(v)) if v.size else np.nan for d, v in vals.items()}
                fold_rows.append({"Parameter": B_LABEL[col], "Soil": soil, "Genotype": g, "Mean OptE": m[DL_OPT], "Mean IntE": m[DL_INT],
                                  "Mean LowE": m[DL_CONT], "Int/Opt": safe_ratio(m[DL_INT], m[DL_OPT]), "Low/Opt": safe_ratio(m[DL_CONT], m[DL_OPT]),
                                  "Δ(Int−Opt)": fin_diff(m[DL_INT], m[DL_OPT]), "Δ(Low−Opt)": fin_diff(m[DL_CONT], m[DL_OPT])})
                if s[s['Genotype'] == g].empty: continue
                genes.append(g)
                for d in (DL_INT, DL_CONT):
                    r = vals[d] / m[DL_OPT] if (np.isfinite(m[DL_OPT]) and m[DL_OPT] != 0 and vals[d].size) else np.array([], float)
                    pts[d][0].append(float(np.nanmean(r)) if r.size else np.nan)
                    pts[d][1].append(_se_array(r) if r.size else np.nan)
            piv = s.groupby(["Genotype", "Treat"])[col].mean().reset_index().pivot(index="Genotype", columns="Treat", values=col)
            if not piv.empty:
                o = {d: float(np.nanmean(piv[d].to_numpy())) if d in piv.columns else np.nan for d in (DL_OPT, DL_INT, DL_CONT)}
                fold_rows.append({"Parameter": B_LABEL[col], "Soil": soil, "Genotype": "All genotypes (mean of genotype means)",
                                  "Mean OptE": o[DL_OPT], "Mean IntE": o[DL_INT], "Mean LowE": o[DL_CONT],
                                  "Int/Opt": safe_ratio(o[DL_INT], o[DL_OPT]), "Low/Opt": safe_ratio(o[DL_CONT], o[DL_OPT]),
                                  "Δ(Int−Opt)": fin_diff(o[DL_INT], o[DL_OPT]), "Δ(Low−Opt)": fin_diff(o[DL_CONT], o[DL_OPT])})
            info[col][soil] = dict(genes=genes, r={d: np.asarray(v[0], float) for d, v in pts.items()},
                                   se={d: np.asarray(v[1], float) for d, v in pts.items()}, stars=stars, cld=cld)
            s4_rows.extend(_overall_blocked_anova_rows(rep, soil, col, B_ANOVA_LABEL[col]))

    # ---- forest plot (one row per trait, one column per soil) ----
    rc = {"font.size": FS_ALL, "axes.titlesize": FS_AXIS, "axes.labelsize": FS_AXIS, "xtick.labelsize": FS_ALL,
          "ytick.labelsize": FS_AXIS, "legend.fontsize": FS_LEGEND, "figure.titlesize": FS_AXIS}
    ekw = dict(capsize=6, linestyle='none', ecolor='black', elinewidth=1.6, capthick=1.6, zorder=3)
    with plt.rc_context(rc):
        plt.close('all')
        n_rows = len(metrics)
        fig, axes = plt.subplots(n_rows, 2, figsize=(20.0 * FIG_SCALE * 1.10, (3.2 * n_rows + 1.6) * FIG_SCALE * PUB_FIG_PAD), constrained_layout=False)
        for r, col in enumerate(metrics):
            vals = [v for soil in (SOIL_SILT, SOIL_SAND) for d in (DL_CONT, DL_INT) if info[col][soil]["r"][d].size
                    for v in (np.abs(info[col][soil]["r"][d] - 1.0) + np.nan_to_num(info[col][soil]["se"][d], nan=0.0)).tolist() if np.isfinite(v)]
            x_max = max(max(vals) if vals else 0.5, 1e-6)
            xlim = (1.0 - 1.1 * x_max, 1.0 + 1.1 * x_max)          # symmetric about 1
            margin = 0.02 * (xlim[1] - xlim[0])
            for c, soil in enumerate((SOIL_SILT, SOIL_SAND)):
                ax, it, label = axes[r, c], info[col][soil], f"({chr(97 + r * 2 + c)})"
                genes = it["genes"]
                ax.axvline(1.0, color='black', linewidth=1.2, zorder=0)
                if not genes:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS)
                    ax.set_yticks([]); ax.set_xlim(xlim); ax.grid(axis='x', linestyle=':', alpha=0.5)
                    ax.tick_params(axis="x", labelsize=FS_ALL, width=1.2, length=5)
                    _inset_title(ax, label, f"{B_LABEL[col]} : {soil}", FS_PANEL, FS_AXIS)
                    continue
                y = np.arange(len(genes))
                sc = SOIL_COLORS.get(soil, '#888888')
                ax.errorbar(it["r"][DL_INT], y + 0.15, xerr=it["se"][DL_INT], fmt='^', markersize=9, color=sc, label=r'Int$_{Const}$/Opt$_{Const}$', **ekw)
                ax.errorbar(it["r"][DL_CONT], y - 0.15, xerr=it["se"][DL_CONT], fmt='s', markersize=10, color=sc, label=r'Low$_{Const}$/Opt$_{Const}$', **ekw)
                for i, g in enumerate(genes):
                    for d, dy, comp, case in ((DL_INT, 0.15, "Int vs Opt", str.upper), (DL_CONT, -0.15, "Low vs Opt", str.lower)):
                        rr, ss = it["r"][d], it["se"][d]
                        rv = rr[i] if i < len(rr) else np.nan
                        sv = ss[i] if (i < len(ss) and np.isfinite(ss[i])) else 0.0
                        st = _stars(it["stars"][comp].get(str(g), np.nan))
                        if st and np.isfinite(rv):
                            ax.text(rv, y[i] + dy + 0.22, st, ha='center', va='bottom', fontsize=FS_AXIS)
                        ltr = case(it["cld"][d].get(g, "") or "")
                        if ltr and np.isfinite(rv):
                            ax.text((rv + sv + margin) if rv >= 1.0 else (rv - sv - margin), y[i] + dy, ltr, ha='left' if rv >= 1.0 else 'right',
                                    va='center', fontsize=FS_AXIS)
                ax.set_xlim(xlim); ax.grid(axis='x', linestyle=':', alpha=0.5)
                ax.set_ylim(-0.85, (len(genes) - 1) + 0.15 + 0.22 + 0.35 + 1.45)
                ax.set_yticks(y)
                ax.set_yticklabels(genes if c == 0 else [], fontsize=FS_AXIS)
                ax.tick_params(axis="x", labelsize=FS_ALL, width=1.2, length=5)
                ax.tick_params(axis="y", labelsize=FS_AXIS, width=1.2, length=5)
                if r == n_rows - 1:
                    ax.set_xlabel(r"Relative change from Opt$_{Const}$", fontsize=FS_AXIS)
                _inset_title(ax, label, f"{B_LABEL[col]} : {soil}", FS_PANEL, FS_AXIS)
                if r == 0 and c == 0:
                    ax.legend(handles=[Line2D([0], [0], marker='^', linestyle='none', markersize=9, color='black', label=r'Int$_{Const}$/Opt$_{Const}$'),
                                       Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='black', label=r'Low$_{Const}$/Opt$_{Const}$')],
                              loc='lower right', fontsize=FS_LEGEND, frameon=True, framealpha=0.9)
                if r == 0 and c == 1:
                    ax.legend(handles=[Patch(facecolor=SOIL_COLORS[s], edgecolor='black', linewidth=0.8, label=s) for s in (SOIL_SILT, SOIL_SAND)],
                              loc='upper right', fontsize=FS_LEGEND, frameon=True, framealpha=0.9, handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
        fig.suptitle("Forest plots (Cam=Cam2): IntConst/OptConst & LowConst/OptConst", fontsize=FS_AXIS, y=0.998)
        plt.tight_layout(rect=[0.02, 0.03, 0.995, 0.980], h_pad=2.6, w_pad=2.2)
        fig.canvas.draw()
        tag = "_".join(metrics)
        fig.savefig(os.path.join(outdir, f"Cam2_ForestPlots_{tag}_ANOVA.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

    # ---- Tables B1-B4 ----
    def fmt_cols(d, spec):
        for c, nd in spec:
            d[c] = pd.to_numeric(d[c], errors="coerce").apply(lambda x: _fmt_f(x, nd) if np.isfinite(x) else "")
    add_panel = lambda d: d.apply(lambda r: panel_for(r["Parameter"], r["Soil"]), axis=1)
    path = lambda n: os.path.join(outdir, f"Table_{n}_{tag}.png")
    d = pd.DataFrame(fac_rows)
    for c in ["p_norm", "p_hetero", "P_Group", "P_Genotype", "P_Group×Geno"]:
        d[c] = d[c].apply(_fmt_p)
    d["N"] = d["N"].apply(_fmt_i)
    d["Panel"] = add_panel(d)
    d = d[["Panel", "Parameter", "Soil", "Comp", "N", "Transform", "Method", "p_norm", "p_hetero", "Assump_OK", "P_Group", "P_Genotype",
           "P_Group×Geno", "Sig_effects"]].rename(columns={"Transform": "Transf", "Method": "Infer", "p_norm": "p(norm)", "p_hetero": "p(BP)",
                                                         "Assump_OK": "Assump", "P_Group": "p(Treatment)", "P_Genotype": "p(Geno)",
                                                         "P_Group×Geno": "p(Int)", "Sig_effects": "Sig effects"})
    save_df_as_table_png(d, path("B1_FactorialANOVA"), title="Table B1. Two-way factorial ANOVA within soil (Int vs Opt; Low vs Opt; Low vs Int)",
                         footnote=("Per Soil × Parameter × Comparison: Y ~ C(Group)*C(Genotype) on replicate means. Shapiro/JB p≥0.05; BP p≥0.05. "
                                   "If assumptions failed: HC3 robust Wald p-values. Asterisks in plot = within-genotype Group contrasts. "
                                   "CLD letters = pairwise genotype contrasts (UPPERCASE=IntConst, lowercase=LowConst)."),
                         id_cols=["Panel", "Parameter", "Soil", "Comp"])
    fold_df = pd.DataFrame(fold_rows)
    d2 = fold_df.copy()
    fmt_cols(d2, [("Mean OptE", 4), ("Mean IntE", 4), ("Mean LowE", 4), ("Int/Opt", 3), ("Low/Opt", 3), ("Δ(Int−Opt)", 4), ("Δ(Low−Opt)", 4)])
    d2["_k"] = d2["Genotype"].apply(lambda g: (0 if "All genotypes" in str(g) else 1, str(g)))
    d2 = d2.sort_values(["Parameter", "Soil", "_k", "Genotype"]).drop(columns=["_k"]).reset_index(drop=True)
    d2.insert(0, "Panel", add_panel(d2))
    save_df_as_table_png(d2, path("B2_FoldChanges_vs_OptConst"), title="Table B2. Means, ratios (fold-change), and deltas vs OptConst within soil",
                         footnote="Means from replicate means (Rep-level). Overall row: mean of genotype means (equal weight). Ratio < 1 = reduction vs OptConst.",
                         id_cols=["Panel", "Parameter", "Soil", "Genotype"])
    base = fold_df[~fold_df["Genotype"].astype(str).str.contains("All genotypes", na=False)].copy()
    for c in ("Mean IntE", "Mean LowE"):
        base[c] = pd.to_numeric(base[c], errors="coerce")
    grp = base.groupby(["Parameter", "Soil"], as_index=False).agg(
        AllMean_M2=("Mean IntE", "mean"), AllMean_M3=("Mean LowE", "mean"),
        nG_M2=("Mean IntE", lambda x: int(pd.Series(x).dropna().size)), nG_M3=("Mean LowE", lambda x: int(pd.Series(x).dropna().size)))
    s3 = base.merge(grp, on=["Parameter", "Soil"], how="left")
    for cm, ca, cr, cp in [("Mean IntE", "AllMean_M2", "Ratio Int", "%Δ Int"), ("Mean LowE", "AllMean_M3", "Ratio Low", "%Δ Low")]:
        s3[cr] = np.where((s3[ca] != 0) & s3[ca].apply(np.isfinite), s3[cm] / s3[ca], np.nan)
        s3[cp] = (s3[cr] - 1.0) * 100.0
    direction = lambda r: ("Increase" if r > 1 else ("Decrease" if r < 1 else "No change")) if np.isfinite(r) else ""
    s3["Dir Int"], s3["Dir Low"] = s3["Ratio Int"].apply(direction), s3["Ratio Low"].apply(direction)
    fmt_cols(s3, [("Mean IntE", 4), ("AllMean_M2", 4), ("Ratio Int", 3), ("%Δ Int", 1), ("Mean LowE", 4), ("AllMean_M3", 4), ("Ratio Low", 3), ("%Δ Low", 1)])
    s3["Panel"] = add_panel(s3)
    s3 = (s3[["Panel", "Parameter", "Soil", "Genotype", "Mean IntE", "AllMean_M2", "Ratio Int", "%Δ Int", "Dir Int", "Mean LowE", "AllMean_M3",
              "Ratio Low", "%Δ Low", "Dir Low", "nG_M2", "nG_M3"]]
          .rename(columns={"AllMean_M2": "AllMean IntE", "AllMean_M3": "AllMean LowE", "nG_M2": "nG Int", "nG_M3": "nG Low"})
          .sort_values(["Parameter", "Soil", "Genotype"]).reset_index(drop=True))
    save_df_as_table_png(s3, path("B3_Genotype_vs_AllMean_IntLow"),
                         title="Table B3. Genotype deviation vs mean of genotype means (within soil), IntConst and LowConst",
                         footnote="AllMean = mean of genotype means (equal-weight). Ratio = Genotype/AllMean; %Δ = (Ratio−1)×100. Int=IntConst, Low=LowConst.",
                         id_cols=["Panel", "Parameter", "Soil", "Genotype"])
    d4 = pd.DataFrame(s4_rows)
    d4["nG"] = d4["nG"].apply(_fmt_i)
    for c in ("p(norm)", "p(BP)", "p"):
        d4[c] = d4[c].apply(_fmt_p)
    d4["η²p"] = d4["η²p"].apply(lambda x: _fmt_f(x, 3) if np.isfinite(x) else "")
    fmt_cols(d4, [("Mean A", 4), ("Mean B", 4), ("Diff", 4)])
    d4["Panel"] = add_panel(d4)
    d4 = d4[["Panel", "Parameter", "Soil", "Comp", "nG", "Mean A", "Mean B", "Diff", "Transf", "Infer", "p(norm)", "p(BP)", "Assump", "p", "η²p"]]
    save_df_as_table_png(d4, path("B4_OverallBlockedANOVA_IntLow"), title="Table B4. Overall drought contrasts (equal-weight genotype means), per soil",
                         footnote=("Within each Soil × Parameter, genotype means computed per drought level (equal weight). Blocked ANOVA: "
                                   "Y ~ C(Genotype) + C(Treat2), testing C(Treat2). Comp: Int vs Opt = IntConst vs OptConst; Low vs Opt = LowConst vs OptConst."),
                         id_cols=["Panel", "Parameter", "Soil", "Comp"])

plot_forest_6subplots_cam2(df_code2)


# ================================ [SOIL-ENGINE] soil effect: two soil x treatment groups, Cam2 ================================
# 9-panel bar figure per genotype (physiology + root traits), collapsed group-mean figure, Tables C1 (ANOVA) and C2 (fold change)
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
SOIL_GENO_ORDER = ("Alpenfex", "Braues.S", "Pfarrkirchner", "Benedictio KWS", "Luigi CS", "P8888")
SOIL_WSPACE, SOIL_HSPACE_MIN = 0.40, 0.08
_BASE = dict(tick_width=1.1, tick_length=5.0, spine_width=1.1, err_elinewidth=1.0, err_capthick=1.0, err_capsize=4.0,
             legend_edge_lw=1.0, legend_framealpha=0.9, legend_borderaxespad=0.2, legend_handlelength=1.4,
             legend_handletextpad=0.4, legend_labelspacing=0.25)
_NOSCALE = ("legend_framealpha", "legend_borderaxespad", "legend_handlelength", "legend_handletextpad", "legend_labelspacing")
_MISS = {"", "nan", "na", "n/a", "none", "null", "<na>", "nat"}
_BGENO = {"unknown", "unk", "unidentified", "not identified", "not recorded", "missing", "na", "n/a"}

def _fs(scale):
    """Line widths / tick sizes scaled with the figure."""
    return {k: (v * float(scale) if k not in _NOSCALE else v) for k, v in _BASE.items()}

def _rc(fs, style):
    FS = int(fs)
    return {"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
            "legend.fontsize": FS, "figure.titlesize": FS, "axes.linewidth": style["spine_width"],
            "xtick.major.width": style["tick_width"], "ytick.major.width": style["tick_width"],
            "xtick.major.size": style["tick_length"], "ytick.major.size": style["tick_length"]}

def _sname(s): return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s)).strip('_')

def _ctxt(s):
    """Clean text column: non-breaking spaces, repeated spaces, 'nan'/'none'... -> NA."""
    s = s.astype("string").str.replace(" ", " ", regex=False).str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.str.lower().isin(_MISS), pd.NA)

def _cgeno(x):
    if pd.isna(x): return pd.NA
    t = str(x).strip(); l = t.lower()
    return "Braues.S" if ("braunes" in l and any(k in l for k in ("schindel", "schiendel"))) else t

def _cgens(s):
    s = _ctxt(s); s = s.mask(s.str.lower().isin(_BGENO), pd.NA); return s.map(_cgeno)

def _canon_treat(s):
    if pd.isna(s): return None
    x = str(s).strip().lower()
    return "Moderate Drought" if ("moderate" in x or x == "m2") else "Severe Drought" if ("severe" in x or x == "m3") else None

def _eta_aov(fit):
    """Partial eta squared of Group, Genotype and their interaction."""
    try:
        aov = anova_lm(fit, typ=2)
        ss_r = float(aov.loc["Residual", "sum_sq"]) if "Residual" in aov.index else np.nan
        e = lambda term: _eta2p(float(aov.loc[term, "sum_sq"]), ss_r) if term in aov.index else np.nan
        ei = next((e(c) for c in ["C(Group):C(Genotype)", "C(Genotype):C(Group)"] if c in aov.index), np.nan)
        return e("C(Group)"), e("C(Genotype)"), ei
    except: return np.nan, np.nan, np.nan

def _sig_label(pg, pge, pi, a=0.05):
    parts = [n for n, p in [("Group", pg), ("Genotype", pge), ("Group*Genotype", pi)] if np.isfinite(p) and p < a]
    return "*".join(parts) if parts else "None"

def _t_test_p_from_fit(fit, diag, contrast_vec, rb=True):
    """Contrast p (HC3 if assumptions failed)."""
    try:
        if rb and not bool(diag.get("assumptions_met", False)):
            try: return float(fit.get_robustcov_results(cov_type="HC3").t_test(contrast_vec).pvalue)
            except Exception: pass
        return float(fit.t_test(contrast_vec).pvalue)
    except Exception:
        return np.nan

def _dm_row(fit, group_label, geno_label, col="Group"):
    row = pd.DataFrame({col: [str(group_label)], "Genotype": [str(geno_label)]})
    return np.asarray(patsy.build_design_matrices([fit.model.data.design_info], row)[0], float).ravel()

def _cld_from_factorial_fit(fit, diag, workY, group_label, genos, col="Group"):
    """CLD letters of genotypes within one group (Holm)."""
    sub = workY[workY[col].astype(str) == str(group_label)]
    present = [g for g in genos if (sub["Genotype"].astype(str) == str(g)).any()]
    if len(present) < 2:
        return {str(g): "a" for g in present}
    raw_ps, pairs = [], []
    for i, j in combinations(range(len(present)), 2):
        try:
            p = _t_test_p_from_fit(fit, diag, _dm_row(fit, group_label, str(present[i]), col) - _dm_row(fit, group_label, str(present[j]), col))
            p = p if np.isfinite(p) else 1.0
        except Exception:
            p = 1.0
        raw_ps.append(float(p)); pairs.append((i, j))
    return _cld_pw([str(g) for g in present], dict(zip(pairs, _holm(raw_ps))))

def _p_by_genotype_from_fit(fit, diag, workY, genos, groupA, groupB, col="Group", min_n=2):
    """p of group B vs A within each genotype (NaN if < min_n replicates in a cell)."""
    out = {str(g): np.nan for g in genos}
    for g in map(str, genos):
        n = [int(((workY["Genotype"].astype(str) == g) & (workY[col].astype(str) == str(G))).sum()) for G in (groupA, groupB)]
        if min(n) < min_n:
            continue
        try: out[g] = _t_test_p_from_fit(fit, diag, _dm_row(fit, groupB, g, col) - _dm_row(fit, groupA, g, col))
        except Exception: out[g] = np.nan
    return out

def _stars_by_genotype_from_factorial_fit(fit, diag, workY, genos, groupA, groupB, col="Group"):
    return {g: _stars(p) for g, p in _p_by_genotype_from_fit(fit, diag, workY, genos, groupA, groupB, col).items()}

def _inset_label(ax, label, fontsize):
    if label:
        ax.text(0.02, 0.97, label, transform=ax.transAxes, ha='left', va='top', fontsize=fontsize, fontweight='bold', clip_on=False, zorder=10)

def _axes_grid_centered(fig, n_pan, ncol, hspace, wspace):
    """ncol panels per row, a shorter last row centred (grid of half-columns)."""
    nrow = int(np.ceil(n_pan / float(ncol)))
    gs = GridSpec(nrow, 2 * ncol, figure=fig, hspace=hspace, wspace=2.0 * wspace)
    axes = []
    for r in range(nrow):
        k = int(min(ncol, n_pan - r * ncol))
        off = (2 * ncol - 2 * k) // 2
        axes += [fig.add_subplot(gs[r, off + 2 * c: off + 2 * c + 2]) for c in range(k)]
    return axes

def _relax_spacing(fig, axes, ncol, horizontal, pad_in, cap, max_iter=8):
    """Widen wspace (y-labels vs left neighbour) or hspace (x tick labels vs row below) until nothing overlaps."""
    if not axes:
        return fig
    gs = axes[0].get_subplotspec().get_gridspec()
    fig.canvas.draw()
    for _ in range(int(max_iter)):
        r = fig.canvas.get_renderer()
        size = float(fig.bbox.width if horizontal else fig.bbox.height)
        pad = float(pad_in) * float(fig.dpi) / size
        worst, gap_now = 0.0, None
        for i, ax in enumerate(axes):
            if horizontal:
                if i % ncol == 0:
                    continue
                edge = ax.yaxis.label.get_window_extent(renderer=r).x0
                for t in ax.get_yticklabels():
                    if t.get_text(): edge = min(edge, t.get_window_extent(renderer=r).x0)
                edge /= size
                nb = axes[i - 1].get_position().x1
                worst = max(worst, (nb + pad) - edge)
                g = ax.get_position().x0 - nb
            else:
                row = i // ncol
                below = axes[(row + 1) * ncol:(row + 2) * ncol]
                if not below:
                    continue
                edge = ax.get_position().y0
                for t in ax.get_xticklabels():
                    if t.get_text(): edge = min(edge, t.get_window_extent(renderer=r).y0 / size)
                nb = max(b.get_position().y1 for b in below)
                worst = max(worst, (nb + pad) - edge)
                g = ax.get_position().y0 - nb
            gap_now = g if gap_now is None else min(gap_now, g)
        if worst <= 0 or not gap_now or gap_now <= 0:
            break
        params = gs.get_subplot_params(fig)
        old = float(params.wspace if horizontal else params.hspace)
        new = min(old * (gap_now + worst) / gap_now, float(cap))
        gs.update(**{"wspace" if horizontal else "hspace": new})
        fig.canvas.draw()
        if new >= float(cap):
            break
    return fig

def _bar_hatch(ax, x, h, w, fc, yerr=None, ekw=None, hatch="//", z=2):
    ax.bar(x, h, width=w, facecolor=fc, edgecolor="none", linewidth=0, yerr=yerr, error_kw=ekw, zorder=z)
    ax.bar(x, h, width=w, facecolor="none", edgecolor="white", linewidth=0, hatch=hatch, zorder=z + 0.1)

def _pvec(df, vcol, grp, genos, field):
    """Mean or SE per genotype of one group (NaN where missing)."""
    g = df.groupby(["Genotype", "Group"])[vcol].agg(["mean", "count", "std"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    out = []
    for gg in genos:
        r = g[(g["Genotype"] == gg) & (g["Group"] == grp)]
        out.append(float(r[field].iloc[0]) if (not r.empty and np.isfinite(r[field].iloc[0])) else np.nan)
    return np.array(out, float)

def _save_table(df, path, title, footnote=None, id_cols=None):
    """Table picture in the C-table style."""
    save_df_as_table_png(df, path, title, footnote, id_cols=id_cols, wide={"Inference": 22, "Notes": 28, "Transform_details": 28},
                         none_vals=(["none"], ["None"], [""]))

def _group_rows(df_any, vcol, labs):
    """Genotype / Group / Rep / value rows of the two groups (numeric, complete)."""
    tmp = df_any[["Genotype", "Group", "Rep", vcol]].copy()
    tmp[vcol] = pd.to_numeric(tmp[vcol], errors="coerce")
    tmp = tmp.dropna(subset=["Genotype", "Group", "Rep", vcol])
    return tmp[tmp["Group"].isin(labs)].copy()

def _preprocess(df_code2, window_rows, soil2, soil3, treat2, treat3, lab2, lab3, traits):
    """Replicate means per Genotype x Group x Rep: root traits + Ksp_soil (Cam2), Psi_x, predawn, transpiration/area (window)."""
    df2 = df_code2.copy()
    df2["Cam"] = _ctxt(df2["Cam"])
    df2 = df2[df2["Cam"] == "Cam2"].copy()
    dc = next(c for c in ("DroughtLevel", "Drought Category", "Drought Treatment") if c in df2.columns)
    rc = next(c for c in ("Rep", "Replicate") if c in df2.columns)
    for c in ["Soil", dc, rc]: df2[c] = _ctxt(df2[c])
    df2["Genotype"] = _cgens(df2["Genotype"])
    df2["Soil"] = df2["Soil"].map(canon_soil); df2["Treat"] = df2[dc].map(_canon_treat); df2["Rep"] = df2[rc]
    group = lambda d: np.where((d["Soil"] == soil2) & (d["Treat"] == treat2), lab2,
                               np.where((d["Soil"] == soil3) & (d["Treat"] == treat3), lab3, pd.NA))
    df2["Group"] = group(df2)
    df2 = df2.dropna(subset=["Group", "Genotype", "Rep"]).copy()
    traits = [c for c in traits if c in df2.columns]
    tr = df2[["Genotype", "Group", "Rep"] + traits].copy()
    for t in traits:
        tr[t] = pd.to_numeric(tr[t], errors="coerce")
        if t.lower() in ("ksp", "ksp_soil", "k_sp", "ksp_value"):
            tr.loc[tr[t] <= 0, t] = np.nan
    tr_rep = tr.dropna(subset=["Genotype", "Group", "Rep"]).groupby(["Genotype", "Group", "Rep"], as_index=False).agg({t: "mean" for t in traits})
    hx = next(c for c in ("hXylem_Ψx", "hXylem_Psi_x", "hXylem_PsiX", "hXylem_Psi", "hXylem_psi_x") if c in df2.columns)
    df2["_Psi"] = pd.to_numeric(df2[hx], errors="coerce")
    psi_rep = df2.groupby(["Genotype", "Group", "Rep"], as_index=False).agg(Psi=("_Psi", "mean"))
    pdc = next((c for c in ("hLeaf_Predawn", "hLeaf_predawn", "hLeafPredawn", "hLeaf_Predawn_Ψ", "hLeaf_Predawn_Psi") if c in df2.columns), None)
    if pdc is None:
        pdawn_rep = pd.DataFrame(columns=["Genotype", "Group", "Rep", "Predawn"])
    else:
        df2["_Predawn"] = pd.to_numeric(df2[pdc], errors="coerce")
        pdawn_rep = df2.groupby(["Genotype", "Group", "Rep"], as_index=False).agg(Predawn=("_Predawn", "mean"))
    wr = window_rows.copy()
    for c in ["Soil", "Drought Category", "Replicate"]: wr[c] = _ctxt(wr[c])
    wr["Genotype"] = _cgens(wr["Genotype"])
    wr["Soil"] = wr["Soil"].map(canon_soil); wr["Treat"] = wr["Drought Category"].map(_canon_treat); wr["Rep"] = wr["Replicate"]
    for src, dst in [("Trans_for_norm", "_Trans"), ("Area_filled", "_Area"), ("NormTrans", "_NormTrans")]:
        wr[dst] = pd.to_numeric(wr[src], errors="coerce")
    wr["Group"] = group(wr)
    wr = wr.dropna(subset=["Group", "Genotype", "Rep"]).copy()
    wr_rep = wr.groupby(["Genotype", "Group", "Rep"], as_index=False).agg(Trans=("_Trans", "mean"), Area=("_Area", "mean"),
                                                                          NormTrans=("_NormTrans", "mean"))
    wr_rep["NormTrans_x1e3"] = pd.to_numeric(wr_rep["NormTrans"], errors="coerce") * 1e3
    univ = sorted(set.intersection(*[set(df["Genotype"].dropna().astype(str)) for df in (tr_rep, wr_rep, psi_rep)]))
    genos = [g for g in SOIL_GENO_ORDER if g in univ] + [g for g in univ if g not in SOIL_GENO_ORDER]
    for df in [tr_rep, wr_rep, psi_rep, pdawn_rep]:
        if df is not None and not df.empty:
            df.drop(df.index[~df["Genotype"].astype(str).isin(genos)], inplace=True)
    return genos, tr_rep, wr_rep, psi_rep, pdawn_rep

TRAIT_TITLE = {"trl": "Total root length (m)", "rdmean": "Root diameter (mm)", "Root_to_Shoot": "Root-to-shoot ratio (–)",
               "srm_upscaled": "Rhizosheath mass (g)"}
YL_TRANS, YL_AREA = r"Transpiration (cm$^3$ h$^{-1}$)", r"Shoot area (cm$^2$)"
YL_NORM = "Normalized transpiration\n" + r"($\times 10^{-3}$ cm$^3$ h$^{-1}$ cm$^{-2}$)"
PT_NORM = "Normalised transpiration\n" + r"($\times 10^{-3}$ cm$^3$ h$^{-1}$ cm$^{-2}$)"
PSI_YL = "Xylem water potential,\n" + r"$h_{\mathrm{x,md}}$ (kPa)"
PSI_PT = r"Xylem water potential, $h_{\mathrm{x,md}}$ (kPa)"

def soil_effect(df_code2, window_rows, out_dir, lab2, lab3, disp2, disp3, leg2, leg3, subfolder, treat2="Moderate Drought",
                treat3="Moderate Drought", psi_ylabel=PSI_YL, psi_paneltitle=PSI_PT, tag="AvgD28_30"):
    """Groups lab2 = Silt Loam + treat2 and lab3 = Sandy Loam + treat3: 9-panel figure + Tables C1, C2, then the collapsed figure."""
    FS, FIG_SCALE = int(PUB_FS_TICK), 1.30 * PUB_FIG_GROWTH
    _FS = _fs(FIG_SCALE); rc_fig = _rc(FS, _FS)
    out_dir = str(Path(os.path.join(out_dir, subfolder)).resolve()); Path(out_dir).mkdir(parents=True, exist_ok=True)
    labs, COLORS = [lab2, lab3], {lab2: "#8B5A2B", lab3: "#C9A97C"}
    HATCHES = {lab2: '..', lab3: 'xx'}
    TABLE_PARAM = {"Trans": "Transpiration", "Area": "Shoot Area", "Psi": "hx,md", "NormTrans_x1e3": "Trans_Norm", "Predawn": "hx,pd",
                   "trl": "trl", "rdmean": "rd", "Root_to_Shoot": "root_to_shoot", "srm_upscaled": "Rhizosheath", "Ksp_soil": "Ksp_soil"}
    genos, tr_rep, wr_rep, psi_rep, pdawn_rep = _preprocess(df_code2, window_rows, "Silt Loam", "Sandy Loam", treat2, treat3, lab2, lab3,
                                                            ("trl", "rdmean", "srm_upscaled", "Ksp_soil"))
    ksp_ttl = "$K_{sp}$\n" + r"(cm$^3$ h$^{-1}$ kPa$^{-1}$)"
    ylbl = {**TRAIT_TITLE, "Ksp_soil": "Soil–plant conductance - $K_{sp}$\n" + r"(cm$^3$ h$^{-1}$ kPa$^{-1}$)"}
    phys = [("Trans", YL_TRANS, YL_TRANS, wr_rep), ("Area", YL_AREA, YL_AREA, wr_rep), ("Psi", psi_paneltitle, psi_ylabel, psi_rep)]
    if pdawn_rep is not None and not pdawn_rep.empty:
        phys.append(("Predawn", r"Predawn water potential, $h_{\mathrm{x,pd}}$ (kPa)", "Predawn water potential,\n" + r"$h_{\mathrm{x,pd}}$ (kPa)", pdawn_rep))
    phys.append(("NormTrans_x1e3", PT_NORM, YL_NORM, wr_rep))
    traits = [(t, {**TRAIT_TITLE, "Ksp_soil": ksp_ttl}[t], ylbl[t], tr_rep) for t in ("trl", "rdmean", "srm_upscaled", "Ksp_soil") if t in tr_rep.columns]
    panels = [(pc, ti, yl, dfp, f"({chr(97 + i)})") for i, (pc, ti, yl, dfp) in enumerate(phys + traits)]
    anova_rows, fold_rows = [], []

    def add_fold_rows(pcode, ptag, df_any):
        tmp = _group_rows(df_any, pcode, labs)
        if tmp.empty: return
        gm = tmp.groupby(["Genotype", "Group"], as_index=False)[pcode].mean().pivot(index="Genotype", columns="Group", values=pcode)
        if pcode == "NormTrans_x1e3": gm = gm / 1e3                       # back to cm3 h-1 cm-2 for the table
        r = lambda a, b: float(b / a) if (np.isfinite(a) and np.isfinite(b) and a > 0) else np.nan
        pc = lambda a, b: float(100 * (b - a) / a) if (np.isfinite(a) and np.isfinite(b) and a != 0) else np.nan
        om = [float(np.nanmean(gm[L].to_numpy())) if L in gm.columns else np.nan for L in labs]
        rows = [("All genotypes", om[0], om[1])] + [(str(gg), *[gm.loc[gg, L] if (gg in gm.index and L in gm.columns) else np.nan for L in labs])
                                                    for gg in genos]
        for i, (g, m2, m3) in enumerate(rows):
            if i:
                m2, m3 = (float(m2) if np.isfinite(m2) else np.nan), (float(m3) if np.isfinite(m3) else np.nan)
            fold_rows.append({"Panel": ptag, "Parameter": TABLE_PARAM.get(pcode, pcode), "Genotype": g, f"Mean_{lab2}": m2, f"Mean_{lab3}": m3,
                              "Ratio_M3_over_M2": r(m2, m3), "Delta_M3_minus_M2": fin_diff(m3, m2), "Percent_change_%": pc(m2, m3)})

    def plot_panel(ax, pcode, title, ylabel, df_any, ptag, show_x):
        use_log, width = pcode == "Ksp_soil", 0.38
        ekw = dict(ecolor="black", elinewidth=_FS["err_elinewidth"], capsize=_FS["err_capsize"], capthick=_FS["err_capthick"])
        mm2, sm2 = _pvec(df_any, pcode, lab2, genos, "mean"), _pvec(df_any, pcode, lab2, genos, "se")
        mm3, sm3 = _pvec(df_any, pcode, lab3, genos, "mean"), _pvec(df_any, pcode, lab3, genos, "se")
        xs = np.arange(len(genos))
        _bar_hatch(ax, xs - width / 2, mm2, width, COLORS[lab2], yerr=sm2, ekw=ekw, hatch=HATCHES[lab2])
        _bar_hatch(ax, xs + width / 2, mm3, width, COLORS[lab3], yerr=sm3, ekw=ekw, hatch=HATCHES[lab3])
        tmp = _group_rows(df_any, pcode, labs).rename(columns={pcode: "Value"})
        n_obs = int(tmp.shape[0]); n_geno = int(tmp["Genotype"].nunique()) if n_obs else 0
        pg = pge = pi = eta_g = eta_ge = eta_i = pn = ph = np.nan
        cht = method = note = ""; assump = False
        cldm2 = cldm3 = {}; stg = {str(g): "" for g in genos}
        if not tmp.empty and tmp["Group"].nunique() >= 2:
            two = tmp["Genotype"].nunique() >= 2
            try:
                f, cht, d, meta, work = _fit_choose(tmp, "Y ~ C(Group) * C(Genotype)" if two else "Y ~ C(Group)")
                pn, ph, assump = d["p_norm"], d["p_hetero"], bool(d["assumptions_met"])
                pg, method = _p4t(f, "C(Group)", 2, True, d)
                if two:
                    pge, _ = _p4t(f, "C(Genotype)", 2, True, d)
                    pi, _ = _p4t(f, "C(Group):C(Genotype)", 2, True, d)
                eta_g, eta_ge, eta_i = _eta_aov(f)
                if not assump: note = "assumptions failed; HC3 robust p"
                workY = work[["Genotype", "Group", "Rep", "Y"]].copy()
                workY["Genotype"], workY["Group"] = workY["Genotype"].astype(str), workY["Group"].astype(str)
                if two:
                    cldm2, cldm3 = (_cld_from_factorial_fit(f, d, workY, L, genos) for L in labs)
                else:
                    cldm2 = cldm3 = {str(genos[0]): "a"} if genos else {}
                stg = _stars_by_genotype_from_factorial_fit(f, d, workY, genos, lab2, lab3)
            except Exception as e:
                note = f"fit_failed:{type(e).__name__}"
        else:
            note = "insufficient data"
        anova_rows.append({"Panel": ptag, "Parameter": TABLE_PARAM.get(pcode, pcode), "n": n_obs, "nG": n_geno, "Transform": cht,
                           "Inference": method, "p_norm": pn, "p_BP": ph, "Assump": assump, "p_Group": pg, "p_Geno": pge, "p_Int": pi,
                           "eta2p_Group": eta_g, "eta2p_Geno": eta_ge, "eta2p_Int": eta_i, "Sig_effects": _sig_label(pg, pge, pi), "Notes": note})
        add_fold_rows(pcode, ptag, df_any)
        t2, t3 = np.nan_to_num(mm2, nan=0) + np.nan_to_num(sm2, nan=0), np.nan_to_num(mm3, nan=0) + np.nan_to_num(sm3, nan=0)
        raw_max = max(np.nanmax(t2) if t2.size else 0, np.nanmax(t3) if t3.size else 0, 1e-9)
        if use_log:                       # log axis from the decade (1/2/5) below the smallest bar
            pos = np.concatenate([mm2[np.isfinite(mm2) & (mm2 > 0)], mm3[np.isfinite(mm3) & (mm3 > 0)]])
            lo = float(np.nanmin(pos)) if pos.size else 1e-3
            dec = 10.0 ** np.floor(np.log10(lo)); man = lo / dec
            ymin = dec * (5.0 if man >= 5 else (2.0 if man >= 2 else 1.0))
            ymax = raw_max * (10.0 ** (0.05 + 0.06 + 0.34 + 0.15))
            ax.set_yscale("log"); ax.set_ylim(ymin, ymax)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=15))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: ("%g" % v)))
        else:
            ymax = raw_max * (1.0 + 0.05 + 0.06 + 0.34)
            ax.set_ylim(0, ymax)
        for i, g in enumerate(map(str, genos)):
            if use_log:
                b2 = max(float(t2[i]), ymin) if np.isfinite(t2[i]) else np.nan
                b3 = max(float(t3[i]), ymin) if np.isfinite(t3[i]) else np.nan
                y2, y3 = b2 * 1.12, b3 * 1.12
                ys = max([v for v in (b2, b3) if np.isfinite(v)] or [ymin]) * 1.55
            else:
                y2, y3, ys = t2[i] + 0.025 * ymax, t3[i] + 0.025 * ymax, max(t2[i], t3[i]) + 0.05 * ymax
            if g in cldm2 and np.isfinite(t2[i]):
                ax.text(xs[i] - width / 2, y2, str(cldm2[g]).upper(), ha="center", va="bottom", fontsize=FS, clip_on=True)
            if g in cldm3 and np.isfinite(t3[i]):
                ax.text(xs[i] + width / 2, y3, str(cldm3[g]).lower(), ha="center", va="bottom", fontsize=FS, clip_on=True)
            if stg.get(g, ""):
                ax.text(xs[i], ys, stg[g], ha="center", va="bottom", fontsize=FS, clip_on=True)
        ax.set_ylabel(ylabel, fontsize=PUB_FS_AXIS)
        ax.set_xticks(xs)
        ax.set_xticklabels(genos if show_x else [""] * len(genos), rotation=18, ha="right", fontsize=PUB_FS_AXIS)
        style_axes(ax, _FS, FS)
        _inset_label(ax, ptag, PUB_FS_PANEL)

    def style_axes(ax, st, fs):
        ax.tick_params(axis="y", width=st["tick_width"], length=st["tick_length"], direction="inout", labelsize=fs)
        ax.tick_params(axis="x", width=st["tick_width"], length=st["tick_length"], direction="inout", labelsize=PUB_FS_AXIS)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(st["spine_width"])

    def legend(ax):
        ax.legend(handles=[Patch(facecolor=COLORS[L], edgecolor="white", hatch=HATCHES[L], lw=_FS["legend_edge_lw"], label=lg)
                           for L, lg in ((lab2, leg2), (lab3, leg3))],
                  loc="upper right", frameon=True, framealpha=_FS["legend_framealpha"], fontsize=PUB_FS_LEGEND,
                  borderaxespad=_FS["legend_borderaxespad"], handlelength=_FS["legend_handlelength"],
                  handletextpad=_FS["legend_handletextpad"], labelspacing=_FS["legend_labelspacing"], borderpad=0.25, columnspacing=0.6)

    def finish(fig, axes, name):
        fig.align_ylabels(axes)
        _relax_spacing(fig, axes, 3, True, 0.10, 1.8)
        _relax_spacing(fig, axes, 3, False, 0.12, 1.2)
        fig.savefig(os.path.join(out_dir, _sname(name)), dpi=300, bbox_inches="tight"); plt.show()

    # ---- figure 1: per genotype ----
    with plt.rc_context(rc_fig):
        plt.close("all")
        n_pan, ncol = len(panels), 3
        nrow = int(np.ceil(n_pan / float(ncol)))
        full = n_pan % ncol == 0
        fig = plt.figure(figsize=(8.4 * ncol * FIG_SCALE, 4.35 * nrow * FIG_SCALE * PUB_FIG_PAD))
        axes = _axes_grid_centered(fig, n_pan, ncol, hspace=0.16 if full else 0.30, wspace=SOIL_WSPACE)
        for i, (ax, (pcode, title, ylabel, df_p, ptag)) in enumerate(zip(axes, panels)):
            plot_panel(ax, pcode, title, ylabel, df_p, ptag, show_x=(i // ncol == nrow - 1) or (not full and i // ncol == nrow - 2))
        legend(axes[0])
        finish(fig, axes, f"Cam2_{lab2}_vs_{lab3}_SoilEffect_8panel_{tag}.png")

    # ---- Table C1 (ANOVA) and C2 (fold change) ----
    d = pd.DataFrame(anova_rows).sort_values("Panel").reset_index(drop=True)
    d["n"], d["nG"] = d["n"].apply(_fi), d["nG"].apply(_fi)
    for c in ["p_norm", "p_BP", "p_Group", "p_Geno", "p_Int"]: d[c] = d[c].apply(_fp)
    d["Assump"] = d["Assump"].apply(lambda v: "Yes" if bool(v) else "No")
    for c in ["eta2p_Group", "eta2p_Geno", "eta2p_Int"]: d[c] = d[c].apply(lambda x: _ff(x, 3))
    d = d[["Panel", "Parameter", "n", "nG", "Transform", "Inference", "p_norm", "p_BP", "Assump", "p_Group", "p_Geno", "p_Int",
           "eta2p_Group", "eta2p_Geno", "eta2p_Int", "Sig_effects"]].rename(columns={
        "n": "N", "Transform": "Transf", "Inference": "Infer", "p_norm": "p(norm)", "p_BP": "p(BP)", "p_Group": "p(Treatment)",
        "p_Geno": "p(Geno)", "p_Int": "p(Int)", "eta2p_Group": "η²p(G)", "eta2p_Geno": "η²p(Geno)", "eta2p_Int": "η²p(Int)", "Sig_effects": "Sig effects"})
    _save_table(d, os.path.join(out_dir, _sname(f"Table_C1_SoilEffect_ANOVA_{lab2}_vs_{lab3}_{tag}.png")),
                f"Table C1. Soil effect ANOVA (auto-transform + HC3 fallback): {disp2} vs {disp3} (Cam2)",
                (f"Group definitions: {disp2}=Silt Loam+{treat2}; {disp3}=Sandy Loam+{treat3}. Model: Y ~ C(Group)*C(Genotype). "
                 "Assumptions: Shapiro/JB p≥0.05, BP p≥0.05. Fallback (robust_fallback=True): HC3 robust p-values. "
                 "CLD letters and genotype-specific asterisks are derived from this same factorial model via linear contrasts (Holm-adjusted for CLD)."),
                id_cols=["Panel", "Parameter"])
    d2 = pd.DataFrame(fold_rows)
    if not d2.empty:
        for c, nd in [(f"Mean_{lab2}", 4), (f"Mean_{lab3}", 4), ("Ratio_M3_over_M2", 3), ("Delta_M3_minus_M2", 4), ("Percent_change_%", 1)]:
            d2[c] = pd.to_numeric(d2[c], errors="coerce").apply(lambda x: _ff(x, nd) if np.isfinite(x) else "")
        d2["_k"] = d2["Genotype"].apply(lambda g: (0 if str(g) == "All genotypes" else 1, str(g)))
        d2 = d2.sort_values(["Panel", "Parameter", "_k", "Genotype"]).drop(columns=["_k"]).reset_index(drop=True)
        d2 = d2[["Panel", "Parameter", "Genotype", f"Mean_{lab2}", f"Mean_{lab3}", "Ratio_M3_over_M2", "Delta_M3_minus_M2", "Percent_change_%"]].rename(
            columns={f"Mean_{lab2}": f"Mean {disp2}", f"Mean_{lab3}": f"Mean {disp3}", "Ratio_M3_over_M2": f"Ratio ({disp3}/{disp2})",
                     "Delta_M3_minus_M2": f"Δ ({disp3}-{disp2})", "Percent_change_%": f"%Δ vs {disp2}"})
        _save_table(d2, os.path.join(out_dir, _sname(f"Table_C2_SoilEffect_FoldChange_{lab2}_vs_{lab3}_{tag}.png")),
                    f"Table C2. Soil effect fold-change and differences: {disp2} vs {disp3} (Cam2)",
                    f"Means from replicate means. Overall = mean of genotype means (equal weight). Ratio shown when baseline>0. Δ = {disp3}−{disp2}.",
                    id_cols=["Panel", "Parameter", "Genotype"])

    # ---- figure 2: collapsed group means (mean +- SE of genotype means), letters a/b + stars from the Group term ----
    cpanels = [(YL_TRANS, YL_TRANS, wr_rep, "Trans"), (YL_AREA, YL_AREA, wr_rep, "Area"), (psi_paneltitle, psi_ylabel, psi_rep, "Psi"),
               (PT_NORM, YL_NORM, wr_rep, "NormTrans_x1e3")] + [(TRAIT_TITLE[t], TRAIT_TITLE[t], tr_rep, t) for t in ("trl", "rdmean", "srm_upscaled")
                                                                  if t in tr_rep.columns]

    def group_mean_se(df_any, vcol):
        tmp = _group_rows(df_any, vcol, labs)
        if tmp.empty: return (np.nan, np.nan), (np.nan, np.nan)
        gm = tmp.groupby(["Genotype", "Group"])[vcol].mean().unstack("Group")

        def ms(g):
            if g not in gm.columns: return np.nan, np.nan
            v = gm[g].to_numpy(float); v = v[np.isfinite(v)]
            return (np.nan, np.nan) if not v.size else (float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)
        return ms(lab2), ms(lab3)

    def group_p(df_any, vcol):
        tmp = _group_rows(df_any, vcol, labs).rename(columns={vcol: "Value"})
        if tmp.empty or tmp["Group"].nunique() < 2: return np.nan
        try:
            f, _, d, _, _ = _fit_choose(tmp, "Y ~ C(Group) * C(Genotype)" if tmp["Genotype"].nunique() >= 2 else "Y ~ C(Group)")
            p, _ = _p4t(f, "C(Group)", 2, True, d)
            return float(p) if np.isfinite(p) else np.nan
        except: return np.nan
    with plt.rc_context(rc_fig):
        plt.close("all")
        n_pan, ncol = len(cpanels), 3
        fig = plt.figure(figsize=(8.4 * ncol * FIG_SCALE, 4.35 * int(np.ceil(n_pan / float(ncol))) * FIG_SCALE * PUB_FIG_PAD))
        axes = _axes_grid_centered(fig, n_pan, ncol, hspace=SOIL_HSPACE_MIN, wspace=SOIL_WSPACE)
        ekw = dict(ecolor="black", elinewidth=_FS["err_elinewidth"], capsize=_FS["err_capsize"], capthick=_FS["err_capthick"])
        for i, (ax, (title, ylabel, df_p, vcol)) in enumerate(zip(axes, cpanels)):
            (m2, se2), (m3, se3) = group_mean_se(df_p, vcol); pG = group_p(df_p, vcol)
            xs = np.array([0, 1], float)
            _bar_hatch(ax, xs[0], m2, 0.55, COLORS[lab2], yerr=se2, ekw=ekw, hatch=HATCHES[lab2])
            _bar_hatch(ax, xs[1], m3, 0.55, COLORS[lab3], yerr=se3, ekw=ekw, hatch=HATCHES[lab3])
            tops = np.array([(m2 if np.isfinite(m2) else 0) + (se2 if np.isfinite(se2) else 0),
                             (m3 if np.isfinite(m3) else 0) + (se3 if np.isfinite(se3) else 0)], float)
            yr = max(float(np.nanmax(tops)), 1e-9) * (1.0 + 0.06 + 0.06 + 0.34)
            ax.set_ylim(0, yr)
            sig = np.isfinite(pG) and pG < 0.05 and np.isfinite(m2) and np.isfinite(m3)
            L2, L3 = ("a", "b") if (sig and m2 >= m3) else (("b", "a") if sig else ("a", "a"))
            ax.text(xs[0], tops[0] + 0.06 * yr, L2, ha="center", va="bottom", fontsize=FS, clip_on=True)
            ax.text(xs[1], tops[1] + 0.06 * yr, L3, ha="center", va="bottom", fontsize=FS, clip_on=True)
            if _stars(pG):
                ax.text(0.5, max(tops[0], tops[1]) + 0.06 * yr + 0.05 * yr, _stars(pG), ha="center", va="bottom", fontsize=FS, clip_on=True)
            ax.set_ylabel(ylabel, fontsize=PUB_FS_AXIS)
            ax.set_xticks(xs)
            ax.set_xticklabels([disp2, disp3], fontsize=PUB_FS_AXIS)
            ax.yaxis.grid(True, linestyle=":", alpha=0.5); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(_FS["spine_width"])
            ax.tick_params(axis="y", width=_FS["tick_width"], length=_FS["tick_length"], direction="inout", labelsize=FS)
            ax.tick_params(axis="x", width=_FS["tick_width"], length=_FS["tick_length"], direction="inout", labelsize=PUB_FS_AXIS)
            _inset_label(ax, f"({chr(97 + i)})", PUB_FS_PANEL)
        legend(axes[0])
        finish(fig, axes, f"Cam2_{lab2}_vs_{lab3}_SoilEffect_8panel_COLLAPSED_{tag}.png")

# [SOIL-INT] IntSilt vs IntSandy, [SOIL-SEV] LowSilt vs LowSandy, [SOIL-MOD] IntSilt vs LowSandy
soil_effect(df_code2, window_rows_28_30, OUT_SOIL, "IntSilt", "IntSandy", r'Int$_{\mathrm{Silt}}$', r'Int$_{\mathrm{Sandy}}$',
            r'Int$_{\mathrm{Silt}}$', r'Int$_{\mathrm{Sandy}}$', "int_vs_int")
soil_effect(df_code2, window_rows_28_30, OUT_SOIL, "LowSilt", "LowSandy", r'Low$_{\mathrm{Silt}}$', r'Low$_{\mathrm{Sandy}}$',
            r'Low$_{\mathrm{Silt}}$', r'Low$_{\mathrm{Sandy}}$', "low_vs_low", treat2="Severe Drought", treat3="Severe Drought",
            psi_ylabel=PSI_PT)
soil_effect(df_code2, window_rows_28_30, OUT_SOIL, "IntSilt", "LowSandy", r'Int$_{\mathrm{Silt}}$', r'Low$_{\mathrm{Sandy}}$',
            r'Int$_{\mathrm{Silt}}$', r'Low$_{\mathrm{Sandy}}$', "", treat3="Severe Drought", psi_paneltitle=PSI_YL)


# ================================ [M5-M3] LowLate vs LowConst: 6-panel figure + supplementary tables ================================
# (a,b) Psi_x and (c,d) shoot area per genotype (CLD letters, stars), (e,f) transpiration vs normalised transpiration
from matplotlib.patches import Ellipse
GENO_COLORS_M5 = {'Alpenfex': '#00441B', 'Braunes Schindelmeiser': '#41AB5D', 'Pfarrkirchner': '#C7E9C0',
                  'P8888': '#3F007D', 'Luigi CS': '#807DBA', 'Benedictio KWS': '#DADAEB'}
DL_LATE, DL_SEV = 'Late Drought', 'Severe Drought'
LAB_M5, LAB_M3 = 'LowLate', 'LowConst'
DISP_M5, DISP_M3 = r'Low$_{\mathrm{Late}}$', r'Low$_{\mathrm{Const}}$'
HATCH, BAR_EDGE_W, SPINE_W, TICK_W, TICK_L, CAPSIZE, ELW = 'xx', 1.2, 1.2, 1.2, 5, 6, 1.6
FS = PUB_FS_TICK
_SOILMAP = {'Silt': SOIL_SILT, 'SiltLoam': SOIL_SILT, 'Silt Loam': SOIL_SILT, 'Silty Loam': SOIL_SILT,
            'Sand': SOIL_SAND, 'Sandy': SOIL_SAND, 'SandyLoam': SOIL_SAND, 'Sandy Loam': SOIL_SAND}
_DRT2MAP = {'Drying Cycle': DL_LATE, 'Late': DL_LATE, 'Late Drought': DL_LATE,          # code2 'DroughtLevel'
            'Severe Drought': DL_SEV, 'Continuous': DL_SEV, 'Continuous Drought': DL_SEV, 'Severe': DL_SEV}
_DRT1MAP = {'Late Drought': DL_LATE, 'Late': DL_LATE,                                     # code1 'Drought Category'
            'Severe': DL_SEV, 'Severe Drought': DL_SEV, 'Continuous Drought': DL_SEV, 'Continuous': DL_SEV}
_csoil = lambda s: _SOILMAP.get(str(s).strip(), str(s).strip()) if not pd.isna(s) else s
_cdrt2 = lambda s: _DRT2MAP.get(str(s).strip(), str(s).strip()) if not pd.isna(s) else s
_cdrt1 = lambda s: _DRT1MAP.get(str(s).strip(), str(s).strip()) if not pd.isna(s) else s

def _cgeno_p(g):
    if pd.isna(g): return g
    x = str(g).strip(); return "P8888" if x.replace(" ", "").upper() in ("P888", "P8888") else x

def _se(s):
    v = pd.to_numeric(pd.Series(np.ravel(s)), errors='coerce').dropna().to_numpy(float)
    return np.nan if v.size <= 1 else float(np.nanstd(v, ddof=1) / np.sqrt(v.size))

def _sig_label_m5(pt, pg, pi, a=0.05):
    parts = [n for n, p in [(f"Treatment ({DISP_M3} vs {DISP_M5})", pt), ("Genotype", pg), ("Treatment×Genotype", pi)] if np.isfinite(p) and p < a]
    return "; ".join(parts) if parts else "none"

def _rep_means(df, scol, gcol, dcol, rcol, vcol, smap, dmap):
    """Replicate means per soil x genotype x drought x replicate."""
    t = df.dropna(subset=[scol, gcol, dcol, rcol, vcol]).copy()
    t[scol] = t[scol].map(smap); t[dcol] = t[dcol].map(dmap)
    t[gcol] = t[gcol].map(_cgeno_p).astype(str).str.strip()
    t[rcol] = t[rcol].astype(str).str.strip()
    t[vcol] = pd.to_numeric(t[vcol], errors='coerce')
    return t.dropna(subset=[vcol]).groupby([scol, gcol, dcol, rcol], as_index=False).agg(**{vcol: (vcol, 'mean')})

def _agg(rep, scol, gcol, dcol, vcol):
    return rep.groupby([scol, gcol, dcol], as_index=False).agg(mean=(vcol, 'mean'), se=(vcol, _se))

def _spines(ax):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(SPINE_W)
    ax.tick_params(direction='inout', length=TICK_L, width=TICK_W)
    ax.set_axisbelow(True)

def _m5_data(rep_df, scol, gcol, dcol, vcol, soil, late_lbl, sev_lbl):
    """Rows of one soil as Treatment (LowLate / LowConst) / Genotype / Value."""
    sub = rep_df[(rep_df[scol] == soil) & (rep_df[dcol].isin([late_lbl, sev_lbl]))].copy()
    sub[vcol] = pd.to_numeric(sub[vcol], errors="coerce")
    sub = sub.dropna(subset=[vcol, gcol, dcol]).copy()
    sub["Treatment"] = sub[dcol].map({late_lbl: LAB_M5, sev_lbl: LAB_M3}).astype(str)
    sub["Genotype"] = sub[gcol].astype(str)
    return sub

def _bar_panel(ax, soil, rep_df, agg_df, vcol, genos, scol, gcol, dcol, late_lbl, sev_lbl, panel_label, ylabel=None,
               show_xlbl=False, yscale=None, late_upper=True):
    """LowConst (filled) and LowLate (open) bars per genotype with CLD letters and treatment stars."""
    width, color, x = 0.38, SOIL_COLORS[soil], np.arange(len(genos))
    ekw = dict(ecolor='black', elinewidth=ELW, capsize=CAPSIZE, capthick=ELW)
    sub = agg_df[agg_df[scol] == soil].set_index([gcol, dcol])
    arr = lambda lv, f: np.array([float(sub.loc[(g, lv), f]) if (g, lv) in sub.index else np.nan for g in genos], float)
    mL, eL, mS, eS = arr(late_lbl, 'mean'), arr(late_lbl, 'se'), arr(sev_lbl, 'mean'), arr(sev_lbl, 'se')
    for i in range(len(genos)):
        ax.bar(x[i] - width / 2, mS[i], width=width, facecolor=color, edgecolor='white', hatch=HATCH, linewidth=BAR_EDGE_W,
               yerr=eS[i] if np.isfinite(eS[i]) else None, error_kw=ekw, zorder=3)
        ax.bar(x[i] + width / 2, mL[i], width=width, facecolor='white', edgecolor=color, hatch=HATCH, linewidth=BAR_EDGE_W,
               yerr=eL[i] if np.isfinite(eL[i]) else None, error_kw=ekw, zorder=3)
    cldL = cldS = {}; strs = {str(g): "" for g in genos}
    rs = rep_df[(rep_df[scol] == soil) & (rep_df[dcol].isin([late_lbl, sev_lbl]))]
    if not rs.empty:
        rs = _m5_data(rep_df, scol, gcol, dcol, vcol, soil, late_lbl, sev_lbl)
        if not rs.empty and rs[dcol].nunique() >= 2:
            tmp = rs[["Treatment", "Genotype", vcol]].rename(columns={vcol: "Value"}).copy()
            two = tmp["Genotype"].nunique() >= 2
            try:
                f, _, d, _, work = _fit_choose(tmp, "Y ~ C(Treatment) * C(Genotype)" if two else "Y ~ C(Treatment)", min_rows=4)
                if two:
                    cldL, cldS = (_cld_from_factorial_fit(f, d, work, L, [str(g) for g in genos], "Treatment") for L in (LAB_M5, LAB_M3))
                else:
                    cldL = cldS = {str(genos[0]): "a"} if genos else {}
                strs = _stars_by_genotype_from_factorial_fit(f, d, work, genos, LAB_M5, LAB_M3, "Treatment")
            except Exception:
                cldL = cldS = {}; strs = {str(g): "" for g in genos}
    tL, tS = np.nan_to_num(mL, nan=0) + np.nan_to_num(eL, nan=0), np.nan_to_num(mS, nan=0) + np.nan_to_num(eS, nan=0)
    raw_max = max(float(np.nanmax(np.r_[tL, tS])) if np.isfinite(np.nanmax(np.r_[tL, tS])) else 1., 1e-6)
    if yscale == "symlog": ax.set_yscale('symlog', linthresh=80, linscale=1.2)
    ax.set_ylim(0., raw_max * (1.0 + 0.025 + 0.05 + 0.38))
    for i, g in enumerate(map(str, genos)):
        if g in cldS and np.isfinite(tS[i]):
            ax.text(x[i] - width / 2, tS[i] + 0.025 * raw_max, str(cldS[g]).upper() if late_upper else str(cldS[g]).lower(),
                    ha='center', va='bottom', fontsize=FS, clip_on=False)
        if g in cldL and np.isfinite(tL[i]):
            ax.text(x[i] + width / 2, tL[i] + 0.025 * raw_max, str(cldL[g]).lower() if late_upper else str(cldL[g]).upper(),
                    ha='center', va='bottom', fontsize=FS, clip_on=False)
        yp = max(tL[i] if np.isfinite(tL[i]) else -np.inf, tS[i] if np.isfinite(tS[i]) else -np.inf)
        if strs.get(g, "") and np.isfinite(yp):
            ax.text(x[i], yp + 0.05 * raw_max, strs[g], ha='center', va='bottom', fontsize=FS, clip_on=False)
    if ylabel: ax.set_ylabel(ylabel, fontsize=PUB_FS_AXIS)
    ax.set_xticks(x)
    ax.set_xticklabels(genos if show_xlbl else [''] * len(genos), rotation=18, ha='right', fontsize=PUB_FS_AXIS)
    _spines(ax)
    _inset_label(ax, panel_label, PUB_FS_PANEL)

def _tradeoff_panel(ax, soil, rep_trade, agg_trade, genos, scol, gcol, dcol, late_lbl, sev_lbl, panel_label,
                    show_ylabel=False, show_xlabel=True, xlim=(0., 0.0025)):
    """Transpiration vs normalised transpiration (genotype means +- SE); grey ellipse where a genotype's drop is significant."""
    ss, sr = agg_trade[agg_trade[scol] == soil], rep_trade[rep_trade[scol] == soil].copy()
    mpts = {}
    for g in genos:
        for drt, mk in [(late_lbl, 'o'), (sev_lbl, 's')]:
            sub = ss[(ss[gcol] == g) & (ss[dcol] == drt)]
            if sub.empty: continue
            xv, xse = float(sub['NormTrans_mean'].iloc[0]), float(sub['NormTrans_se'].iloc[0])
            yv, yse = float(sub['Trans_mean'].iloc[0]), float(sub['Trans_se'].iloc[0])
            mpts[(g, drt)] = (xv, yv, xse, yse)
            ax.errorbar([xv], [yv], xerr=None if (not np.isfinite(xse) or xse <= 0) else [xse],
                        yerr=None if (not np.isfinite(yse) or yse <= 0) else [yse], fmt='none', ecolor='black', alpha=0.8,
                        elinewidth=1.15, capsize=5.5, capthick=1.15, zorder=2)
            c = GENO_COLORS_M5.get(g, '#777')
            ax.scatter([xv], [yv], s=175, marker=mk, facecolors=c, edgecolors=c, linewidths=SPINE_W, zorder=3)
    pY, pX = {str(g): np.nan for g in genos}, {str(g): np.nan for g in genos}
    if not sr.empty:
        for val_col, p_dict in [('Trans_for_norm', pY), ('NormTrans', pX)]:
            t = _m5_data(sr, scol, gcol, dcol, val_col, soil, late_lbl, sev_lbl).rename(columns={val_col: 'Value'})[["Treatment", "Genotype", "Value"]]
            if not t.empty and t["Treatment"].nunique() >= 2:
                try:
                    fD, _, dD, _, wD = _fit_choose(t, "Y ~ C(Treatment) * C(Genotype)" if t["Genotype"].nunique() >= 2 else "Y ~ C(Treatment)",
                                                   min_rows=4)
                    p_dict.update(_p_by_genotype_from_fit(fD, dD, wD, genos, LAB_M5, LAB_M3, "Treatment"))
                except Exception: pass
        for g in map(str, genos):
            rl, rs2 = sr[(sr[gcol] == g) & (sr[dcol] == late_lbl)], sr[(sr[gcol] == g) & (sr[dcol] == sev_lbl)]
            if rl.empty or rs2.empty: continue
            v = lambda d, c: (lambda a: a[np.isfinite(a)])(pd.to_numeric(d[c], errors='coerce').to_numpy(float))
            lx, ly, sx, sy = v(rl, 'NormTrans'), v(rl, 'Trans_for_norm'), v(rs2, 'NormTrans'), v(rs2, 'Trans_for_norm')
            for p, cond, key, xs_, ys_, fc, al in [(pY.get(g, np.nan), ly.size and sy.size and np.nanmedian(ly) > np.nanmedian(sy), (g, late_lbl), lx, ly, '#d0d0d0', 0.35),
                                                   (pX.get(g, np.nan), lx.size and sx.size and np.nanmedian(sx) > np.nanmedian(lx), (g, sev_lbl), sx, sy, '#aaaaaa', 0.40)]:
                if np.isfinite(p) and p < 0.05 and cond and key in mpts:
                    cx, cy, xse, yse = mpts[key]
                    w = max(4 * (np.nanstd(xs_, ddof=1) if xs_.size > 1 else 0.), (xse if np.isfinite(xse) else 0.), 0.00005)
                    h = max(4 * (np.nanstd(ys_, ddof=1) if ys_.size > 1 else 0.), (yse if np.isfinite(yse) else 0.), 0.05)
                    if w <= 0.0015 and h <= 1.:
                        ax.add_patch(Ellipse((cx, cy), width=w, height=h, facecolor=fc, edgecolor='none', alpha=al, zorder=1))
    ax.set_xlim(*xlim)
    if show_xlabel: ax.set_xlabel(r'Normalised Transpiration (cm$^3$ h$^{-1}$ cm$^{-2}$)', fontsize=PUB_FS_AXIS)
    if show_ylabel: ax.set_ylabel(r'Transpiration (cm$^3$ h$^{-1}$)', fontsize=PUB_FS_AXIS)
    _spines(ax)
    _inset_label(ax, panel_label, PUB_FS_PANEL)

def _save_table_m5(df, path, title, footnote=None, id_cols=None):
    save_df_as_table_png(df, path, title, footnote, id_cols=id_cols, none_vals=(["none"], ["None"], [""]),
                         wide={"Infer": 22, "Inference": 22, "Note": 28, "Notes": 28, "Transf details": 28, "Transform_details": 28})

def _anova_row(rep_df, scol, gcol, dcol, vcol, soil, late_lbl, sev_lbl):
    """Y ~ C(Treatment) * C(Genotype) within one soil (LowLate vs LowConst)."""
    sub = _m5_data(rep_df, scol, gcol, dcol, vcol, soil, late_lbl, sev_lbl)
    empty = {"N": 0, "Transf": "", "Infer": "", "p(norm)": np.nan, "p(BP)": np.nan, "Assump": "No", "p(Treatment)": np.nan,
             "p(Geno)": np.nan, "p(Int)": np.nan, "Sig effects": "none", "Note": "insufficient data", "Transf details": ""}
    if sub.empty or sub[gcol].nunique() < 2 or sub[dcol].nunique() < 2 or sub[vcol].count() < 6:
        return empty
    tmp = pd.DataFrame({"Treatment": sub["Treatment"].to_numpy(), "Genotype": sub["Genotype"].to_numpy(), "Value": sub[vcol].to_numpy(float)})
    try:
        f, tn, d, meta, work = _fit_choose(tmp, "Y ~ C(Treatment) * C(Genotype)", min_rows=4)
        pt, mt = _p4t(f, "C(Treatment)", 2, True, d)
        pg, _ = _p4t(f, "C(Genotype)", 2, True, d)
        pi, _ = _p4t(f, "C(Treatment):C(Genotype)", 2, True, d)
        if not np.isfinite(pi): pi, _ = _p4t(f, "C(Genotype):C(Treatment)", 2, True, d)
        infer = "robust_hc3" if mt == "robust_hc3" else "standard_anova" if d.get("assumptions_met") else "standard_anova_assump_failed"
        note = ("assumptions failed; HC3 robust p-values" if infer == "robust_hc3" else "" if infer == "standard_anova"
                else "assumptions failed; classical ANOVA shown")
        return {"N": int(work.shape[0]), "Transf": tn, "Infer": infer, "p(norm)": d.get("p_norm", np.nan), "p(BP)": d.get("p_hetero", np.nan),
                "Assump": "Yes" if d.get("assumptions_met") else "No", "p(Treatment)": pt, "p(Geno)": pg, "p(Int)": pi,
                "Sig effects": _sig_label_m5(pt, pg, pi), "Note": note, "Transf details": meta.get("details", "")}
    except Exception as e:
        return {**empty, "Note": f"fit_failed:{type(e).__name__}"}

def _fc_table(rep_df, scol, gcol, dcol, vcol, soils, late_lbl, sev_lbl):
    """Per genotype: N, mean, SE of LowLate and LowConst, ratio LowConst/LowLate, % change; plus an all-genotype row per soil."""
    df = rep_df.copy(); df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=[scol, gcol, dcol, vcol])
    df = df[df[dcol].isin([late_lbl, sev_lbl])].copy()
    df["Treat2"] = df[dcol].map({late_lbl: LAB_M5, sev_lbl: LAB_M3}).astype(str)
    agg = df.groupby([scol, gcol, "Treat2"], as_index=False).agg(N=(vcol, "count"), Mean=(vcol, "mean"), SE=(vcol, _se))
    wide = agg.pivot_table(index=[scol, gcol], columns="Treat2", values=["N", "Mean", "SE"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]; wide = wide.reset_index()
    for lb in (LAB_M5, LAB_M3):
        for pf in ("N", "Mean", "SE"):
            if f"{pf}_{lb}" not in wide.columns: wide[f"{pf}_{lb}"] = np.nan
    m5, m3 = wide[f"Mean_{LAB_M5}"].to_numpy(float), wide[f"Mean_{LAB_M3}"].to_numpy(float)
    ratio, pct = np.full_like(m5, np.nan, float), np.full_like(m5, np.nan, float)
    ok = np.isfinite(m5) & np.isfinite(m3) & (m5 != 0)
    ratio[ok] = m3[ok] / m5[ok]; pct[ok] = 100. * (m3[ok] - m5[ok]) / m5[ok]
    wide[f"Ratio_{LAB_M3}/{LAB_M5}"] = ratio; wide["Percent_change_%"] = pct
    wide = wide[wide[scol].isin(soils)].copy()
    overall = []
    for soil in soils:
        ss = wide[wide[scol] == soil]
        if ss.empty: continue
        gm5, gm3 = pd.to_numeric(ss[f"Mean_{LAB_M5}"], errors="coerce"), pd.to_numeric(ss[f"Mean_{LAB_M3}"], errors="coerce")
        mm5 = float(np.nanmean(gm5)) if gm5.notna().any() else np.nan
        mm3 = float(np.nanmean(gm3)) if gm3.notna().any() else np.nan
        ok1 = np.isfinite(mm5) and np.isfinite(mm3) and mm5 != 0
        overall.append({scol: soil, gcol: "All genotypes (mean of genotype means)", f"N_{LAB_M5}": int(gm5.notna().sum()), f"Mean_{LAB_M5}": mm5,
                        f"SE_{LAB_M5}": _se(gm5), f"N_{LAB_M3}": int(gm3.notna().sum()), f"Mean_{LAB_M3}": mm3, f"SE_{LAB_M3}": _se(gm3),
                        f"Ratio_{LAB_M3}/{LAB_M5}": mm3 / mm5 if ok1 else np.nan, "Percent_change_%": 100. * (mm3 - mm5) / mm5 if ok1 else np.nan})
    return pd.concat([pd.DataFrame(overall), wide], ignore_index=True) if overall else wide

def combined_cam2_sixpanel(df_code2, window_rows1, outdir=OUT_M5VSM3, xlim_tradeoff=(0., 0.0025)):
    FIG_SCALE = 1.30 * PUB_FIG_GROWTH
    s2, g2, r2, c2, d2, hx = 'Soil', 'Genotype', 'Rep', 'Cam', 'DroughtLevel', 'hXylem_Ψx'
    s1, g1, r1, d1 = 'Soil', 'Genotype', 'Replicate', 'Drought Category'
    b2 = df_code2.copy(); b2.columns = [str(c).strip() for c in b2.columns]
    b2[s2] = b2[s2].map(_csoil); b2[d2] = b2[d2].map(_cdrt2); b2[g2] = b2[g2].map(_cgeno_p).astype(str)
    b2 = b2[(b2[c2].astype(str) == "Cam2") & b2[s2].isin([SOIL_SILT, SOIL_SAND]) & b2[d2].isin([DL_LATE, DL_SEV]) & b2[g2].isin(GENO_ORDER)].copy()
    b2[hx] = pd.to_numeric(b2[hx], errors="coerce")
    b2 = b2.dropna(subset=[hx]); b2[hx] = b2[hx].clip(lower=1e-6)
    rep_hx = _rep_means(b2, s2, g2, d2, r2, hx, _csoil, _cdrt2)
    agg_hx = _agg(rep_hx, s2, g2, d2, hx)
    w1 = window_rows1.copy(); w1.columns = [str(c).strip() for c in w1.columns]
    w1[s1] = w1[s1].map(_csoil); w1[d1] = w1[d1].map(_cdrt1); w1[g1] = w1[g1].map(_cgeno_p).astype(str)
    w1 = w1[w1[s1].isin([SOIL_SILT, SOIL_SAND]) & w1[d1].isin([DL_LATE, DL_SEV]) & w1[g1].isin(GENO_ORDER)].copy()
    genos = [g for g in GENO_ORDER if g in set(rep_hx[g2].unique()) or g in set(w1[g1].unique())]
    w1['_Area'] = pd.to_numeric(w1['Area_filled'], errors='coerce')
    rep_area = _rep_means(w1, s1, g1, d1, r1, '_Area', _csoil, _cdrt1).rename(columns={'_Area': 'Area'})
    agg_area = _agg(rep_area, s1, g1, d1, 'Area')
    w1['_Trans'], w1['_NormTrans'] = pd.to_numeric(w1['Trans_for_norm'], errors='coerce'), pd.to_numeric(w1['NormTrans'], errors='coerce')
    tmp_t = (w1.dropna(subset=[s1, g1, d1, r1, '_Trans', '_NormTrans'])
             .groupby([s1, g1, d1, r1], as_index=False).agg(Trans_for_norm=('_Trans', 'mean'), NormTrans=('_NormTrans', 'mean')))
    trade_agg = pd.DataFrame([{s1: soil, g1: geno, d1: drt, 'NormTrans_mean': float(np.nanmean(g['NormTrans'])), 'NormTrans_se': _se(g['NormTrans']),
                               'Trans_mean': float(np.nanmean(g['Trans_for_norm'])), 'Trans_se': _se(g['Trans_for_norm'])}
                              for (soil, geno, drt), g in tmp_t.groupby([s1, g1, d1])])
    rc = {"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
          "legend.fontsize": PUB_FS_LEGEND, "figure.titlesize": FS}
    leg = dict(frameon=True, framealpha=0.9, fontsize=PUB_FS_LEGEND, title_fontsize=PUB_FS_LEGEND)
    with plt.rc_context(rc):
        plt.close('all')
        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(18. * FIG_SCALE * 1.06, 15. * FIG_SCALE * PUB_FIG_PAD),
                                 gridspec_kw={"height_ratios": [1.25, 1.25, 1.00]})
        fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06, hspace=0.02, wspace=0.03)
        for ax_, soil_, lbl_ in [(axes[0, 0], SOIL_SILT, '(a)'), (axes[0, 1], SOIL_SAND, '(b)')]:
            _bar_panel(ax_, soil_, rep_hx, agg_hx, hx, genos, s2, g2, d2, DL_LATE, DL_SEV, lbl_, yscale="linear",
                       ylabel=r'Xylem water potential, $h_{\mathrm{x,md}}$ (kPa)' if soil_ == SOIL_SILT else None)
        for ax_, soil_, lbl_ in [(axes[1, 0], SOIL_SILT, '(c)'), (axes[1, 1], SOIL_SAND, '(d)')]:
            _bar_panel(ax_, soil_, rep_area, agg_area, 'Area', genos, s1, g1, d1, DL_LATE, DL_SEV, lbl_, show_xlbl=True,
                       ylabel=r'Shoot area (cm$^2$)' if soil_ == SOIL_SILT else None)
        _tradeoff_panel(axes[2, 0], SOIL_SILT, tmp_t, trade_agg, genos, s1, g1, d1, DL_LATE, DL_SEV, '(e)', show_ylabel=True, xlim=xlim_tradeoff)
        _tradeoff_panel(axes[2, 1], SOIL_SAND, tmp_t, trade_agg, genos, s1, g1, d1, DL_LATE, DL_SEV, '(f)', xlim=xlim_tradeoff)
        axes[0, 0].legend(handles=[Patch(facecolor=SOIL_COLORS[SOIL_SILT], edgecolor='white', hatch=HATCH, linewidth=BAR_EDGE_W, label=DISP_M3),
                                   Patch(facecolor='white', edgecolor=SOIL_COLORS[SOIL_SILT], hatch=HATCH, linewidth=BAR_EDGE_W, label=DISP_M5)],
                          loc="upper right", bbox_to_anchor=(0.98, 0.97), borderaxespad=0, handlelength=1.2, handletextpad=0.4, labelspacing=0.25, **leg)
        axes[0, 1].legend(handles=[Patch(facecolor=SOIL_COLORS[s], edgecolor='black', label=s) for s in (SOIL_SILT, SOIL_SAND)],
                          loc="upper right", bbox_to_anchor=(0.98, 0.97), borderaxespad=0, handlelength=1.2, handletextpad=0.4, labelspacing=0.25, **leg)
        dot = lambda mk, fc, lab: Line2D([0], [0], marker=mk, linestyle='None', markersize=FS / 2, markerfacecolor=fc, markeredgecolor='black', label=lab)
        axes[2, 1].legend([dot('o', GENO_COLORS_M5.get(g, '#777'), g) for g in genos] + [dot('o', '#777', DISP_M5), dot('s', '#777', DISP_M3)],
                          list(genos) + [DISP_M5, DISP_M3], loc='upper right', title="Genotype & treatment", borderaxespad=0.2,
                          handlelength=1.0, handletextpad=0.4, labelspacing=0.3, **leg)
        fig.savefig(os.path.join(outdir, f"Cam2_Combined_6panel_hXylemPsi_ShootArea_Tradeoff_{LAB_M5}_vs_{LAB_M3}_ANOVA.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ---- supplementary tables ----
    soils = [SOIL_SILT, SOIL_SAND]
    panel_for = lambda par, soil: (f"({chr(97 + {'hx,md': 0, 'Shoot Area': 2, 'Transpiration': 4, 'Trans_Norm': 4}[str(par).strip()] + (0 if str(soil).strip() == 'Silt Loam' else 1))})"
                                   if str(par).strip() in ('hx,md', 'Shoot Area', 'Transpiration', 'Trans_Norm') else "")
    params = [("hx,md", rep_hx, s2, g2, d2, hx), ("Shoot Area", rep_area, s1, g1, d1, "Area"),
              ("Transpiration", tmp_t, s1, g1, d1, "Trans_for_norm"), ("Trans_Norm", tmp_t, s1, g1, d1, "NormTrans")]
    da = pd.DataFrame([{"Parameter": pn, "Soil": soil, **_anova_row(rdf, sc, gc, dc, vc, soil, DL_LATE, DL_SEV)}
                       for pn, rdf, sc, gc, dc, vc in params for soil in soils])
    for col in ["p(norm)", "p(BP)", "p(Treatment)", "p(Geno)", "p(Int)"]:
        da[col] = da[col].apply(_fp)
    da["N"] = da["N"].apply(_fi)
    da.insert(0, "Panel", da.apply(lambda r: panel_for(r["Parameter"], r["Soil"]), axis=1))
    _save_table_m5(da[["Panel", "Parameter", "Soil", "N", "Transf", "Infer", "p(norm)", "p(BP)", "Assump", "p(Treatment)", "p(Geno)", "p(Int)", "Sig effects"]],
                   os.path.join(outdir, f"SUPP_Table_ANOVA_{LAB_M5}_vs_{LAB_M3}_within_soil.png"),
                   f"Supplementary Table. Two-way ANOVA within soil ({DISP_M5} vs {DISP_M3}): Treatment, Genotype, Treatment×Genotype",
                   f"Model: Value ~ Treatment * Genotype. {DISP_M3}=LowConst, {DISP_M5}=LowLate. Standard ANOVA if normality+BP pass; else transforms; else HC3 robust.",
                   id_cols=["Panel", "Parameter", "Soil"])
    for rdf, sc, gc, dc, vc, pname, fname in [(rep_hx, s2, g2, d2, hx, "hXylem", "hXylemPsi"), (rep_area, s1, g1, d1, "Area", "Shoot Area", "ShootArea"),
                                              (tmp_t, s1, g1, d1, "Trans_for_norm", "Transpiration", "Transpiration"),
                                              (tmp_t, s1, g1, d1, "NormTrans", "Trans_Norm", "NormTrans")]:
        fc = _fc_table(rdf, sc, gc, dc, vc, soils, DL_LATE, DL_SEV).rename(columns={sc: "Soil", gc: "Genotype"})
        for col in [f"Mean_{LAB_M5}", f"SE_{LAB_M5}", f"Mean_{LAB_M3}", f"SE_{LAB_M3}"]:
            fc[col] = pd.to_numeric(fc[col], errors="coerce").map(lambda x: _ff(x, 4))
        fc[f"N_{LAB_M5}"], fc[f"N_{LAB_M3}"] = fc[f"N_{LAB_M5}"].map(_fi), fc[f"N_{LAB_M3}"].map(_fi)
        fc[f"Ratio_{LAB_M3}/{LAB_M5}"] = pd.to_numeric(fc[f"Ratio_{LAB_M3}/{LAB_M5}"], errors="coerce").map(lambda x: _ff(x, 3))
        fc["Percent_change_%"] = pd.to_numeric(fc["Percent_change_%"], errors="coerce").map(lambda x: _ff(x, 1))
        fc.insert(0, "Parameter", pname)
        fc.insert(0, "Panel", fc["Soil"].map(lambda s: panel_for(pname, s)))
        _save_table_m5(fc[["Panel", "Parameter", "Soil", "Genotype", f"N_{LAB_M5}", f"Mean_{LAB_M5}", f"SE_{LAB_M5}", f"N_{LAB_M3}", f"Mean_{LAB_M3}",
                           f"SE_{LAB_M3}", f"Ratio_{LAB_M3}/{LAB_M5}", "Percent_change_%"]],
                       os.path.join(outdir, f"SUPP_Table_FoldChange_{fname}_{LAB_M3}_over_{LAB_M5}.png"),
                       f"Supplementary Table. Fold-change ({DISP_M3}/{DISP_M5}) — {pname}",
                       f"Ratio=Mean({LAB_M3})/Mean({LAB_M5}). Percent_change=100*({LAB_M3}-{LAB_M5})/{LAB_M5}. {LAB_M3}={DISP_M3}, {LAB_M5}={DISP_M5}. Values from replicate means.",
                       id_cols=["Panel", "Parameter", "Soil", "Genotype"])

combined_cam2_sixpanel(df_code2, window_rows_28_30)


# ================================ [DRYDOWN] Fig S7: K(h) + MFP -> transpiration, per genotype, Silt Loam ================================
# Trans ~ K(h) + MFP (OLS per genotype on replicate means, LowLate + LowConst); prediction at the LowConst K(h) and MFP
import matplotlib.gridspec as mgs
import statsmodels.api as sm_api
GENO_MK = {'Alpenfex': 'o', 'Braunes Schindelmeiser': 's', 'Pfarrkirchner': '^', 'Benedictio KWS': 'D', 'Luigi CS': 'v', 'P8888': 'P'}

def _stars_ns(p):
    if not np.isfinite(p): return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def kh_mfp_trans_pergenotype(window_rows1, outdir=OUT_M5VSM3, min_n=6):
    FIG_SCALE, LATE, SEV = 1.10 * PUB_FIG_GROWTH, r'Low$_{\mathrm{Late}}$', r'Low$_{\mathrm{Const}}$'
    df = window_rows1.copy()
    df.columns = [str(c).strip() for c in df.columns]
    soil_map = {'Silt': 'Silt Loam', 'SiltLoam': 'Silt Loam', 'Silt Loam': 'Silt Loam', 'Sand': 'Sandy Loam', 'Sandy': 'Sandy Loam', 'Sandy Loam': 'Sandy Loam'}
    df['Soil'] = df['Soil'].map(lambda x: soil_map.get(str(x).strip(), str(x).strip()))
    df['Drought Category'] = df['Drought Category'].astype(str).str.strip()
    df['Genotype'] = df['Genotype'].map(_cgeno_p).astype(str)
    df = df[(df['Soil'] == SOIL_SILT) & df['Drought Category'].isin([DL_LATE, DL_SEV])].copy()
    for col in ['K(h)', 'MFP', 'Trans_for_norm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['K(h)', 'MFP', 'Trans_for_norm', 'Genotype', 'Drought Category', 'Replicate'])
    df = df[(df['K(h)'] > 0) & (df['MFP'] > 0)].copy()
    rep = df.groupby(['Genotype', 'Drought Category', 'Replicate'], as_index=False).agg(Kh=('K(h)', 'mean'), MFP=('MFP', 'mean'),
                                                                                        Trans=('Trans_for_norm', 'mean'))
    s_late, s_sev = rep[rep['Drought Category'] == DL_LATE], rep[rep['Drought Category'] == DL_SEV]
    obs_gap = float(s_late['Trans'].mean()) - float(s_sev['Trans'].mean())
    kh_ratios, mfp_ratios, results, models = [], [], [], {}
    for g in GENO_ORDER:                                        # LowLate / LowConst ratios of K(h) and MFP
        gl, gs = s_late[s_late['Genotype'] == g], s_sev[s_sev['Genotype'] == g]
        if gl.empty or gs.empty: continue
        khs, mfps = gs['Kh'].mean(), gs['MFP'].mean()
        kh_ratios.append(gl['Kh'].mean() / khs if khs > 0 else np.nan)
        mfp_ratios.append(gl['MFP'].mean() / mfps if mfps > 0 else np.nan)
    for g in GENO_ORDER:                                        # per-genotype model
        g_all = rep[rep['Genotype'] == g].copy()
        gl, gs = s_late[s_late['Genotype'] == g], s_sev[s_sev['Genotype'] == g]
        if len(g_all) < min_n or gl.empty or gs.empty:
            continue
        try:
            model = sm_api.OLS(g_all['Trans'].values, sm_api.add_constant(g_all[['Kh', 'MFP']].values)).fit()
        except Exception:
            continue
        models[g] = model
        obs_t_late = float(gl['Trans'].mean())
        pred_t = float(model.predict(np.array([[1.0, float(gs['Kh'].mean()), float(gs['MFP'].mean())]]))[0])
        results.append(dict(Genotype=g, r2=model.rsquared, p_model=float(model.f_pvalue), obs_t_late=obs_t_late,
                            obs_t_sev=float(gs['Trans'].mean()), pred_t=pred_t,
                            pct_chg=100. * (pred_t - obs_t_late) / obs_t_late if obs_t_late != 0 else np.nan))
    res = pd.DataFrame(results)
    wn = res['r2'].values.clip(min=1e-6); wn = wn / wn.sum()           # R²-weighted summary
    w_pred, w_obs_late, w_obs_sev = (float(np.average(res[c].values, weights=wn)) for c in ('pred_t', 'obs_t_late', 'obs_t_sev'))
    w_pct = 100. * (w_pred - w_obs_late) / w_obs_late if w_obs_late != 0 else np.nan
    expl_pct = 100. * (w_obs_late - w_pred) / obs_gap if obs_gap != 0 else np.nan
    corr_ratio = w_pred / w_obs_sev if w_obs_sev != 0 else np.nan
    rep['Pred'] = np.nan
    for g, model in models.items():
        idx = rep[rep['Genotype'] == g].index
        rep.loc[idx, 'Pred'] = model.predict(sm_api.add_constant(rep.loc[idx, ['Kh', 'MFP']].values))
    vals = rep[['Trans', 'Pred']].values.ravel()
    pad = (np.nanmax(vals) - np.nanmin(vals)) * 0.06
    lim = (max(0., np.nanmin(vals) - pad), np.nanmax(vals) + pad)
    rc = {'font.size': PUB_FS_TICK, 'axes.titlesize': PUB_FS_AXIS, 'axes.labelsize': PUB_FS_AXIS, 'xtick.labelsize': PUB_FS_TICK,
          'ytick.labelsize': PUB_FS_TICK, 'legend.fontsize': PUB_FS_LEGEND}
    with plt.rc_context(rc):
        plt.close('all')
        fig = plt.figure(figsize=(18 * FIG_SCALE, 12 * FIG_SCALE * PUB_FIG_PAD))
        fig.suptitle(r'Supporting Information Fig. S7  |  K(h) + MFP → Trans per genotype  |  Silt Loam  |  Nov 28–30' '\n'
                     r'Predicted (K(h) + MFP model) vs Observed transpiration  |  Low$_{\mathrm{Late}}$ (filled) vs Low$_{\mathrm{Const}}$ (open)',
                     fontsize=PUB_FS_AXIS, fontweight='bold', y=1.02)
        grid = mgs.GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22, left=0.07, right=0.98, top=0.87, bottom=0.10)
        for idx, g in enumerate(GENO_ORDER):
            ax = fig.add_subplot(grid[idx // 3, idx % 3])
            gc, mk = GENO_COLORS.get(g, '#555'), GENO_MK.get(g, 'o')
            g_all = rep[rep['Genotype'] == g].dropna(subset=['Pred'])
            gl, gs = g_all[g_all['Drought Category'] == DL_LATE], g_all[g_all['Drought Category'] == DL_SEV]
            ax.plot(lim, lim, color='grey', lw=1.2, linestyle='--', alpha=0.6, zorder=1)
            if not gl.empty:
                ax.scatter(gl['Pred'], gl['Trans'], s=90, marker=mk, facecolors=gc, edgecolors='black', linewidths=0.7, alpha=0.85, zorder=5, label=LATE)
            if not gs.empty:
                ax.scatter(gs['Pred'], gs['Trans'], s=90, marker=mk, facecolors='white', edgecolors=gc, linewidths=1.6, alpha=0.75, zorder=5, label=SEV)
            if g in models:                    # mean LowLate (diamond) -> prediction at LowConst conditions (red star)
                row = res[res['Genotype'] == g].iloc[0]
                mp = float(gl['Pred'].mean())
                ax.scatter([mp], [row['obs_t_late']], s=220, marker='D', facecolors=gc, edgecolors='black', linewidths=1.5, zorder=8)
                ax.scatter([row['pred_t']], [row['pred_t']], s=240, marker='*', facecolors='red', edgecolors='darkred', linewidths=0.8, zorder=9)
                ax.annotate('', xy=(row['pred_t'], row['pred_t']), xytext=(mp, row['obs_t_late']), zorder=10,
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.8, linestyle='dashed', connectionstyle='arc3,rad=-0.20'))
                sig = row['p_model'] < 0.05
                ax.text(0.97, 0.03, f"R²={row['r2']:.3f} {_stars_ns(row['p_model'])}\nΔTrans={row['pct_chg']:+.1f}%\nweight={row['r2'] / res['r2'].sum():.2f}",
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=PUB_FS_LEGEND - 2, zorder=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9' if sig else '#FFF8F0', edgecolor='#2E7D32' if sig else '#999', alpha=0.90))
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_title(g, fontsize=PUB_FS_AXIS, fontweight='bold', color=gc, pad=5)
            ax.text(0.02, 0.97, f"({chr(97 + idx)})", transform=ax.transAxes, ha='left', va='top', fontsize=PUB_FS_PANEL, fontweight='bold', clip_on=False)
            if idx % 3 == 0:
                ax.set_ylabel(r'Observed Trans  (cm$^3$ h$^{-1}$)', fontsize=PUB_FS_AXIS)
            if idx // 3 == 1:
                ax.set_xlabel(r'Predicted Trans  (cm$^3$ h$^{-1}$)', fontsize=PUB_FS_AXIS)
            ax.yaxis.grid(True, linestyle=':', alpha=0.35); ax.xaxis.grid(True, linestyle=':', alpha=0.35)
            for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(1.1)
            ax.tick_params(direction='inout', length=4, width=1.1); ax.set_axisbelow(True)
            if idx == 0:
                mkl = lambda m, ms, fc, ec, lab, **k: Line2D([0], [0], marker=m, linestyle='None', markersize=ms, markerfacecolor=fc, markeredgecolor=ec, label=lab, **k)
                ax.legend(handles=[mkl(mk, 8, gc, 'black', LATE + ' (filled)'), mkl(mk, 8, 'white', gc, SEV + ' (open)', markeredgewidth=1.5),
                                   Line2D([0], [0], color='grey', lw=1.2, linestyle='--', label='1:1 line'), mkl('D', 9, gc, 'black', 'Mean ' + LATE),
                                   mkl('*', 11, 'red', 'darkred', r'Pred at Low$_{\mathrm{Const}}$ K(h)+MFP')],
                          loc='upper left', fontsize=PUB_FS_LEGEND - 5, frameon=True, framealpha=0.9, handlelength=1.0, handletextpad=0.3,
                          labelspacing=0.2, borderpad=0.3, markerscale=0.8)
        fig.text(0.50, 0.01, f'R²-weighted:  ΔTrans={w_pct:+.1f}%  |  Explains ≈{expl_pct:.0f}% of gap  |  Corrected ratio={corr_ratio:.2f}×  |  '
                             f'Mean K(h) ratio={np.mean(kh_ratios):.1f}×  Mean MFP ratio={np.mean(mfp_ratios):.1f}×',
                 ha='center', va='bottom', fontsize=PUB_FS_LEGEND, style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#AAAAAA', alpha=0.9))
        fig.savefig(os.path.join(outdir, 'FigS7_KhMFP_Trans_pergenotype_SiltLoam.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    return res

kh_mfp_results = kh_mfp_trans_pergenotype(window_rows_28_30)


# ================================ [ROOT-TRAITS] 8-panel root traits, LowLate vs LowConst (Cam2) ================================
# NOTE: this cell never ran in the original main.py (it called _remove_outliers_iqr / _diagnose_coverage, which were not
# defined, and asked for table columns the helpers do not make). It is switched off; set RUN_ROOT_TRAITS = True to run it.
# The outlier rule below (drop replicate means outside Q1 - 1.5 IQR .. Q3 + 1.5 IQR within each soil) is a reconstruction.
RUN_ROOT_TRAITS = False
OUTLIER_IQR_K = 1.5

def _remove_outliers_iqr(rep, scol, vcol, iqr_k=OUTLIER_IQR_K):
    """Drop replicate means outside Q1 - k*IQR .. Q3 + k*IQR within each soil."""
    keep = pd.Series(True, index=rep.index)
    for soil, g in rep.groupby(scol):
        q1, q3 = np.nanpercentile(g[vcol], 25), np.nanpercentile(g[vcol], 75)
        keep.loc[g.index] = g[vcol].between(q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1))
    return rep[keep].copy()

def root_traits_cam2_eightpanel(df_code2, outdir=OUT_M5VSM3):
    global FS
    FIG_SCALE, s2, g2, r2, d2 = 1.10 * PUB_FIG_GROWTH, 'Soil', 'Genotype', 'Rep', 'DroughtLevel'
    traits = [('trl', 'Total root length (m)', 'trl'), ('rdmean', 'Root diameter (mm)', 'rdmean'),
              ('srm_upscaled', 'Rhizosheath mass (g)', 'srm'), ('Root_to_Shoot', 'Root-to-shoot ratio (–)', 'RootToShoot')]
    b2 = df_code2.copy(); b2.columns = [str(c).strip() for c in b2.columns]
    b2[s2] = b2[s2].map(_csoil); b2[d2] = b2[d2].map(_cdrt2); b2[g2] = b2[g2].map(_cgeno_p).astype(str)
    b2 = b2[(b2['Cam'].astype(str) == "Cam2") & b2[s2].isin([SOIL_SILT, SOIL_SAND]) & b2[d2].isin([DL_LATE, DL_SEV]) & b2[g2].isin(GENO_ORDER)].copy()
    for col, *_ in traits:
        b2[col] = pd.to_numeric(b2[col], errors="coerce")
    genos = [g for g in GENO_ORDER if g in set().union(*[set(b2.dropna(subset=[c])[g2].unique()) for c, *_ in traits])]
    reps = {c: _remove_outliers_iqr(_rep_means(b2.dropna(subset=[c]), s2, g2, d2, r2, c, _csoil, _cdrt2), s2, c) for c, *_ in traits}
    rc = {"font.size": PUB_FS_TICK, "axes.titlesize": PUB_FS_AXIS, "axes.labelsize": PUB_FS_AXIS, "xtick.labelsize": PUB_FS_AXIS,
          "ytick.labelsize": PUB_FS_TICK, "legend.fontsize": PUB_FS_LEGEND, "figure.titlesize": PUB_FS_AXIS + 2}
    leg = dict(loc="upper right", bbox_to_anchor=(0.98, 0.97), frameon=True, framealpha=0.9, fontsize=PUB_FS_LEGEND,
               title_fontsize=PUB_FS_LEGEND, borderaxespad=0, handlelength=1.2, handletextpad=0.4, labelspacing=0.25)
    FS_BACKUP, FS = FS, PUB_FS_AXIS                    # bigger letters in this figure
    try:
        with plt.rc_context(rc):
            plt.close('all')
            fig, axes = plt.subplots(4, 2, constrained_layout=True, figsize=(18. * FIG_SCALE * 1.06, 21. * FIG_SCALE * PUB_FIG_PAD))
            fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06, hspace=0.02, wspace=0.03)
            for r, (c, label, _) in enumerate(traits):
                for k, soil in enumerate((SOIL_SILT, SOIL_SAND)):
                    _bar_panel(axes[r, k], soil, reps[c], _agg(reps[c], s2, g2, d2, c), c, genos, s2, g2, d2, DL_LATE, DL_SEV,
                               f"({chr(97 + 2 * r + k)})", ylabel=label if k == 0 else None, show_xlbl=(r == 3))
            axes[0, 0].legend(handles=[Patch(facecolor=SOIL_COLORS[SOIL_SILT], edgecolor='white', hatch=HATCH, linewidth=BAR_EDGE_W, label=DISP_M3),
                                       Patch(facecolor='white', edgecolor=SOIL_COLORS[SOIL_SILT], hatch=HATCH, linewidth=BAR_EDGE_W, label=DISP_M5)], **leg)
            axes[0, 1].legend(handles=[Patch(facecolor=SOIL_COLORS[s], edgecolor='black', label=s) for s in (SOIL_SILT, SOIL_SAND)], **leg)
            fig.savefig(os.path.join(outdir, f"Cam2_RootTraits_8panel_{LAB_M5}_vs_{LAB_M3}_ANOVA.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
    finally:
        FS = FS_BACKUP
    note = f"Outliers removed using IQR×{OUTLIER_IQR_K} per soil before analysis."
    panel_for = lambda par, soil: f"({chr(97 + 2 * [t[1] for t in traits].index(par) + (0 if soil == SOIL_SILT else 1))})"
    da = pd.DataFrame([{"Parameter": label, "Soil": soil, **_anova_row(reps[c], s2, g2, d2, c, soil, DL_LATE, DL_SEV)}
                       for c, label, _ in traits for soil in (SOIL_SILT, SOIL_SAND)])
    for col in ["p(norm)", "p(BP)", "p(Treatment)", "p(Geno)", "p(Int)"]:
        da[col] = da[col].apply(_fp)
    da["N"] = da["N"].apply(_fi)
    da.insert(0, "Panel", da.apply(lambda r: panel_for(r["Parameter"], r["Soil"]), axis=1))
    _save_table_m5(da[["Panel", "Parameter", "Soil", "N", "Transf", "Infer", "p(norm)", "p(BP)", "Assump", "p(Treatment)", "p(Geno)", "p(Int)", "Sig effects"]]
                   .rename(columns={"p(Treatment)": f"p({LAB_M3} vs {LAB_M5})"}),
                   os.path.join(outdir, f"SUPP_Table_ANOVA_RootTraits_{LAB_M5}_vs_{LAB_M3}_within_soil.png"),
                   f"Supplementary Table. Two-way ANOVA within soil — Root traits ({DISP_M5} vs {DISP_M3}): Treatment, Genotype, Treatment×Genotype",
                   f"Model: Value ~ Treatment * Genotype. {DISP_M3}=LowConst, {DISP_M5}=LowLate. Standard ANOVA if normality+BP pass; else transforms; "
                   f"else HC3 robust. " + note, id_cols=["Panel", "Parameter", "Soil"])
    for c, label, fname in traits:
        fc = _fc_table(reps[c], s2, g2, d2, c, [SOIL_SILT, SOIL_SAND], DL_LATE, DL_SEV).rename(columns={s2: "Soil", g2: "Genotype"})
        for col in [f"Mean_{LAB_M5}", f"SE_{LAB_M5}", f"Mean_{LAB_M3}", f"SE_{LAB_M3}"]:
            fc[col] = pd.to_numeric(fc[col], errors="coerce").map(lambda x: _ff(x, 4))
        fc[f"N_{LAB_M5}"], fc[f"N_{LAB_M3}"] = fc[f"N_{LAB_M5}"].map(_fi), fc[f"N_{LAB_M3}"].map(_fi)
        fc[f"Ratio_{LAB_M3}/{LAB_M5}"] = pd.to_numeric(fc[f"Ratio_{LAB_M3}/{LAB_M5}"], errors="coerce").map(lambda x: _ff(x, 3))
        fc["Percent_change_%"] = pd.to_numeric(fc["Percent_change_%"], errors="coerce").map(lambda x: _ff(x, 1))
        fc.insert(0, "Parameter", label)
        fc.insert(0, "Panel", fc["Soil"].map(lambda s: panel_for(label, s)))
        _save_table_m5(fc[["Panel", "Parameter", "Soil", "Genotype", f"N_{LAB_M5}", f"Mean_{LAB_M5}", f"SE_{LAB_M5}", f"N_{LAB_M3}", f"Mean_{LAB_M3}",
                           f"SE_{LAB_M3}", f"Ratio_{LAB_M3}/{LAB_M5}", "Percent_change_%"]],
                       os.path.join(outdir, f"SUPP_Table_FoldChange_{fname}_{LAB_M3}_over_{LAB_M5}.png"),
                       f"Supplementary Table. Fold-change ({DISP_M3}/{DISP_M5}) — {label}",
                       f"Ratio=Mean({LAB_M3})/Mean({LAB_M5}). Percent_change=100*({LAB_M3}-{LAB_M5})/{LAB_M5}. {LAB_M3}={DISP_M3}, "
                       f"{LAB_M5}={DISP_M5}. Values from replicate means. " + note, id_cols=["Panel", "Parameter", "Soil", "Genotype"])

if RUN_ROOT_TRAITS:
    root_traits_cam2_eightpanel(df_code2)