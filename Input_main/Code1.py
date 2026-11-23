
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ========= Paths =========
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))      # folder that holds this script
except NameError:                                            # run cell-by-cell (e.g. Spyder): use the working folder
    _HERE = os.getcwd()
BASE = _HERE                                                 # = Input_main/
INPUT_DIR = os.path.join(BASE, "Input_code1")       # shoot area, weight records, soil filling
# HYPROP files: Input_main\Hyprob_data if it exists, else Data&code\Hyprob_data
Path2Hyprop = next(d for d in (os.path.join(BASE, "Hyprob_data"), os.path.join(os.path.dirname(BASE), "Hyprob_data"), BASE)
                   if os.path.isdir(d))
shoot_area_file = os.path.join(INPUT_DIR, 'shoot_area_with_ID_Date_Metadata.xlsx')
filtered_file = os.path.join(INPUT_DIR, 'WeightRecords.xlsx')
soil_file = os.path.join(INPUT_DIR, 'V158_First_filling_analysis.xlsx')
HYPROP_FILES = {'Sandy Loam': 'Sandy2.xlsx', 'Silt Loam': 'Silt1.xlsx'}

DATE_MIN = pd.to_datetime("2024-11-13").normalize()
DATE_MAX = pd.to_datetime("2024-12-01").normalize()
RHO_B = {'Sandy Loam': 1.28, 'Silt Loam': 1.14}     # bulk density g/cm3
CM_H2O_TO_KPA = 0.0980665
PF_DRY, PF_WET, MFP_NPTS = 6.8, -3.0, 400_001       # MFP integration grid (uniform in pF)

# main.py runs after this script in the same Python, so these settings also apply to its figures
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "figure.dpi": 220, "savefig.dpi": 300})


# ========= Snapshot ID -> metadata (digit 4 = genotype, 5 = soil, second-last = replicate, last = drought) =========
GENOTYPES = {'1': 'Benedictio KWS', '2': 'Luigi CS', '3': 'P8888', '4': 'Alpenfex', '5': 'Braunes Schindelmeiser', '6': 'Pfarrkirchner'}
DROUGHTS = {'1': 'No Drought', '2': 'Moderate Drought', '3': 'Severe Drought', '4': 'Early Drought', '5': 'Late Drought'}

def add_metadata(df):
    sid = df['Snapshot ID Tag'].astype(str)
    df['Soil'] = sid.str[4].map({'1': 'Silt Loam', '2': 'Sandy Loam'}).fillna('Unknown')
    df['Drought Category'] = sid.str[-1].map(DROUGHTS).fillna('Unknown')
    df['Genotype'] = sid.str[3].map(GENOTYPES).fillna('Unknown')
    df['Replicate'] = "Replicate " + sid.str[-2]
    return df

def read_with_id(path):
    """Excel file with the ID column named 'Snapshot ID Tag'."""
    df = pd.read_excel(path)
    for alt in ('Snapshot ID', 'Snapshot_ID_Tag'):
        if alt in df.columns and 'Snapshot ID Tag' not in df.columns:
            df = df.rename(columns={alt: 'Snapshot ID Tag'})
    return df

def num(s):
    return pd.to_numeric(s, errors='coerce')


# ========= 1. Shoot area: daily mean, then gaps filled linearly between first and last image =========
sa = read_with_id(shoot_area_file)
if 'Date' in sa.columns:
    sa['Date'] = pd.to_datetime(sa['Date'], errors='coerce').dt.normalize()
else:
    sa['Date'] = pd.to_datetime(sa['Snapshot Time Stamp'], errors='coerce').dt.normalize()
sa['Area'] = num(sa['Area'])
sa_daily = sa.dropna(subset=['Snapshot ID Tag', 'Date']).groupby(['Snapshot ID Tag', 'Date'], as_index=False)['Area'].mean()
sa_daily = add_metadata(sa_daily)
sa_daily = sa_daily[(sa_daily['Date'] >= DATE_MIN) & (sa_daily['Date'] <= DATE_MAX)].copy()
if sa_daily.empty:
    raise RuntimeError("No shoot-area data in 2024-11-13..2024-12-01.")

def gapfill(sub):
    """Every day between the first and last valid image; missing days on the straight line between those two."""
    sub = sub[['Date', 'Area']].drop_duplicates('Date').sort_values('Date')
    s = sub.set_index('Date')['Area'].astype(float)
    t1, tN = s.first_valid_index(), s.last_valid_index()
    if t1 is None or tN is None or t1 == tN:
        return sub['Date'].values, sub['Area'].astype(float).values
    days = pd.date_range(t1, tN, freq='D')
    line = float(s.loc[t1]) + ((days - t1) / (tN - t1)).astype(float) * (float(s.loc[tN]) - float(s.loc[t1]))
    y = s.reindex(days).copy()
    y[y.isna()] = line[y.isna()]
    return days.values, y.values.astype(float)

area_filled = pd.concat([pd.DataFrame({'Snapshot ID Tag': sid, 'Date': pd.to_datetime(d), 'Area_filled': y})
                         for sid, sub in sa_daily.groupby('Snapshot ID Tag', sort=True)
                         for d, y in [gapfill(sub.sort_values('Date'))]], ignore_index=True)
meta_map = sa_daily[['Snapshot ID Tag', 'Soil', 'Genotype', 'Replicate', 'Drought Category']].drop_duplicates('Snapshot ID Tag')


# ========= 2. Dry soil mass per box (filling sheet; removed soil corrected for its water content) =========
soil = read_with_id(soil_file)
after = 'Weight After Re-filling' if 'Weight After Re-filling' in soil.columns else 'Weight After'
before = 'Weight Before Re-filling' if 'Weight Before Re-filling' in soil.columns else 'Weight Before'
dry0 = num(soil['Weight of Wet Soil']) / ((num(soil['Water Content (%)']) * 0.01) + 1)
added = num(soil.get(after, np.nan)) - num(soil.get(before, np.nan))
wc_removed = soil['Soil'].astype(str).str.lower().map(lambda s: 3.72 if s.startswith('silt') else 1.59 if s.startswith('sandy') else np.nan)
added = added.where(~((added < 0) & wc_removed.notna()), added / ((wc_removed * 0.01) + 1))
soil['New Mass of Dry Soil'] = num(dry0) + num(added)
soil_small = soil[[c for c in ['Snapshot ID Tag', 'Weight of empty boxes', 'Gravel and Frame', 'New Mass of Dry Soil', 'Soil']
                   if c in soil.columns]].drop_duplicates('Snapshot ID Tag')


# ========= 3. Weights at 17-20 h -> transpiration and soil water =========
filt = read_with_id(filtered_file)
filt['Snapshot Time Stamp'] = pd.to_datetime(filt['Snapshot Time Stamp'])
filt['Date'] = filt['Snapshot Time Stamp'].dt.normalize()
filt = filt[(filt['Date'] >= DATE_MIN) & (filt['Date'] <= DATE_MAX)].copy()
f1720 = filt[(filt['Snapshot Time Stamp'].dt.hour >= 17) & (filt['Snapshot Time Stamp'].dt.hour < 20)].copy()
f1720 = f1720.sort_values(['Snapshot ID Tag', 'Date', 'Snapshot Time Stamp'])

# transpiration: each 17-20 h reading vs the next one of the same box ("/2" convention kept)
aft = f1720.copy()
aft['Next Day Weight'] = num(aft.groupby('Snapshot ID Tag')['Weight Before'].shift(-1))
aft['Weight Before'] = num(aft['Weight Before'])
hours = num((aft.groupby('Snapshot ID Tag')['Snapshot Time Stamp'].shift(-1) - aft['Snapshot Time Stamp']).dt.total_seconds() / 3600.0)
wa_col = next((c for c in ('Weight After', 'Weight After Re-filling') if c in aft.columns), None)
water_added = (num(aft[wa_col]) - aft['Weight Before']).clip(lower=0).fillna(0) if wa_col else 0.0
aft['Trans_for_norm'] = (aft['Weight Before'] - aft['Next Day Weight'] + water_added) / (2 * hours)
aft['hours'] = hours
trans_df = (aft.dropna(subset=['Trans_for_norm', 'hours']).loc[lambda d: d['hours'] > 0]
            .groupby(['Snapshot ID Tag', 'Date'], as_index=False)['Trans_for_norm'].mean())

# soil water: last 'Weight After' of the 17-20 h window
WA = num(f1720.get('Weight After')).combine(num(f1720.get('Weight After Re-filling')), lambda a, b: a if pd.notna(a) else b)
gwc = (f1720.assign(Gross17_20=WA).groupby(['Snapshot ID Tag', 'Date'], as_index=False)['Gross17_20'].last()
       .merge(soil_small, on='Snapshot ID Tag', how='left'))
tare = num(gwc['Weight of empty boxes']).fillna(0) + num(gwc['Gravel and Frame']).fillna(0)
dry = num(gwc['New Mass of Dry Soil'])
with np.errstate(divide='ignore', invalid='ignore'):
    gwc['SWC_g'] = np.where(dry > 0, (num(gwc['Gross17_20']) - tare - dry) / dry, np.nan)
gwc.loc[(gwc['SWC_g'] < 0) & (gwc['SWC_g'] > -0.01), 'SWC_g'] = 0.0      # scale noise


# ========= 4. Merge =========
merged = (trans_df.merge(gwc[['Snapshot ID Tag', 'Date', 'SWC_g']], on=['Snapshot ID Tag', 'Date'], how='inner')
          .merge(area_filled, on=['Snapshot ID Tag', 'Date'], how='inner')
          .merge(meta_map, on='Snapshot ID Tag', how='left'))
merged['NormTrans'] = merged['Trans_for_norm'] / merged['Area_filled']
merged['GWC'] = merged['SWC_g']


# ========= 5. HYPROP PDI curves: SWC_v -> SMP_est [kPa], K(h) [cm/h], MFP [cm2/d] =========
def read_hyprop(fname):
    """Interpolators theta -> SMP (kPa), log SMP -> log K (cm/h), and theta -> MFP (cm2/d)."""
    path = os.path.join(Path2Hyprop, fname)
    ret = pd.read_excel(path, sheet_name='Fitting-Retention \u0398(pF)')
    con = pd.read_excel(path, sheet_name='Fitting-Conductivity K(pF)')
    ret.columns, con.columns = ['pF', 'theta_pct'], ['pF', 'log10K_cmd']
    ret, con = ret.sort_values('pF').reset_index(drop=True), con.sort_values('pF').reset_index(drop=True)
    theta = ret['theta_pct'].astype(float).values / 100.0                      # Vol% -> cm3/cm3
    h_ret = 10.0 ** ret['pF'].astype(float).values * CM_H2O_TO_KPA              # kPa
    psi = 10.0 ** con['pF'].astype(float).values * CM_H2O_TO_KPA
    k = 10.0 ** con['log10K_cmd'].astype(float).values / 24.0                   # cm/d -> cm/h
    ok = np.isfinite(theta) & np.isfinite(h_ret)
    f_smp = interp1d(theta[ok], h_ret[ok], bounds_error=False, fill_value='extrapolate')
    ok = np.isfinite(psi) & np.isfinite(k) & (psi > 0) & (k > 0)
    psi, k = psi[ok], k[ok]
    o = np.argsort(psi)
    psi, k = psi[o], k[o]
    f_k = interp1d(np.log10(psi), np.log10(k), kind='linear', bounds_error=False, fill_value=(np.log10(k[0]), np.log10(k[-1])))
    # MFP = integral of K dh from pF to PF_DRY (trapezoid, dh = ln10 * h * dpF)
    grid = np.linspace(PF_WET, PF_DRY, MFP_NPTS)
    f = 10.0 ** np.interp(grid, con['pF'].values, con['log10K_cmd'].values) * np.log(10.0) * 10.0 ** grid
    cum = np.concatenate([[0.0], np.cumsum(np.diff(grid) * (f[1:] + f[:-1]) / 2.0)])
    o = np.argsort(theta)
    th_s, pf_s = theta[o], ret['pF'].values[o]
    keep = np.concatenate([[True], np.diff(th_s) > 0])
    return dict(f_smp=f_smp, f_k=f_k, psi_range=(float(psi.min()), float(psi.max())), grid=grid, phi=cum[-1] - cum,
                th=th_s[keep], pf=pf_s[keep], th_min=float(theta.min()), th_max=float(theta.max()))

HYP = {s: read_hyprop(f) for s, f in HYPROP_FILES.items()}

def soil_scales(swc_v, h):
    """SWC_v -> (SMP_est, K(h), MFP); K clamped to the curve range, MFP NaN outside the retention curve."""
    smp = h['f_smp'](swc_v)
    kh, mfp = np.full_like(swc_v, np.nan), np.full(swc_v.shape, np.nan)
    good = np.isfinite(smp) & (smp > 0)
    if good.any():
        kh[good] = np.power(10.0, h['f_k'](np.log10(np.clip(smp[good], *h['psi_range']))))
    good = np.isfinite(swc_v) & (swc_v > h['th_min']) & (swc_v < h['th_max'])
    if good.any():
        phi = np.interp(np.interp(swc_v[good], h['th'], h['pf']), h['grid'], h['phi'])
        mfp[good] = np.where(phi > 0, phi, np.nan)
    return smp, kh, mfp

for c in ('SWC_v', 'SMP_est', 'K(h)', 'MFP'):
    merged[c] = np.nan
for s in merged['Soil'].dropna().unique():
    m = merged['Soil'].eq(s)
    swc_v = merged.loc[m, 'SWC_g'].values.astype(float) * (RHO_B['Sandy Loam'] if 'Sandy' in s else RHO_B['Silt Loam'])
    merged.loc[m, 'SWC_v'] = swc_v
    if s in HYP:
        merged.loc[m, 'SMP_est'], merged.loc[m, 'K(h)'], merged.loc[m, 'MFP'] = soil_scales(swc_v, HYP[s])

print(f"[code1] merged: {len(merged)} rows | K(h) {merged['K(h)'].notna().sum()} | MFP {merged['MFP'].notna().sum()}")