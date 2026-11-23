
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
plt.close('all')

# ========= Settings =========
BASE = os.path.dirname(os.path.abspath(__file__))        # folder of this script (Input_main)
Path2MainDir = os.path.join(BASE, "Input_code2")      # Cam1/ and Cam2/; combined_df.xlsx saved here
CAMPAIGN_DIRS = {'Campaign1': os.path.join(Path2MainDir, "Cam1"), 'Campaign2': os.path.join(Path2MainDir, "Cam2")}
# HYPROP files: Input_main\Hyprob_data if it exists, else Data&code\Hyprob_data
Path2Hyprop = next(d for d in (os.path.join(BASE, "Hyprob_data"), os.path.join(os.path.dirname(BASE), "Hyprob_data"), BASE)
                   if os.path.isdir(d))
HYPROP_FILES = {'Sandy Loam': 'Sandy2.xlsx', 'Silt Loam': 'Silt1.xlsx'}
CM_H2O_TO_KPA = 0.0980665
PF_DRY, PF_WET, MFP_NPTS = 6.8, -3.0, 400_001          # MFP integration grid (uniform in pF)
RHO_B = {'Sandy Loam': 1.28, 'Silt Loam': 1.14}        # bulk density g/cm3
OutputFolder = Path2MainDir                             # combined_df.xlsx
os.makedirs(OutputFolder, exist_ok=True)


# ========= 1. HYPROP PDI curves =========
def read_hyprop(fname):
    """Curves (h [kPa], theta [cm3/cm3], K [cm/h]) + MFP grid (cm2/d) and theta -> pF inversion."""
    path = os.path.join(Path2Hyprop, fname)
    ret = pd.read_excel(path, sheet_name='Fitting-Retention Θ(pF)')
    con = pd.read_excel(path, sheet_name='Fitting-Conductivity K(pF)')
    ret.columns, con.columns = ['pF', 'theta_pct'], ['pF', 'log10K_cmd']
    ret, con = ret.sort_values('pF').reset_index(drop=True), con.sort_values('pF').reset_index(drop=True)
    theta = ret.theta_pct.astype(float).values / 100.0
    curves = pd.concat([pd.Series(10.0 ** ret.pF.astype(float).values * CM_H2O_TO_KPA, name='h_Theta'),
                        pd.Series(theta, name='Theta'),
                        pd.Series(10.0 ** con.pF.astype(float).values * CM_H2O_TO_KPA, name='h_K'),
                        pd.Series(10.0 ** con.log10K_cmd.astype(float).values / 24.0, name='K')], axis=1)
    grid = np.linspace(PF_WET, PF_DRY, MFP_NPTS)
    f = 10.0 ** np.interp(grid, con.pF.values, con.log10K_cmd.values) * np.log(10.0) * 10.0 ** grid   # dh = ln10*h*dpF
    cum = np.concatenate([[0.0], np.cumsum(np.diff(grid) * (f[1:] + f[:-1]) / 2.0)])
    o = np.argsort(theta)
    th_s, pf_s = theta[o], ret.pF.values[o]
    keep = np.concatenate([[True], np.diff(th_s) > 0])
    return curves, dict(pF_grid=grid, phi_grid=cum[-1] - cum, theta_sorted=th_s[keep], pf_of_theta=pf_s[keep],
                        theta_min=float(theta.min()), theta_max=float(theta.max()))

HYP = {s: read_hyprop(f) for s, f in HYPROP_FILES.items()}
HydrProp_Sand, HydrProp_Silt = HYP['Sandy Loam'][0], HYP['Silt Loam'][0]


# ========= 2. Measurements per campaign (merged on Snapshot ID Tag) =========
def read(campaign, fname, **kw):
    return pd.read_excel(os.path.join(CAMPAIGN_DIRS[campaign], fname), **kw)

def Combine_Readings(campaign):
    mid = read(campaign, 'hLeaf_Midday.xlsx').rename(columns={'Leaf water potential - mid-day': 'hXylem_Ψx'})
    mid['hXylem_Ψx'] = mid['hXylem_Ψx'] * 100                                 # bar -> kPa
    pre = read(campaign, 'hLeaf_Predawn.xlsx').rename(columns={'Leaf water potential - pre-dawn': 'hLeaf_Predawn'})
    pre['hLeaf_Predawn'] = pre['hLeaf_Predawn'] * 100
    smp = read(campaign, 'SWC_SMP.xlsx', sheet_name='soil water potential').rename(columns={'Soil water potential': 'SMP_teros21'})
    smp['SMP_teros21'] = smp['SMP_teros21'] * -1
    swc = read(campaign, 'SWC_SMP.xlsx', sheet_name='Soil water content').rename(columns={'TDR': 'SWC_tdr'})
    swc['SWC_tdr'] = swc['SWC_tdr'] * 0.01                                     # % -> cm3/cm3
    licor = read(campaign, 'LiCOR.xlsx').rename(columns={'Transpiration': 'Trans_licor', 'Stomatal Conductance': 'gs'})
    licor.loc[licor['Trans_licor'] < 0, 'Trans_licor'] = np.nan
    licor.loc[licor['gs'] < 0, 'gs'] = np.nan
    tables = [pre, mid, smp, licor]
    if campaign == 'Campaign2':
        la = read(campaign, 'LeafArea.xlsx')
        la = la[(la['No. Of leaves'] != 'Non box') & (la['No. Of leaves'] != 'Tst')]
        la = la.rename(columns={'leaf area': 'LeafArea'})[['Snapshot ID Tag', 'LeafArea', 'height (cm)']]
        gwc = read(campaign, 'GWC_Weights.xlsx').rename(columns={'GWC': 'GWC_OvenWeights'})
        gwc['GWC_OvenWeights'] = gwc['GWC_OvenWeights'] * 0.01
        tables += [la, gwc, read(campaign, 'Root_to_Shoot.xlsx'), read(campaign, 'Root_morphology_rhizosheath_updated_DM.xlsx')]
    df = swc
    for t in tables:
        df = pd.merge(df, t, on='Snapshot ID Tag', how='outer', indicator=False)
    df = df.drop(columns=['ID', 'Drought Treatment', 'Soil Texture', 'Replicate Number', 'Order', 'Exp', 'Treat.Soil.Name',
                          'Treat.H2O.Name', 'Treat.Soil.Code', 'Geno.Code', 'Treat.N.Name', 'Treat.N.Code', 'Sensor',
                          'File.Name', 'Treat.H2O.Code', 'Geno.Name'], errors='ignore')
    df = df.rename(columns={'Snapshot ID Tag': 'ID'})
    df['Cam'] = 'Cam1' if campaign == 'Campaign1' else 'Cam2'
    return df.reset_index(drop=True)

combined_df = pd.concat([Combine_Readings('Campaign1'), Combine_Readings('Campaign2')], ignore_index=True, sort=False)


# ========= 3. Dry soil per box, transpiration and soil water from weights =========
def process_soil_data(path):
    d = pd.read_excel(path, sheet_name='Sheet1')
    d['Mass of Dry Soil'] = d['Weight of Wet Soil'] / ((d['Water Content (%)'] * 0.01) + 1)
    d['Soil Added/Removed'] = d['Weight After Re-filling'] - d['Weight Before Re-filling']
    d['Updated Soil Added/Removed'] = d['Soil Added/Removed']
    for i, row in d.iterrows():                    # removed soil: subtract its water content
        if row['Soil Added/Removed'] < 0 and row['Soil'].lower() in ('silt', 'sandy'):
            wc = 3.72 if row['Soil'].lower() == 'silt' else 1.59
            d.at[i, 'Updated Soil Added/Removed'] = row['Soil Added/Removed'] / ((wc * 0.01) + 1)
    d['FinalDrySoilMass'] = d['Mass of Dry Soil'] + d['Updated Soil Added/Removed']
    d['PotPulsRest'] = d['Weight of empty boxes'] + d['Gravel and Frame']
    d = d[['Snapshot ID Tag', 'Soil', 'PotPulsRest', 'FinalDrySoilMass']].rename(columns={'Snapshot ID Tag': 'ID'})
    d.loc[d['Soil'] == 'Silt', 'Soil'] = 'Silt Loam'
    d.loc[d['Soil'] == 'Sandy', 'Soil'] = 'Sandy Loam'
    return d

soil_data2 = process_soil_data(os.path.join(CAMPAIGN_DIRS['Campaign2'], 'Weights_DrySoil_Rhizobox.xlsx'))

def in_window(ts, day, hours):
    """Nov `day`; hours = list of (hour, max minute)."""
    m = (ts.dt.month == 11) & (ts.dt.day == day)
    return m & np.logical_or.reduce([(ts.dt.hour == h) & (ts.dt.minute <= mm) for h, mm in hours])

def calculate_daily_transpiration_avg(weight_file, soil_data2, days=(25, 27, 28, 29)):
    """Avg_Daily_Transpiration (median of 17-20 h day-to-day losses x 24), Trans_Midday (Nov 28, 10-11:32 vs 12-13:25)
    and SWC_g / SWC_v (last weight Nov 28, 12-14 h). All transpiration values halved ('/2' convention)."""
    w = pd.read_excel(weight_file).rename(columns={'Snapshot ID Tag': 'ID'})
    w['Snapshot Time Stamp'] = pd.to_datetime(w['Snapshot Time Stamp'], dayfirst=False, errors='coerce')
    w = w[w['ID'].isin(soil_data2['ID'].unique())]
    ts = w['Snapshot Time Stamp']
    daily = {}
    for day in days:
        t0, t1 = pd.Timestamp(year=2024, month=11, day=day, hour=17), pd.Timestamp(year=2024, month=11, day=day + 1, hour=17)
        today = w[(ts >= t0) & (ts < t0 + pd.Timedelta(hours=3))].groupby('ID')
        nxt = w[(ts >= t1) & (ts < t1 + pd.Timedelta(hours=3))].groupby('ID')
        wb_today, wb_next = today['Weight Before'].first(), nxt['Weight Before'].first()
        hours = (nxt['Snapshot Time Stamp'].first() - today['Snapshot Time Stamp'].first()).dt.total_seconds() / 3600
        added = (today['Weight After'].last() - wb_today).clip(lower=0)
        daily[f'Transpiration_{day}'] = (wb_today - wb_next + added) / hours / 2
    daily = pd.DataFrame(daily).reset_index().rename(columns={'index': 'ID'})
    daily['Avg_Daily_Transpiration'] = daily[[f'Transpiration_{d}' for d in days]].median(axis=1) * 24

    morning = w[in_window(ts, 28, [(10, 59), (11, 32)])].groupby('ID')
    noon = w[in_window(ts, 28, [(12, 59), (13, 25)])].groupby('ID')
    gap = (noon['Snapshot Time Stamp'].min() - morning['Snapshot Time Stamp'].min()).dt.total_seconds() / 3600
    midday = pd.concat([morning['Weight After'].mean().rename("Morning Weight"), noon['Weight After'].mean().rename("Afternoon Weight"),
                        gap.rename("Time Gap (hours)")], axis=1)
    midday['Trans_Midday'] = (midday['Morning Weight'] - midday['Afternoon Weight']) / midday['Time Gap (hours)'] / 2

    win = w[(ts.dt.month == 11) & (ts.dt.day == 28) & (ts.dt.hour >= 12) & (ts.dt.hour < 14)]
    last_weight = win.loc[win.groupby('ID')['Snapshot Time Stamp'].idxmax()].set_index('ID')['Weight After']
    s = soil_data2.set_index('ID')
    s['SWC_g'] = (last_weight - s['PotPulsRest'] - s['FinalDrySoilMass']) / s['FinalDrySoilMass']
    s['SWC_v'] = s['SWC_g']
    for soil, rho in (('Sandy Loam', RHO_B['Sandy Loam']), ('Silt Loam', RHO_B['Silt Loam'])):
        s.loc[s['Soil'] == soil, 'SWC_v'] *= rho
    return (daily.set_index('ID').join(midday[['Trans_Midday']], how='outer')
            .join(s[['SWC_g', 'SWC_v']], how='outer').reset_index())

DataCam = pd.concat([calculate_daily_transpiration_avg(os.path.join(CAMPAIGN_DIRS[c], 'WeightRecords.xlsx'), soil_data2).assign(Cam=cam)
                     for c, cam in (('Campaign1', 'Cam1'), ('Campaign2', 'Cam2'))], ignore_index=True, sort=False)
combined_df = pd.merge(combined_df, DataCam, on=['ID', 'Cam'], how='outer', indicator=False)


# ========= 4. Treatment labels from the ID (digit 4 genotype, 5 soil, second-last replicate, last drought) =========
GENO = {'1': 'Benedictio KWS', '2': 'Luigi CS', '3': 'P8888', '4': 'Alpenfex', '5': 'Braunes Schindelmeiser', '6': 'Pfarrkirchner'}
ORIGIN = {'1': 'Hybrid', '2': 'Hybrid', '3': 'Hybrid', '4': 'Landrace', '5': 'Landrace', '6': 'Landrace'}
DROUGHT = {'1': 'Optimum', '2': 'Moderate Drought', '3': 'Severe Drought', '4': 'Rewetting Cycle', '5': 'Drying Cycle'}

ids = combined_df['ID'].astype(str).str.strip()
drop = ((ids.str.len().ge(4) & (ids.str[3] == '9')) | (ids.str.len().ge(6) & (ids.str[5] == '1'))
        | (ids.str.len().ge(2) & (ids.str.slice(-2, -1) == '6')))
combined_df = combined_df[~drop]
sid = combined_df['ID'].astype(str)
combined_df['DroughtLevel'] = sid.str[-1].map(DROUGHT).fillna('Unknown')
combined_df['Genotype'] = sid.str[3].map(GENO).fillna('Blank')          # other digits = blank box (no plant)
combined_df['Origin'] = sid.str[3].map(ORIGIN).fillna('Blank')
combined_df['Soil'] = sid.str[4].map({'1': 'Silt Loam', '2': 'Sandy Loam'}).fillna('Unknown')
combined_df['Rep'] = "R" + sid.str[-2]


# ========= 5. SMP_RETC [kPa], K_h_RETC [cm/h], MFP_RETC [cm2/d] from SWC_v =========
combined_df['SMP_RETC'] = np.nan
for soil, (cur, _) in HYP.items():
    m = combined_df['Soil'] == soil
    combined_df.loc[m, 'SMP_RETC'] = interp1d(cur['Theta'].values, cur['h_Theta'].values, bounds_error=False,
                                              fill_value='extrapolate')(combined_df.loc[m, 'SWC_v'].values)

def k_from_smp(psi, cur):
    """log-log interpolation of the K(h) curve, clamped to its range."""
    p, k = cur['h_K'].values, cur['K'].values
    ok = np.isfinite(p) & np.isfinite(k) & (p > 0) & (k > 0)
    o = np.argsort(p[ok])
    p, k = p[ok][o], k[ok][o]
    f = interp1d(np.log10(p), np.log10(k), kind='linear', bounds_error=False, fill_value=(np.log10(k[0]), np.log10(k[-1])))
    psi = np.asarray(psi, dtype=float)
    out = np.full_like(psi, np.nan, dtype=float)
    good = np.isfinite(psi) & (psi > 0)
    if good.any():
        out[good] = np.power(10.0, f(np.log10(np.clip(psi[good], float(p.min()), float(p.max())))))
    return out

def mfp_from_swc(theta, p):
    """theta -> pF -> MFP; NaN outside the retention curve."""
    theta = np.asarray(theta, dtype=float)
    out = np.full(theta.shape, np.nan, dtype=float)
    good = np.isfinite(theta) & (theta > p['theta_min']) & (theta < p['theta_max'])
    if good.any():
        phi = np.interp(np.interp(theta[good], p['theta_sorted'], p['pf_of_theta']), p['pF_grid'], p['phi_grid'])
        out[good] = np.where(phi > 0, phi, np.nan)
    return out

combined_df['K_h_RETC'] = np.nan
combined_df['MFP_RETC'] = np.nan
for soil, (cur, p) in HYP.items():
    m = combined_df['Soil'] == soil
    if m.any():
        combined_df.loc[m, 'K_h_RETC'] = k_from_smp(combined_df.loc[m, 'SMP_RETC'].values, cur)
        combined_df.loc[m, 'MFP_RETC'] = mfp_from_swc(combined_df.loc[m, 'SWC_v'].values, p)


# ========= 6. Subtract soil evaporation measured in blank boxes =========
def blank_correction(df, col):
    """Mean of blank boxes per drought x soil x campaign; Rewetting uses Optimum, Drying uses Severe Drought."""
    ref = (df[df["Genotype"] == "Blank"].groupby(["DroughtLevel", "Soil", "Cam"])[col].mean().reset_index()
           .rename(columns={col: "ref"}))
    extra = []
    for target, source in (('Rewetting Cycle', 'Optimum'), ('Drying Cycle', 'Severe Drought')):
        for soil in ['Silt Loam', 'Sandy Loam']:
            for cam in ['Cam1', 'Cam2']:
                sel = ref[(ref['Soil'] == soil) & (ref['DroughtLevel'] == source) & (ref['Cam'] == cam)]['ref']
                extra.append(pd.DataFrame({'DroughtLevel': [target], 'Soil': [soil], 'Cam': [cam],
                                           'ref': [sel.iloc[0] if not sel.empty else np.nan]}))
    df = df.merge(pd.concat([ref] + extra, ignore_index=True, sort=False), on=["DroughtLevel", "Soil", "Cam"], how="left")
    df[col] = df[col] - df["ref"].fillna(0)
    return df.drop(columns=["ref"])

combined_df = blank_correction(combined_df, "Trans_Midday")
combined_df = blank_correction(combined_df, "Avg_Daily_Transpiration")
combined_df = combined_df[combined_df["Genotype"] != "Blank"].reset_index(drop=True)


# ========= 7. Check plots (shown, not saved) =========
sa, si = combined_df['Soil'] == 'Sandy Loam', combined_df['Soil'] == 'Silt Loam'
plt.figure('RTEC')
plt.semilogy(combined_df.loc[sa]['SWC_tdr'], combined_df.loc[sa]['SMP_teros21'], 'o', color='r', label='SandyLoam_tdr&teros21')
plt.semilogy(combined_df.loc[si]['SWC_tdr'], combined_df.loc[si]['SMP_teros21'], 'o', color='b', label='SiltyLoam_tdr&teros21')
plt.semilogy(combined_df.loc[sa]['SWC_v'], combined_df.loc[sa]['SMP_teros21'], '*', color='r', label='SandyLoam_weights&teros21')
plt.semilogy(combined_df.loc[si]['SWC_v'], combined_df.loc[si]['SMP_teros21'], '*', color='b', label='SiltyLoam_weights&teros21')
plt.semilogy(HydrProp_Sand['Theta'], HydrProp_Sand['h_Theta'], '-', color='r', label='SandyLoam_hyprop')
plt.semilogy(HydrProp_Silt['Theta'], HydrProp_Silt['h_Theta'], '-', color='b', label='SiltyLoam_hyprop')
plt.xlabel('Theta_m'); plt.ylabel('SMP'); plt.legend()

# LiCor transpiration mmol m-2 s-1 -> g/h (x leaf area)
combined_df['Trans_licor'] = combined_df['Trans_licor'] * combined_df['LeafArea'] * 0.018 * 10**-4 * 3600

def scatter(x, y, xlabel, ylabel, line=None, **kw):
    plt.figure()
    sns.scatterplot(data=combined_df, x=x, y=y, hue="DroughtLevel", s=140, edgecolor='black', linewidth=1.5, **kw)
    if line:
        plt.plot(combined_df[line].unique(), combined_df[line].unique(), color='k', label='1:1 line')
    plt.xlabel(xlabel); plt.ylabel(ylabel)

scatter("Avg_Daily_Transpiration", 'Trans_licor', 'Transpiraiton_weight [g/h]', 'Transpiraiton_licor [g/h]', 'Avg_Daily_Transpiration')
scatter("GWC_OvenWeights", 'SWC_g', 'GWC_OvenWeights', 'SWC_g', 'GWC_OvenWeights')
scatter("SWC_tdr", 'SWC_v', 'SWC_tdr', 'SWC_v', 'SWC_tdr')
scatter("Avg_Daily_Transpiration", 'LeafArea', 'Transpiraiton [g/h]', 'Leaf Area [cm^2]')
scatter("SMP_teros21", 'hLeaf_Predawn', 'SMP_teros21 [kPa]', 'hLeaf_predawn [kPa]', 'hLeaf_Predawn')
scatter("SMP_RETC", 'hLeaf_Predawn', 'SMP_RETC[kPa]', 'hLeaf_predawn [kPa]', 'hLeaf_Predawn', style='Cam')


# ========= 8. Ksp = Trans_Midday / (Psi_x - Psi_predawn); missing Cam2 predawn filled from Cam1 =========
def ksp(trans, psi_x, predawn):
    d = psi_x - predawn
    d[d <= 0] = np.nan
    k = trans / d
    k[k < 0] = np.nan
    return k.replace([np.inf, -np.inf], np.nan)

c = combined_df
c['Ksp'] = c['Trans_Midday'] / (c['hXylem_Ψx'] - c['hLeaf_Predawn'])
c.loc[c['Ksp'] < 0, 'Ksp'] = np.nan
c['Ksp'] = c['Ksp'].replace([np.inf, -np.inf], np.nan)
cam1 = c[(c['Cam'] == 'Cam1') & c['hLeaf_Predawn'].notna()]
c = c.merge(cam1[['Genotype', 'Soil', 'DroughtLevel', 'Rep', 'hLeaf_Predawn']].rename(columns={'hLeaf_Predawn': 'hLP_cam1_rep'}),
            on=['Genotype', 'Soil', 'DroughtLevel', 'Rep'], how='left')
cam1 = c[(c['Cam'] == 'Cam1') & c['hLeaf_Predawn'].notna()]          # (after the merge above, as in the original)
c = c.merge(cam1.groupby(['Genotype', 'Soil', 'DroughtLevel'])['hLeaf_Predawn'].mean().reset_index()
            .rename(columns={'hLeaf_Predawn': 'hLP_cam1_group'}), on=['Genotype', 'Soil', 'DroughtLevel'], how='left')
cam2 = c['Cam'] == 'Cam2'
use_rep = cam2 & c['Ksp'].isna() & c['hXylem_Ψx'].notna() & c['hLP_cam1_rep'].notna()       # 1: same replicate in Cam1
c.loc[use_rep, 'Ksp'] = ksp(c.loc[use_rep, 'Trans_Midday'], c.loc[use_rep, 'hXylem_Ψx'], c.loc[use_rep, 'hLP_cam1_rep'])
use_group = cam2 & c['Ksp'].isna() & c['hXylem_Ψx'].notna() & c['hLP_cam1_group'].notna()  # 2: Cam1 group mean
c.loc[use_group, 'Ksp'] = ksp(c.loc[use_group, 'Trans_Midday'], c.loc[use_group, 'hXylem_Ψx'], c.loc[use_group, 'hLP_cam1_group'])
c['Ksp_flag'] = np.nan
c.loc[cam2 & c['Ksp'].notna() & ~use_rep & ~use_group, 'Ksp_flag'] = 'measured_cam2'
c.loc[use_rep & c['Ksp'].notna(), 'Ksp_flag'] = 'filled_cam1_same_rep'
c.loc[use_group & c['Ksp'].notna(), 'Ksp_flag'] = 'filled_cam1_group_mean'
c.loc[cam2 & c['Ksp'].isna() & c['hXylem_Ψx'].isna(), 'Ksp_flag'] = 'missing_Psi_x'
c.loc[cam2 & c['Ksp'].isna() & c['hXylem_Ψx'].notna(), 'Ksp_flag'] = 'unfixable_sign_error'
c = c.drop(columns=['hLP_cam1_rep', 'hLP_cam1_group'])
c['Ksp_soil'] = c['Trans_Midday'] / (c['hXylem_Ψx'] - c['SMP_RETC'])
c.loc[c['Ksp_soil'] < 0, 'Ksp_soil'] = np.nan
c['Ksp_soil'] = c['Ksp_soil'].replace([np.inf, -np.inf], np.nan)
print('Ksp_flag breakdown (Cam2 only):')
print(c.loc[c['Cam'] == 'Cam2', 'Ksp_flag'].value_counts())


# ========= 9. Normalised transpiration and derived metrics =========
c['E_Norm'] = c['Trans_Midday'] / c['LeafArea']
c['E_Norm2'] = c['Avg_Daily_Transpiration'] / c['LeafArea']
opt = (c[c["DroughtLevel"] == "Optimum"].groupby(["Genotype", "Soil", 'Cam'])["Trans_Midday"].mean().reset_index()
       .rename(columns={"Trans_Midday": "ref"}))
c = c.merge(opt, on=["Genotype", "Soil", 'Cam'], how="left")
c["Trans_Norm_MaxTr"] = c["Trans_Midday"] / c["ref"]
c = c.drop(columns=["ref"])
c['hXylem_Ψx_div_Avg_Daily_Transpiration'] = c['hXylem_Ψx'] / c['Avg_Daily_Transpiration']
c['hXylem_Ψx_div_Trans_Norm_LA'] = c['hXylem_Ψx'] / c['E_Norm']
c['hXylem_Ψx_div_Trans_Norm_MaxTr'] = c['hXylem_Ψx'] / c['Trans_Norm_MaxTr']
c['hLeaf_Diff'] = c['hXylem_Ψx'] - c['SMP_RETC']
for col in c.columns:                                   # every non-text column numeric
    if c[col].dtype != 'O':
        c[col] = pd.to_numeric(c[col], errors='coerce')
combined_df = c

combined_df.to_excel(os.path.join(OutputFolder, "combined_df.xlsx"), index=False)
print(f"[code2] combined_df: {len(combined_df)} rows -> {os.path.join(OutputFolder, 'combined_df.xlsx')}")