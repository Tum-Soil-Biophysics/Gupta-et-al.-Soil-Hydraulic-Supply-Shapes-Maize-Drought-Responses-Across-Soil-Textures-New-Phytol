# Soil hydraulic supply shapes maize drought responses across soil textures: data and code

*New Phytologist, 2026*

Data, code and Supporting Information for the article

> Gupta S, Wild AJ, Heid A, Thiel J, Humpert J, Visconti A, Holz M, Wiesmeier M, Lueders T, Geyer M, Groth J, Pausch J, Hafner BD, Zarebanadkouki M (2026). Soil Hydraulic Supply Shapes Maize Drought Responses Across Soil Textures. *New Phytologist*.

**Related article:** [DOI] *(added at publication)*
**Data and code DOI (Zenodo):** https://doi.org/10.5281/zenodo.24460474 (all versions, always the latest)

The Supporting Information of the paper is also included (`Supporting Information.docx`).

## Repository structure

```
├── Drydown.py
├── main.py
├── compareretentioncurves.py
├── NormTrans.xlsx
├── Supporting Information.docx
├── Hyprob_data/
├── Input_main/
│   ├── Code1.py
│   ├── Code2.py
│   ├── Input_code1/
│   └── Input_code2/
│       ├── Cam1/
│       └── Cam2/
└── Output/
```

## Scripts

### 1. `Drydown.py`: transpiration thresholds (manuscript §3.1)

Reads: `NormTrans.xlsx`, `Hyprob_data/`

Does:

- For each mesocosm, fits a linear-plateau model of transpiration against θ. It finds θ<sub>crit</sub>, where transpiration starts to decline, and the θ at which relative transpiration (Tr/Trmax) falls to 0.35–0.45.
- Expresses these thresholds as θ, h, K(h) and Φ(h), using the HYPROP curves (Φ integrated to pF 4.2).
- Compares the two soils using |Cohen's d|, with 95% intervals from a cluster bootstrap over genotypes (8000 iterations).
- Tests the ordering of the variables with paired bootstrap differences and Wilcoxon signed-rank tests.
- Checks the sensitivity of Φ(h) to the integration method and grid resolution, and of the variable ordering to the HYPROP replicate fits.

Outputs: Fig. 3, Figs S4–S5, Tables S1–S3

### 2. `main.py`: drought, soil and genotype effects (manuscript §3.2–3.3)

Reads: the outputs of `Input_main/Code1.py` and `Input_main/Code2.py`, which it runs first, which prepare the plant and soil data per mesocosm (see 2.1 and 2.2).

Does:

- Fits two-way factorial ANOVAs (Y ~ Treatment × Genotype) for these comparisons:
  1. Drought effect: Opt<sub>Const</sub> vs Int<sub>Const</sub> vs Low<sub>Const</sub>, as pairwise models within each soil.
  2. Soil texture at matched θ class: silt loam vs sandy loam at Int<sub>Const</sub>, and at Low<sub>Const</sub>.
  3. Soil texture at comparable K(h) and Φ(h): silt loam Int<sub>Const</sub> vs sandy loam Low<sub>Const</sub>.
  4. Drought timing: Low<sub>Late</sub> vs Low<sub>Const</sub> within each soil.

Where model assumptions are not met, it transforms the data or uses robust standard errors (HC3). Genotype differences are shown as Holm-adjusted compact letter displays. For Fig. S9, it also regresses transpiration on K(h) and Φ(h) per genotype in silt loam, to correct for the different soil hydraulic supply of the two drought-timing treatments.

Outputs: Figs 4–5, Figs S6–S10, Tables S4–S9

#### 2.1 `Input_main/Code1.py`: daily plant data

Reads: `Input_main/Input_code1/`, `Hyprob_data/`

Does: for each mesocosm and day, `Code1.py` calculates:

- transpiration, from box weights
- shoot area, from RGB images
- normalised transpiration (transpiration ÷ shoot area)
- soil water content (θ), from box weights and dry soil mass, and from it h, K(h) and Φ(h) using the HYPROP curves

#### 2.2 `Input_main/Code2.py`: harvest data

Reads: `Input_main/Input_code2/Cam1/`, `Input_main/Input_code2/Cam2/`, `Hyprob_data/`

Does: for each mesocosm at harvest, `Code2.py` compiles or calculates:

- midday and predawn leaf water potential
- transpiration
- root traits (total root length, mean diameter, root-to-shoot ratio) and rhizosheath mass
- soil–plant hydraulic conductance (Ksp), from mid-day transpiration and the difference between leaf and soil water potential. The soil water potential comes from soil water content at harvest via the HYPROP curves. Transpiration is corrected for soil evaporation measured in boxes without plants.

## Input data

- `NormTrans.xlsx`: transpiration per shoot area (cm³ h⁻¹ cm⁻²) and gravimetric water content per mesocosm over the dry-down, with soil, genotype and replicate. Used by `Drydown.py`.
- `Hyprob_data/`: HYPROP measurements of water retention θ(h) and hydraulic conductivity K(h) on three replicate samples per soil (silt loam, sandy loam), each fitted with the PDI model. Φ(h) is derived by integrating K(h). These curves are shown in Fig. 2.
  - In Fig. S1, `compareretentioncurves.py` compares these HYPROP curves with retention curves from the TEROS 21 sensors in the mesocosms (Methods S5).
  - `Silt1.xlsx`, `Sandy2.xlsx`: one joint fit per soil, across all three replicate samples. Used by all scripts.
  - Replicate files: the separate fit of each replicate sample. Used by `Drydown.py` to test whether the results depend on which replicate's curve is used (Table S3).
- `Input_main/Input_code1/`: Excel files for `Code1.py`:
  1. shoot area from RGB images
  2. box weight records and soil-filling records, used to get transpiration and soil water content
- `Input_main/Input_code2/`: Excel files for `Code2.py`, one subfolder per campaign: `Cam1/` (campaign 1) and `Cam2/` (campaign 2).
  - Both campaigns contain:
    1. leaf water potentials (midday and predawn)
    2. soil water content
    3. matric potential from sensors
    4. transpiration
    5. LI-COR: stomatal conductance
  - `Cam2/` also contains:
    1. leaf area
    2. root morphology (total root length, mean diameter)
    3. rhizosheath mass

## How to run

1. Install Python and the required packages: numpy, pandas, scipy, statsmodels, patsy, matplotlib, seaborn, openpyxl (`pip install -r requirements.txt`).
2. No folder path needs to be set. Each script finds its data relative to its own location.
3. Run from the repository folder:

```bash
python Drydown.py
python main.py
python compareretentioncurves.py     # Fig. S1
```

No need to run `Code1.py` and `Code2.py` separately, because `main.py` runs them.

## Authors

Saniv Gupta¹, Andreas J. Wild², Alica Heid³, Jessica Thiel⁴, Jonas Humpert¹, Anaclara Visconti²˒⁵, Maire Holz⁵, Martin Wiesmeier⁴, Tillmann Lueders³, Manuel Geyer⁶, Jennifer Groth⁶, Johanna Pausch², Benjamin D. Hafner¹, Mohsen Zarebanadkouki¹

1. Chair of Soil Biophysics and Environmental Systems, Technical University of Munich, Freising, Germany
2. Agroecology, Bayreuth Center of Ecology and Environmental Research (BayCEER), University of Bayreuth, Bayreuth, Germany
3. Ecological Microbiology, Bayreuth Center of Ecology and Environmental Research (BayCEER), University of Bayreuth, Bayreuth, Germany
4. Institute for Agroecology and Organic Farming, Bavarian State Research Center for Agriculture (LfL), Freising, Germany
5. Root-Soil Interactions, Leibniz Centre for Agricultural Landscape Research (ZALF), Müncheberg, Germany
6. Institute for Crop Science and Plant Breeding, Bavarian State Research Center for Agriculture (LfL), Freising, Germany

Contact: saniv.gupta@tum.de, mohsen.zare@tum.de

## Funding

German Federal Ministry of Education and Research (BMBF), RhizoWater subproject of RhizoTraits (project number 031B0908).

## License

MIT, see [LICENSE](LICENSE). If you use these data or this code, please cite the related article and this repository (see [CITATION.cff](CITATION.cff)).
