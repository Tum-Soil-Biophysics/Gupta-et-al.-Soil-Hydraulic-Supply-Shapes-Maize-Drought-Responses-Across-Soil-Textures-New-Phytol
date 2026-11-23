# Gupta et al. (2026): Soil Hydraulic Supply Shapes Maize Drought Responses Across Soil Textures

Data, code and Supporting Information for the article

> Gupta S, Wild AJ, Heid A, Thiel J, Humpert J, Visconti A, Holz M, Wiesmeier M, Lueders T, Geyer M, Groth J, Pausch J, Hafner BD, Zarebanadkouki M (2026). Soil Hydraulic Supply Shapes Maize Drought Responses Across Soil Textures. *New Phytologist*.

**Related article:** [DOI] *(added at publication)*
**Data and code DOI (Zenodo):** see the latest [release](../../releases) *(created automatically with each release)*

## Summary

Six maize (*Zea mays* L.) genotypes were grown in silt loam and sandy loam, with drought imposed from establishment or after shoot development. The study tests whether the soil hydraulic supply variables, unsaturated hydraulic conductivity K(h) and matric flux potential Φ(h), give a more consistent basis for comparing drought responses across soil textures than soil water content (θ) or matric potential (h). The scripts in this repository derive the transpiration downregulation thresholds on all four scales, compare above- and belowground traits across drought treatments, soils and drought timing, and write the statistics, figures and tables of the article.

## Repository layout

```
Drydown.py                  thresholds of transpiration downregulation on the theta, h, K(h) and Phi(h) scales
main.py                     trait responses to drought, soil texture and drought timing (runs Code1.py and Code2.py first)
NormTrans.xlsx              normalized transpiration during soil drying (input of Drydown.py)
Hyprob_data/                soil hydraulic properties from HYPROP measurements
  Rep1..3_sandy/silt.xlsx     replicate fits per soil
  Sandy2.xlsx, Silt1.xlsx     joint fits per soil (used for all conversions)
Input_main/
  Code1.py                  shoot area, pot weights and transpiration time series
  Code2.py                  campaign data: leaf water potential, gas exchange, soil water, root traits
  Input_code1/              input files of Code1.py
  Input_code2/Cam1, Cam2/   input files of Code2.py (campaign 1 and campaign 2)
  Input_code2/combined_df.xlsx   merged campaign table written by Code2.py
Output/                     tables written by the scripts (figures are written here when the scripts run)
  Drydown/                    threshold tables, fits and statistics summary
  main/FinalResults/          processed data tables of Code1.py and Code2.py
Supporting_Information/     Supporting Information of the article
```

No script contains a fixed computer path. All folders are found relative to the repository.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Tum-Soil-Biophysics/Gupta-et-al.-Soil-Hydraulic-Supply-Shapes-Maize-Drought-Responses-Across-Soil-Textures-New-Phytol.git
cd Gupta-et-al.-Soil-Hydraulic-Supply-Shapes-Maize-Drought-Responses-Across-Soil-Textures-New-Phytol
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Workflow

Run the scripts from the repository folder. The two scripts are independent.

| Script | Input | Output |
|---|---|---|
| `Drydown.py` | `Hyprob_data/`, `NormTrans.xlsx` | `Output/Drydown/`: soil separation and collapse figures, threshold tables, sensitivity tables for the Φ(h) integration, `stats_summary.txt` |
| `main.py` | `Input_main/` (runs `Code1.py` and `Code2.py`), `Hyprob_data/` | `Output/main/`: processed data (`FinalResults/`), forest plots and ANOVA tables for aboveground (`Plot5by2give/`) and belowground traits (`5by2_below/`), soil effect at comparable hydraulic supply (`Soil_effect/`), early versus late drought (`M5vsM3/`) |

```bash
python Drydown.py
python main.py
```

`main.py` takes several minutes. File names of tables and figures written by the scripts follow the working numbering of the analysis and can differ from the final numbering in the article and its Supporting Information.

## Data

Sample identity is coded in the column `Snapshot ID Tag` (genotype, soil, replicate and drought treatment; decoded in `Input_main/Code1.py`). Genotypes: Benedictio KWS, Luigi CS, P8888, Alpenfex, Braunes Schindelmeiser, Pfarrkirchner. Soils: silt loam and sandy loam (bulk density 1.14 and 1.28 g cm⁻³).

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
