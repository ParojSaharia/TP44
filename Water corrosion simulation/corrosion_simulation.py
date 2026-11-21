"""
BRINE WATER-QUALITY ANALYSIS & THEORETICAL CORROSION SIMULATION
Application: simplified petroleum water-injection / waterflooding study

Samples
  S0 : Distilled water reference (measured)
  S1 : NaCl     1000 ppm
  S2 : KCl      1000 ppm
  S3 : Na2SO4   1000 ppm

IMPORTANT SCIENTIFIC NOTES
  1. Salt-solution pH = 7.0 is an ASSUMED value, not a measurement.
  2. Conductivity / salinity / TDS are transcribed from lab notes.
     Confirm the units against the instrument records.
  3. This is a scenario model, NOT a validated corrosion predictor.
     Water-quality data cannot determine corrosion current density.
  4. Corrosion rates come from ASSUMED current densities (Faraday's law).
  5. Temperature effect uses an ILLUSTRATIVE Arrhenius factor with an
     ASSUMED activation energy.
  6. Not modelled: multiphase flow, CO2/H2S, oxygen transport, deposits,
     MIC, inhibitors, flow-induced mass transfer, pitting.

HOW TO RUN
  VS Code / terminal :  python brine_corrosion_model.py
  Google Colab       :  paste into a cell and run (or upload and %run)
Outputs (CSV + PNG) are saved to the folder  ./outputs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER-EDITABLE INPUTS
# ============================================================
SHOW_PLOTS = True          # False -> only save PNGs (no pop-up windows)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Distilled water baseline (MEASURED). Check units on the meter. ---
BASELINE = {
    "sample": "Distilled water",
    "salt": "Reference",
    "concentration_ppm": 0.0,
    "pH": 7.87,
    "pH_status": "measured",
    "conductivity_uScm": 259.0,   # uS/cm
    "salinity_ppt": 0.17,         # ppt
    "tds_mgL": 148.0,             # mg/L
}

# --- 1000 ppm salt solutions. pH = 7 is ASSUMED. ---
SALT_SOLUTIONS = [
    {"sample": "NaCl 1000 ppm", "salt": "NaCl", "concentration_ppm": 1000.0,
     "pH": 7.0, "pH_status": "assumed",
     "conductivity_uScm": 1750.0,   # 1.75 mS/cm
     "salinity_ppt": 1.13,
     "tds_mgL": 920.0},             # 0.92 ppt
    {"sample": "KCl 1000 ppm", "salt": "KCl", "concentration_ppm": 1000.0,
     "pH": 7.0, "pH_status": "assumed",
     "conductivity_uScm": 1550.0,   # 1.55 mS/cm
     "salinity_ppt": 0.99,
     "tds_mgL": 810.0},             # 0.81 ppt
    {"sample": "Na2SO4 1000 ppm", "salt": "Na2SO4", "concentration_ppm": 1000.0,
     "pH": 7.0, "pH_status": "assumed",
     "conductivity_uScm": 1320.0,   # 1.32 mS/cm
     "salinity_ppt": 0.83,
     "tds_mgL": 700.0},             # 0.70 ppt
]

# --- Salt stoichiometry: molar mass (g/mol), cation/anion charge and count ---
SALT_CHEMISTRY = {
    #            M_salt  cation: (name, M, z, n)      anion: (name, M, z, n)
    "NaCl":    (58.44,  ("Na+", 22.99, 1, 1), ("Cl-", 35.45, 1, 1)),
    "KCl":     (74.55,  ("K+", 39.10, 1, 1),  ("Cl-", 35.45, 1, 1)),
    "Na2SO4":  (142.04, ("Na+", 22.99, 1, 2), ("SO4 2-", 96.06, 2, 1)),
}

# --- Corrosion model settings (ALL ASSUMED) ---
METAL_NAME = "Carbon steel (iron basis)"
IRON_EQUIVALENT_WEIGHT = 27.92      # g/equivalent (Fe -> Fe2+)
STEEL_DENSITY = 7.86                # g/cm3
ICORR_SCENARIOS = {"Low": 5.0, "Moderate": 10.0, "High": 20.0, "Very high": 50.0}
BASE_SCENARIO = "Moderate"          # scenario used for the temperature study
EXPOSURE_YEARS = [1, 2, 5, 10]

# --- Temperature (illustrative Arrhenius scaling) ---
REFERENCE_TEMPERATURE_C = 25.0
SIMULATION_TEMPERATURES_C = [20.0, 30.0, 40.0, 50.0, 60.0]
APPARENT_ACTIVATION_ENERGY_J_MOL = 40000.0   # ASSUMED
GAS_CONSTANT = 8.314462618          # J/(mol K)
FARADAY = 96485.0                   # C/mol

# Fe2+/Fe thermodynamics (for pH indicator)
E0_FE = -0.440                      # V vs SHE
FE2_ACTIVITY = 1e-6                 # mol/L (conventional Pourbaix value)


# ============================================================
# HELPERS
# ============================================================
def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def save_csv(frame, name):
    path = os.path.join(OUTPUT_DIR, name)
    frame.to_csv(path, index=False)
    print(f"   [saved] {path}")


def finish_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"   [saved] {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def corrosion_rate_mm_year(icorr_uA_cm2):
    """Faraday's law: CR = 0.00327 * Icorr * EW / density  (mm/yr)."""
    return (0.00327 * np.asarray(icorr_uA_cm2, dtype=float)
            * IRON_EQUIVALENT_WEIGHT / STEEL_DENSITY)


def nace_class(cr):
    if cr < 0.025:
        return "Low"
    if cr <= 0.12:
        return "Moderate"
    if cr <= 0.25:
        return "High"
    return "Severe"


def arrhenius_factor(temp_c):
    t_ref = REFERENCE_TEMPERATURE_C + 273.15
    t = np.asarray(temp_c, dtype=float) + 273.15
    return np.exp((APPARENT_ACTIVATION_ENERGY_J_MOL / GAS_CONSTANT) * (1.0 / t_ref - 1.0 / t))


def nernst_slope(temp_c):
    return 2.303 * GAS_CONSTANT * (temp_c + 273.15) / FARADAY


# ============================================================
# 2. BUILD DATASET
# ============================================================
df = pd.DataFrame([BASELINE] + SALT_SOLUTIONS)

banner("1. INPUT DATA (measured / transcribed / assumed)")
print(df.to_string(index=False))
print("\npH_status separates measured pH from ASSUMED pH.")
save_csv(df, "01_input_data.csv")

# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================
banner("2. DATA QUALITY CHECKS")
print("Missing values per column:")
print(df.isna().sum().to_string())

if (df["conductivity_uScm"] <= 0).any():
    raise ValueError("Conductivity must be positive.")
if (df["tds_mgL"] < 0).any() or (df["salinity_ppt"] < 0).any():
    raise ValueError("TDS and salinity cannot be negative.")

df["TDS_to_EC_ratio"] = df["tds_mgL"] / df["conductivity_uScm"]
print("\nTDS / EC ratio (natural waters are typically 0.5 - 0.7):")
print(df[["sample", "TDS_to_EC_ratio"]].round(3).to_string(index=False))
if ((df["TDS_to_EC_ratio"] < 0.4) | (df["TDS_to_EC_ratio"] > 0.8)).any():
    print("WARNING: a ratio is outside 0.4-0.8. Re-check the units entered.")
else:
    print("Ratios are consistent, so the units entered appear coherent.")

print("\nUnits assumed: conductivity uS/cm, salinity ppt, TDS mg/L.")
print("Verify these against your original instrument displays.")

# ============================================================
# 4. COMPARISON WITH DISTILLED WATER (incl. net salt contribution)
# ============================================================
ref = df.loc[df["sample"] == BASELINE["sample"]].iloc[0]

for col in ["conductivity_uScm", "salinity_ppt", "tds_mgL"]:
    df[col + "_change_pct_vs_ref"] = (df[col] - ref[col]) / ref[col] * 100.0
    df[col + "_net_of_water"] = df[col] - ref[col]     # salt contribution only
df["pH_difference_vs_ref"] = df["pH"] - ref["pH"]
df["resistivity_ohm_cm"] = 1e6 / df["conductivity_uScm"]

banner("3. COMPARISON WITH DISTILLED-WATER REFERENCE")
cols = ["sample", "pH", "pH_status", "conductivity_uScm",
        "conductivity_uScm_change_pct_vs_ref", "conductivity_uScm_net_of_water",
        "tds_mgL", "tds_mgL_change_pct_vs_ref", "resistivity_ohm_cm"]
print(df[cols].round(2).to_string(index=False))
print("\n'net_of_water' = reading minus distilled-water baseline, i.e. what the")
print("added salt contributes on its own.")
save_csv(df, "02_reference_comparison.csv")

brines = df[df["salt"] != "Reference"].copy()

# ============================================================
# 5. STOICHIOMETRIC ION CHEMISTRY (calculated, not measured)
# ============================================================
banner("4. ION CHEMISTRY FROM FORMULA (assuming 1000 ppm = 1000 mg/L of salt)")
ion_rows = []
for _, r in brines.iterrows():
    m_salt, (cat, m_cat, z_cat, n_cat), (an, m_an, z_an, n_an) = SALT_CHEMISTRY[r["salt"]]
    mol_L = r["concentration_ppm"] / 1000.0 / m_salt           # mol/L of salt
    ionic_strength = 0.5 * (n_cat * mol_L * z_cat ** 2 + n_an * mol_L * z_an ** 2)
    ion_rows.append({
        "sample": r["sample"],
        "salt_mmol_L": mol_L * 1000,
        "cation": cat,
        "cation_mg_L": n_cat * mol_L * m_cat * 1000,
        "anion": an,
        "anion_mg_L": n_an * mol_L * m_an * 1000,
        "chloride_mg_L": n_an * mol_L * m_an * 1000 if an == "Cl-" else 0.0,
        "sulfate_mg_L": n_an * mol_L * m_an * 1000 if an.startswith("SO4") else 0.0,
        "ionic_strength_mol_L": ionic_strength,
        "net_EC_uS_cm_per_mmol_L": (r["conductivity_uScm"] - ref["conductivity_uScm"]) / (mol_L * 1000),
        "Data_Type": "CALCULATED from formula + transcribed EC",
    })
ion_df = pd.DataFrame(ion_rows)
print(ion_df.round(3).to_string(index=False))
print("\nChloride (NaCl, KCl) is the classic aggressive anion toward passive films;")
print("sulfate is less aggressive but supports sulfate-reducing bacteria (MIC) and")
print("BaSO4/SrSO4 scaling in injection systems. This is chemistry context only,")
print("NOT a corrosion rate.")
save_csv(ion_df, "03_ion_chemistry.csv")

# ============================================================
# 6. RANKING
# ============================================================
banner("5. RANKING OF SALT SOLUTIONS")
for col, label in [("conductivity_uScm", "Conductivity"),
                   ("salinity_ppt", "Salinity"),
                   ("tds_mgL", "TDS")]:
    print(f"\n{label} (highest first)")
    print(brines.sort_values(col, ascending=False)[["sample", col]].to_string(index=False))

# ============================================================
# 7. WATER-QUALITY SUMMARY FIGURE (2 x 2)
# ============================================================
banner("6. WATER-QUALITY GRAPHS")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
panels = [
    ("pH", "pH"),
    ("conductivity_uScm", r"Conductivity ($\mu$S/cm)"),
    ("salinity_ppt", "Salinity (ppt)"),
    ("tds_mgL", "TDS (mg/L)"),
]
x = np.arange(len(df))
labels = df["sample"].tolist()
for ax, (col, ylabel) in zip(axes.ravel(), panels):
    bars = ax.bar(x, df[col], color=["tab:blue"] + ["tab:orange"] * (len(df) - 1))
    if col == "pH":      # hatch assumed values
        for bar, status in zip(bars, df["pH_status"]):
            if status == "assumed":
                bar.set_hatch("//")
                bar.set_edgecolor("black")
        ax.axhline(7.0, ls="--", color="grey", lw=1)
        ax.set_title("pH (hatched = ASSUMED, not measured)")
    else:
        ax.set_title(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, df[col]):
        ax.text(xi, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Water-quality parameters: distilled water vs 1000 ppm brines")
fig.tight_layout()
finish_fig(fig, "fig1_water_quality_summary.png")

# ============================================================
# 8. MODULE 2: SENSITIVITY (indicators only, NOT corrosion rates)
# ============================================================
banner("7. SENSITIVITY ANALYSIS (one parameter at a time; indicators only)")

# 8a. pH sweep -> thermodynamic driving force for H2 evolution
T0 = REFERENCE_TEMPERATURE_C
pH_range = np.linspace(4, 9, 51)
s0 = nernst_slope(T0)
E_H2 = -s0 * pH_range
E_Fe = E0_FE + (s0 / 2) * np.log10(FE2_ACTIVITY)
pH_df = pd.DataFrame({
    "pH": pH_range,
    "H_conc_mol_L": 10.0 ** (-pH_range),
    "E_H2_V_SHE": E_H2,
    "E_Fe_V_SHE": E_Fe,
    "driving_force_V": E_H2 - E_Fe,
    "Held_constant": f"T={T0}C",
    "Data_Type": "CALCULATED (indicator, not a rate)",
})
save_csv(pH_df, "04_sensitivity_pH.csv")
print(f"pH 4 -> 9: driving force falls {pH_df['driving_force_V'].iloc[0]:.3f} V -> "
      f"{pH_df['driving_force_V'].iloc[-1]:.3f} V (H2 evolution more favourable in acid).")
print("In near-neutral water, dissolved-O2 reduction usually controls the rate.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(pH_range, pH_df["driving_force_V"], lw=2, color="tab:red")
ax[0].axvline(7.0, ls="--", color="grey", label="pH 7 (assumed for brines)")
ax[0].axvline(BASELINE["pH"], ls=":", color="tab:blue", label=f"Distilled water pH {BASELINE['pH']}")
ax[0].set_xlabel("pH")
ax[0].set_ylabel("E(H+/H2) - E(Fe2+/Fe) (V)")
ax[0].set_title("Thermodynamic driving force vs pH")
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=8)
ax[1].semilogy(pH_range, 10.0 ** (-pH_range), lw=2, color="tab:blue")
ax[1].set_xlabel("pH")
ax[1].set_ylabel("[H+] (mol/L)")
ax[1].set_title("Hydrogen-ion concentration vs pH")
ax[1].grid(alpha=0.3, which="both")
fig.suptitle("pH sensitivity (indicator only, not a corrosion rate)")
fig.tight_layout()
finish_fig(fig, "fig2_pH_sensitivity.png")

# 8b. Conductivity sweep -> solution resistivity; samples marked
k_range = np.logspace(np.log10(10), np.log10(5000), 60)
cond_df = pd.DataFrame({
    "conductivity_uScm": k_range,
    "resistivity_ohm_cm": 1e6 / k_range,
    "est_ionic_strength_mol_L": 1.6e-5 * k_range,   # empirical rule of thumb
    "Data_Type": "CALCULATED (indicator, not a rate)",
})
save_csv(cond_df, "05_sensitivity_conductivity.csv")
print("\nConductivity: higher conductivity -> lower solution resistance -> anodic and")
print("cathodic sites can be further apart and current spreads more easily.")
print("It supports galvanic/localised attack but does not by itself give a rate.")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(k_range, 1e6 / k_range, lw=2, color="tab:green")
ax.scatter(df["conductivity_uScm"], df["resistivity_ohm_cm"], s=70, zorder=3, color="tab:red")
for _, r in df.iterrows():
    ax.annotate(r["sample"], (r["conductivity_uScm"], r["resistivity_ohm_cm"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel(r"Conductivity ($\mu$S/cm)")
ax.set_ylabel(r"Solution resistivity ($\Omega\cdot$cm)")
ax.set_title("Conductivity sensitivity (measured samples marked)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
finish_fig(fig, "fig3_conductivity_sensitivity.png")

# ============================================================
# 9. MODULE 3: FARADAY'S-LAW CORROSION RATE (ASSUMED Icorr)
# ============================================================
banner("8. THEORETICAL CORROSION RATE (FARADAY'S LAW, ASSUMED Icorr)")
print(f"Metal basis: {METAL_NAME}")
print("Icorr values are HYPOTHETICAL. The same set is applied to every solution so")
print("the model does not falsely imply a salt-specific corrosion rate.\n")

names = list(ICORR_SCENARIOS.keys())
icorr_vals = np.array(list(ICORR_SCENARIOS.values()))
cr_vals = corrosion_rate_mm_year(icorr_vals)
rate_df = pd.DataFrame({
    "scenario": names,
    "Icorr_uA_cm2 (ASSUMED)": icorr_vals,
    "CR_mm_per_year": cr_vals,
    "CR_mpy": cr_vals / 0.0254,
    "NACE_severity_class": [nace_class(c) for c in cr_vals],
    "ratio_to_lowest": cr_vals / cr_vals[0],
    "Data_Type": "CALCULATED from ASSUMED Icorr",
})
print(rate_df.round(4).to_string(index=False))
print("\n'scenario' names are relative labels; 'NACE_severity_class' uses published")
print("oilfield bands (<0.025 low, 0.025-0.12 moderate, 0.13-0.25 high, >0.25 severe mm/yr).")
save_csv(rate_df, "06_corrosion_rates.csv")

i_line = np.linspace(0, 60, 200)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(i_line, corrosion_rate_mm_year(i_line), "k-", lw=1.5, label="Faraday's law")
for n, i, c, col in zip(names, icorr_vals, cr_vals,
                        ["tab:green", "tab:olive", "tab:orange", "tab:red"]):
    ax.scatter(i, c, s=80, color=col, zorder=3, label=f"{n}: {c:.3f} mm/yr")
ax.set_xlabel(r"Assumed corrosion current density, Icorr ($\mu$A/cm$^2$)")
ax.set_ylabel("Theoretical corrosion rate (mm/year)")
ax.set_title("Corrosion rate vs corrosion current density")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
finish_fig(fig, "fig4_corrosion_rate_vs_icorr.png")

# ============================================================
# 10. TEMPERATURE SENSITIVITY (ILLUSTRATIVE ARRHENIUS SCALING)
# ============================================================
banner("9. TEMPERATURE SENSITIVITY (illustrative Arrhenius scenario)")
base_icorr = ICORR_SCENARIOS[BASE_SCENARIO]
base_rate = float(corrosion_rate_mm_year(base_icorr))
print(f"Base scenario: '{BASE_SCENARIO}' (Icorr = {base_icorr} uA/cm2, "
      f"CR = {base_rate:.4f} mm/yr at {REFERENCE_TEMPERATURE_C} C)")
print(f"Assumed apparent activation energy: {APPARENT_ACTIVATION_ENERGY_J_MOL/1000:.0f} kJ/mol "
      "(illustrative, not calibrated).")

temp_df = pd.DataFrame({"temperature_C": SIMULATION_TEMPERATURES_C})
temp_df["relative_factor"] = arrhenius_factor(temp_df["temperature_C"])
temp_df["scenario_CR_mm_per_year"] = base_rate * temp_df["relative_factor"]
temp_df["Data_Type"] = "CALCULATED (assumed Ea)"
print("\n" + temp_df.round(4).to_string(index=False))
print("\nIn a real open system, O2 solubility FALLS with temperature while kinetics")
print("rise, so the real trend can differ from this simple scaling.")
save_csv(temp_df, "07_temperature_sensitivity.csv")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(temp_df["temperature_C"], temp_df["scenario_CR_mm_per_year"], marker="o", lw=2)
ax.set_xlabel("Temperature (\u00b0C)")
ax.set_ylabel("Scenario corrosion rate (mm/year)")
ax.set_title("Illustrative temperature sensitivity (Arrhenius, assumed Ea)")
ax.grid(alpha=0.3)
fig.tight_layout()
finish_fig(fig, "fig5_temperature_sensitivity.png")

# ============================================================
# 11. MODULE 4: METAL THICKNESS LOSS (constant rate)
# ============================================================
banner("10. THEORETICAL METAL THICKNESS LOSS (constant corrosion rate)")
loss_df = pd.DataFrame({"scenario": names, "CR_mm_per_year": cr_vals})
for y in EXPOSURE_YEARS:
    loss_df[f"loss_{y}yr_mm"] = cr_vals * y
loss_df["Data_Type"] = "CALCULATED (constant-rate assumption)"
print(f"At reference temperature ({REFERENCE_TEMPERATURE_C} C):")
print(loss_df.round(3).to_string(index=False))
save_csv(loss_df, "08_thickness_loss_by_scenario.csv")

rows = []
for _, r in temp_df.iterrows():
    for y in EXPOSURE_YEARS:
        rows.append({"temperature_C": r["temperature_C"], "exposure_years": y,
                     "CR_mm_per_year": r["scenario_CR_mm_per_year"],
                     "thickness_loss_mm": r["scenario_CR_mm_per_year"] * y})
loss_temp_df = pd.DataFrame(rows)
print(f"\nBase scenario '{BASE_SCENARIO}' with the temperature factor:")
print(loss_temp_df.round(3).to_string(index=False))
save_csv(loss_temp_df, "09_thickness_loss_by_temperature.csv")

t = np.linspace(0, max(EXPOSURE_YEARS), 200)
fig, ax = plt.subplots(figsize=(7.5, 5))
for n, c, col in zip(names, cr_vals, ["tab:green", "tab:olive", "tab:orange", "tab:red"]):
    ax.plot(t, c * t, lw=2, color=col, label=f"{n} ({c:.3f} mm/yr)")
    ax.scatter(EXPOSURE_YEARS, c * np.array(EXPOSURE_YEARS), color=col, zorder=3)
ax.set_xlabel("Exposure time (years)")
ax.set_ylabel("Theoretical thickness loss (mm)")
ax.set_title("Metal thickness loss vs exposure time (constant rate)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
finish_fig(fig, "fig6_thickness_loss_vs_time.png")

# ============================================================
# 12. FINAL SUMMARY FILE
# ============================================================
summary = brines[["sample", "salt", "concentration_ppm", "pH", "pH_status",
                  "conductivity_uScm", "salinity_ppt", "tds_mgL",
                  "conductivity_uScm_change_pct_vs_ref",
                  "tds_mgL_change_pct_vs_ref",
                  "conductivity_uScm_net_of_water"]].copy()
save_csv(summary, "10_final_brine_summary.csv")

banner("SIMULATION FINISHED")
print(f"All CSV tables and PNG figures are in the '{OUTPUT_DIR}' folder.")
print("\nFINAL INTERPRETATION")
print("- The lab data compare water-quality properties of three 1000 ppm brines.")
print("- pH = 7 for the salt solutions is ASSUMED, not measured.")
print("- Corrosion rates depend on ASSUMED current densities, not on the water data.")
print("- Do not present these scenario rates as measured or predicted field corrosion.")
print("- For real prediction, use polarisation/LPR or weight-loss data, or a validated")
print("  model for the relevant steel, water chemistry and operating conditions.")