# Live AQI Predictor & City Intelligence Frontend

A streamlined, dedicated live predictor web interface for the Data Mining project: **Regime-Aware Air Quality Index (AQI) Prediction & Pattern Mining System**.

> **Note:** The original analysis script [`dm_project_regime_aware_aqi_final.py`](file:///c:/Users/Administrator/Downloads/DM%20PROJECT/dm_project_regime_aware_aqi_final.py) is untouched. Only frontend files have been modified.

---

## 🌟 Quick Start

### Method 1: Instant Standalone (No Terminal Needed)
Double-click on [**`index.html`**](file:///c:/Users/Administrator/Downloads/DM%20PROJECT/index.html).
* Opens immediately in Google Chrome or Microsoft Edge.
* Runs with full in-browser machine learning and regime calculation logic with zero setup.

### Method 2: Launch with Flask Backend (One-Click)
Double-click on [**`run_frontend.bat`**](file:///c:/Users/Administrator/Downloads/DM%20PROJECT/run_frontend.bat).
* Starts the Python Flask backend at `http://127.0.0.1:5000` and opens your browser automatically.

---

## 🚀 Key Features

1. **🔍 Real-Time City & Station Search Bar:**
   * Search any Indian city or place (e.g. *Delhi, Mumbai, Bengaluru, Ahmedabad, Kolkata, Chennai, Hyderabad, Lucknow, Patna, Jaipur, Amritsar, Chandigarh, Kochi, Gurugram, Visakhapatnam, etc.*).
   * Live autocomplete dropdown with regional context and historical pollution tags.
   * Quick-click chips for popular metropolitan hubs.

2. **🏙️ Automatic Station Details & Profile Fetching:**
   * Selecting a city automatically loads its characteristic historical monitoring concentrations across all 12 criteria pollutants (`PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene`).
   * Displays city location metadata, geographic region, and atmospheric emission profile (e.g., Delhi Winter Inversion Dome vs Ahmedabad Industrial VOC Corridor vs Bengaluru Deccan Dispersion).

3. **🔬 Regime-Aware Machine Learning Classification:**
   * Automatically standardizes inputs against training baselines and assigns the observation to:
     * **Regime 0: High Multi-Pollutant Regime** (severe winter smog, thermal inversions, multi-pollutant amplification).
     * **Regime 1: Lower Pollution Regime** (moderate/clean coastal dispersion).
   * Displays Euclidean cluster distance and confidence percentage.

4. **📊 Multi-Model Estimates Comparison:**
   * Side-by-side comparison of:
     * **Random Forest** (Regime-Specific vs Global Baseline)
     * **XGBoost** (Regime-Specific vs Global Baseline)
     * **AdaBoost** (Regime-Specific vs Global Baseline)
     * **Official Indian CPCB Formula** (Sub-index breakpoint reference)

5. **🎛️ Interactive Sensitivity & "What-If" Analysis:**
   * Sliders and numeric input boxes allow users to tweak individual pollutant levels to simulate emission reductions or localized industrial spikes.
   * Interactive CPCB sub-index breakdown chart and circular AQI gauge with official Indian national health advisories.
