# 🏭 Industrial Machine Fault Detection Using Deep Learning
## Complete Installation, Setup & Execution Guide for New Computers

> **Welcome!** This guide is designed so anyone (including your friends, evaluators, or lab instructors) can set up and run this project from scratch on any Windows PC without errors.

---

## 🔍 Step 0: Check FIRST Before Installing Anything! (Pre-Flight Check)

Before running any installation commands, verify these 4 requirements on the computer:

### 1. Check Python Version
Open **PowerShell** or **Command Prompt** and type:
```powershell
python --version
```
- ✅ **Required:** Python **3.9**, **3.10**, **3.11**, or **3.12** (64-bit).
- ❌ **If Python is NOT installed or shows an older version:**
  1. Download Python 3.12 from [python.org/downloads](https://www.python.org/downloads/).
  2. ⚠️ **CRITICAL STEP:** During installation, **CHECK the box** that says **`☑ Add python.exe to PATH`** at the bottom of the installer window!
  3. Complete installation, then close and reopen PowerShell.

### 2. Verify Python Architecture is 64-bit
Run:
```powershell
python -c "import struct; print(struct.calcsize('P') * 8)"
```
- It must print **`64`**. (TensorFlow requires 64-bit Python).

### 3. Check PowerShell Execution Policy
Windows often restricts running scripts by default. In PowerShell, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
*(Press `Y` and Enter if prompted. This permits activating virtual environments safely).*

### 4. Check Hardware Requirements
- **RAM:** Minimum 4 GB (8 GB recommended).
- **Processor:** Any modern Intel Core or AMD CPU (No dedicated GPU required; runs smoothly on CPU).
- **Disk Space:** ~1.5 GB free disk space for Python packages and model weights.

---

## 📦 Complete Requirements List

All required libraries are specified in [`requirements.txt`](./requirements.txt):

| Package | Recommended Version | Why It's Needed |
| :--- | :---: | :--- |
| **`python`** | `3.10 – 3.12 (64-bit)` | Core runtime environment |
| **`tensorflow`** | `== 2.16.2` | Deep Neural Network (MLP) architecture & training |
| **`streamlit`** | `>= 1.36.0, < 1.41.0` | Web dashboard & interactive user interface |
| **`numpy`** | `== 1.26.4` *(< 2.0.0)* | High-performance numerical matrix computations *(Must be < 2.0 for TensorFlow compatibility)* |
| **`pyarrow`** | `>= 14.0.0, < 19.0.0` | High-speed data table rendering in Streamlit *(Must be < 19.0 to match NumPy 1.26.4)* |
| **`pandas`** | `>= 2.0.0, < 2.3.0` | CSV ingestion, data manipulation & preprocessing |
| **`scikit-learn`**| `>= 1.4.0, < 1.6.0` | `StandardScaler`, `OneHotEncoder`, confusion matrix, ROC-AUC metrics |
| **`scipy`** | `>= 1.13.0, < 1.14.0` | Scientific computing & statistical distribution tools |
| **`matplotlib`** | `>= 3.7.0` | Plotting training curves, loss history, and diagnostics |
| **`seaborn`** | `>= 0.13.0` | High-resolution confusion matrix heatmaps |
| **`joblib`** | `>= 1.3.0` | Saving & loading feature scalers and encoders |
| **`ucimlrepo`** | `>= 0.0.7` | UCI Machine Learning Repository dataset loader |

---

## 🚀 Step-by-Step Installation & Setup

Follow these exact steps to get the project running:

### Step 1: Open Terminal in the Project Directory
1. Open the project folder.
2. Click the folder address bar, type `powershell`, and press **Enter**.

### Step 2: Create a Clean Virtual Environment
A virtual environment ensures project packages do not conflict with system packages:
```powershell
python -m venv .venv
```
*(Wait 10–20 seconds for the `.venv` folder to be created).*

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
> **Verification:** Your terminal prompt will now start with **`(.venv)`**, indicating the environment is active!

*(If you get a script execution error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry).*

### Step 4: Upgrade Pip
```powershell
python -m pip install --upgrade pip
```

### Step 5: Install All Dependencies
Install the pinned, fully compatible library versions in one command:
```powershell
python -m pip install -r requirements.txt
```
*(This downloads TensorFlow, Streamlit, Pandas, NumPy 1.26.4, PyArrow, Scikit-learn, etc. It takes 1–3 minutes depending on your internet speed).*

---

## 🌐 How to Run the Web Application

With your virtual environment active (`(.venv)` in prompt), run:

```powershell
streamlit run app.py
```
*(Or if running from root folder: `streamlit run industrial_fault_detection_dl\app.py`)*

### 🖥️ Open the Dashboard
Your default browser will automatically open:
👉 **`http://localhost:8501`**

---

## 🧪 How to Test and Explore the System

Once the dashboard loads, here is how you can demonstrate all features:

### 1. ⚡ Mode A: Pre-trained Milling Machine Diagnostic
- **Single Machine Inference:**
  - Adjust the 6 sensor sliders: *Type, Air Temp [K], Process Temp [K], Speed [rpm], Torque [Nm], Tool Wear [min]*.
  - Click **`Run Fault Diagnostic`**.
  - An animated gauge displays the **Failure Probability %**, color-coded status badge, and immediate maintenance advice.
  - Scroll down to review the pre-computed **Confusion Matrix**, **ROC Curve (AUC = 0.973)**, and **Feature Importance ranking**.
- **Batch CSV Processing:**
  - Switch to the **Batch CSV Prediction** tab.
  - Click **`📂 Load Pre-Packaged Test Records`** (or upload your own CSV).
  - The model computes predictions across the fleet in seconds, showing summary counts and a downloadable CSV.

### 2. 🧪 Mode B: Train a Brand New Deep Learning Model
- Click the top tab **`🧪 Mode B: Train New Dataset`**.
- Upload any custom labelled CSV (e.g. `compressor_training_dataset.csv`, `wind_turbine_training_dataset.csv`, or click **`Load Pre-packaged Industrial Water Pump Dataset`**).
- Select your target failure column (e.g., `Fault`).
- Enter a model name (e.g. `gas_compressor_v1`).
- Click **`🚀 Start Neural Network Training`**.
- The neural network builds, applies balanced class weighting, trains with early stopping, and displays the **Held-Out Test Accuracy**, **ROC-AUC**, and **Loss/Accuracy convergence curves**.
- The new model is saved into `models/custom_models/<model_name>/` without modifying Mode A.

### 3. 🔮 Custom Model Inference
- Click the top tab **`🔮 Custom Model Inference`**.
- Select your custom model from the dropdown.
- Inspect its **5 Performance Metric Badges** (Accuracy, Precision, Recall, F1, ROC-AUC) and expand **"📈 View Model Visualizations & Diagnostic Curves"** to see its Confusion Matrix heatmap and Loss curves!
- Upload new test records (or click the convenient one-click demo button for compressors, pumps, or turbines).
- Click **`⚡ Run Predictions`** to generate the failure probabilities and download the results.

---

## ❓ Troubleshooting Common Errors

### 1. Error: `Activate.ps1 cannot be loaded because running scripts is disabled`
- **Solution:** Run this command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  Then reactivate: `.\.venv\Scripts\Activate.ps1`.

### 2. Error: `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`
- **Cause:** A mismatch between PyArrow and NumPy.
- **Solution:** Install the compatible PyArrow version:
  ```powershell
  python -m pip install "pyarrow>=14.0.0,<19.0.0"
  ```

### 3. Error: `streamlit : The term 'streamlit' is not recognized`
- **Cause:** The virtual environment is not activated, or packages were installed globally.
- **Solution:** Activate `.venv` first (`.\.venv\Scripts\Activate.ps1`), then ensure Streamlit is installed:
  ```powershell
  python -m pip install -r requirements.txt
  ```

### 4. Error: `Port 8501 is already in use`
- **Solution:** Run Streamlit on a different port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```
  Then access **`http://localhost:8502`**.

### 5. Missing Microsoft Visual C++ Runtime (On fresh Windows installs)
- **Symptom:** `DLL load failed while importing _pywrap_tensorflow_internal`.
- **Solution:** Download and install the free official Microsoft Visual C++ Redistributable:
  [Visual C++ 2015–2022 x64 Installer](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

## 📁 Repository Directory Map

```text
industrial_fault_detection_dl/
│
├── README.md                                  # This setup & troubleshooting guide
├── requirements.txt                           # Pinned package versions
├── app.py                                     # Streamlit Web Dashboard
├── train_model.py                             # Training pipeline for AI4I dataset
├── custom_trainer.py                          # Custom dataset trainer & inference engine
│
├── data/
│   ├── ai4i2020.csv                           # Baseline 10,000-record AI4I dataset
│   ├── sample_test_records.csv                # Mode A sample batch test records
│   └── sample_custom_pump_dataset.csv         # Mode B pump sample dataset
│
├── models/
│   ├── fault_mlp.keras                        # Pre-trained neural network for AI4I
│   ├── scaler.joblib                          # Fitted StandardScaler
│   └── custom_models/                         # User-trained custom models directory
│
└── reports/                                   # Visual evaluation curves & metrics
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── training_curves.png
    └── metrics.json
```

---
*Happy Diagnosing! Designed for reliable, zero-headache academic and industrial presentations.*
