# AI FraudGuard — Streamlit Web App

A modern interactive educational prototype for credit-card fraud classification. Built with Streamlit, Plotly, scikit-learn, and XGBoost.


## Quick start in VS Code (Windows)
1. Extract the ZIP and open the `AI-FraudGuard` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create a virtual environment:
   ```powershell
   py -3.13 -m venv .venv
   ```
4. Activate it (if PowerShell blocks scripts, run the process-only policy command first):
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
5. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Launch:
   ```powershell
   streamlit run app.py
   ```
7. Streamlit opens the site at `http://localhost:8501`.

If activation is inconvenient, run pip and Streamlit using the venv executable directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Pages
- Dashboard: dataset KPIs, class distribution, amount histogram and model summary
- AI Fraud Detector: model prediction for a chosen transaction row or manually entered features
- Transaction Explorer: fraud/legitimate and amount filters, CSV export
- Model Performance: model metric comparison and confusion matrices
- Fraud Analytics: amount summary and time trends
- Dataset Center: upload CSV, preview, checks and download


## Data upload
Use the sidebar CSV uploader. To train on your own dataset, it must include `Class` (0 = legitimate, 1 = fraud) and numeric feature columns. For the commonly used credit-card fraud dataset, the columns are usually `Time`, `V1` ... `V28`, `Amount`, `Class`.


## Demo mode
When no file is uploaded, the app creates synthetic demo transactions. These data are fictional and are **not** the original dataset or real banking transactions. Demo mode allows you to preview the UI and workflows immediately.


## Deployment
Push this project to GitHub (keep `app.py` and `requirements.txt` at repository root), then create a Streamlit Community Cloud app and select the repository, branch, and `app.py` entry point. Upload your real CSV in the app if it is not stored in the repository.


## Important
This is an academic/demo prototype. Model output is not calibrated, is not a bank decision, and must not be used to approve/decline real transactions. In demo mode, metrics are calculated on synthetic data.

## HTML templates are included
This hybrid Streamlit app includes real HTML templates in `templates/` and uses them for page header/hero components.
- `dashboard.html`
- `fraud_detection.html`
- `transactions.html`
- `analytics.html`
- `dataset_center.html`

Shared static assets: `static/css/style.css` and `static/js/app.js`.
Streamlit remains the app framework and ML/backend runtime; HTML template headers are embedded with Streamlit components.
