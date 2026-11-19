# 🚗 AutoValue AI

### Intelligent Vehicle Valuation & Market Analytics System

An AI-powered platform for used-car price estimation, vehicle discovery and automotive market insights.

Built with **Python, Machine Learning and Streamlit**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💰 Price Prediction | Estimate used-car prices using a trained ML model. |
| 🚘 Budget Car Finder | Discover matching listings using budget and vehicle filters. |
| 📊 Market Analytics | Explore prices, brands, fuel types and city-wise listings. |
| ⚖️ Compare Cars | Compare vehicle specifications and asking prices. |
| 🧠 Explainable AI | Understand model predictions using SHAP. |
| 📈 Model Evaluation | Compare regression algorithms using performance metrics. |
| 📄 PDF Reports | Generate downloadable vehicle valuation reports. |

## 🤖 Machine Learning

Four regression algorithms were evaluated:

- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)

**Selected Model: Tuned Random Forest Regressor**

| Evaluation Metric | Result |
|---|---:|
| MAE | ₹72,117.16 |
| RMSE | ₹133,110.64 |
| R² Score | 0.9197 |

*Results are from the project's test split. The test set was used during model comparison and tuning.*

## 🛠️ Tech Stack

**Language:** Python

**Machine Learning:** Scikit-learn, XGBoost

**Data Processing:** Pandas, NumPy

**Visualization:** Plotly, Matplotlib, Seaborn

**Explainable AI:** SHAP

**Web Application:** Streamlit

**Other Tools:** Joblib, ReportLab, Jupyter Notebook

## 📂 Project Structure

```text
AutoValue-AI/
│
├── assets/          # Images and visualizations
├── data/            # Raw and processed datasets
├── models/          # Trained ML model
├── notebooks/       # Jupyter notebooks
├── reports/         # Project reports
├── src/             # Data processing and ML scripts
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/VedantMule04/AutoValue-AI.git
cd AutoValue-AI
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate it (Windows)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**5. Run the application**

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📊 Dataset

The project uses two separate datasets:

- **Historical CarDekho data:** Used for machine learning, preprocessing and model development.
- **TCO India used-car listings:** Used for market analytics, vehicle comparison and budget-based car discovery.

Recent listing data is a snapshot dated **24 August 2026**, not a live inventory feed.

*Dataset prices represent historical selling-price records or advertised listing prices, not independently verified transactions. Please refer to the original dataset providers for licensing and redistribution conditions.*

## ⚠️ Disclaimer

AutoValue AI is an academic machine learning project. Predicted prices are estimates and should not be considered guaranteed selling prices or professional vehicle appraisals.

## 👨‍💻 Developer

**Vedant Mule**

Computer Engineering | DBATU, Lonere

[GitHub Profile](https://github.com/VedantMule04)

---

<p align="center">
  <strong>AutoValue AI</strong><br/>
  Intelligent Valuation. Smarter Insights.
</p>