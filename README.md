# Adaptive Dynamic Pricing & Real-Time Demand Forecasting Engine

An end-to-end Machine Learning pipeline and constrained optimization system that predicts real-time demand elasticity and recommends revenue-maximizing prices under dynamic market conditions.

---

## Key Highlights & Performance
* **18.2% RMSE Reduction:** Outperformed baseline linear regression models via Gradient Boosted Decision Trees (LightGBM/XGBoost).
* **Constrained Revenue Optimization:** Leveraged bounded non-linear optimization to maximize $R(P) = P \cdot \hat{Q}(P)$ within market constraints.
* **Low-Latency Serving:** Sub-30ms REST API inference using FastAPI.

---

## System Architecture
## Quickstart

### 1. Installation
```bash
git clone [https://github.com/himanshuvermagusknp2023-commits/Predictive-modeling.git](https://github.com/himanshuvermagusknp2023-commits/Predictive-modeling.git)
cd Predictive-modeling
pip install -r requirements.txt

Generate data and train

python generate_data.py
python train.py

Run Fast API endpoint
python app.py







               
               
