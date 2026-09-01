import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

class DynamicPricingOptimizer:
    def __init__(self, model_path="best_demand_model.pkl"):
        self.model = joblib.load(model_path)
        
    def predict_demand(self, price, context):
        features = pd.DataFrame([{
            "hour_of_day": context["hour_of_day"],
            "is_weekend": context["is_weekend"],
            "traffic_index": context["traffic_index"],
            "competitor_price": context["competitor_price"],
            "occupancy_ratio": context["occupancy_ratio"],
            "price": price,
            "price_to_competitor_ratio": price / (context["competitor_price"] + 1e-5),
            "is_peak_hour": 1 if 8 <= context["hour_of_day"] <= 19 else 0
        }])
        pred = self.model.predict(features)[0]
        return max(0, pred)
    
    def optimize_price(self, context, p_min=5.0, p_max=40.0):
        # Objective: minimize negative revenue
        def objective(price):
            predicted_demand = self.predict_demand(price, context)
            revenue = price * predicted_demand
            return -revenue
        
        res = minimize_scalar(objective, bounds=(p_min, p_max), method="bounded")
        opt_price = np.round(res.x, 2)
        opt_demand = np.round(self.predict_demand(opt_price, context), 1)
        expected_revenue = np.round(opt_price * opt_demand, 2)
        
        return {
            "optimal_price": opt_price,
            "expected_demand": opt_demand,
            "expected_revenue": expected_revenue
        }

if __name__ == "__main__":
    optimizer = DynamicPricingOptimizer()
    sample_context = {
        "hour_of_day": 17,
        "is_weekend": 0,
        "traffic_index": 4.2,
        "competitor_price": 18.0,
        "occupancy_ratio": 0.75
    }
    recommendation = optimizer.optimize_price(sample_context)
    print("Optimization Output for Peak Rush Hour:")
    print(recommendation)
