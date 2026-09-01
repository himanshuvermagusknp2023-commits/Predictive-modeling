import numpy as np
import pandas as pd

def generate_parking_demand_dataset(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    
    # Base features
    hour_of_day = np.random.randint(0, 24, size=n_samples)
    is_weekend = np.random.binomial(1, 0.28, size=n_samples)
    traffic_index = np.random.uniform(1.0, 5.0, size=n_samples)  # 1 (low) to 5 (heavy)
    competitor_price = np.random.uniform(5.0, 25.0, size=n_samples)
    occupancy_ratio = np.random.uniform(0.1, 0.95, size=n_samples)
    
    # Pricing policy (current posted price)
    price = competitor_price * np.random.uniform(0.85, 1.25, size=n_samples)
    
    # Ground truth demand simulation with non-linear price elasticity
    # Elasticity: Higher price reduces demand, high traffic/peak hours boost demand
    peak_multiplier = np.where((hour_of_day >= 8) & (hour_of_day <= 19), 1.4, 0.7)
    weekend_multiplier = np.where(is_weekend == 1, 1.25, 1.0)
    
    # Demand function: Demand decreases as (Price / CompetitorPrice) increases
    price_ratio = price / (competitor_price + 1e-5)
    base_demand = (
        120 * peak_multiplier * weekend_multiplier 
        + 15 * traffic_index 
        - 45 * np.log1p(price_ratio * 3) 
        + 10 * (1.0 - occupancy_ratio)
    )
    
    noise = np.random.normal(0, 3.5, size=n_samples)
    demand = np.maximum(0, base_demand + noise).round().astype(int)
    
    df = pd.DataFrame({
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "traffic_index": np.round(traffic_index, 2),
        "competitor_price": np.round(competitor_price, 2),
        "occupancy_ratio": np.round(occupancy_ratio, 2),
        "price": np.round(price, 2),
        "demand": demand
    })
    
    df.to_csv("parking_pricing_data.csv", index=False)
    print(f"Dataset generated with {n_samples} samples saved to 'parking_pricing_data.csv'.")

if __name__ == "__main__":
    generate_parking_demand_dataset()
