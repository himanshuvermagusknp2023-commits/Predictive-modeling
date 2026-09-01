from fastapi import FastAPI
from pydantic import BaseModel
from optimize import DynamicPricingOptimizer
import uvicorn

app = FastAPI(
    title="Adaptive Dynamic Pricing API",
    description="Real-time demand forecasting and constrained pricing engine"
)

optimizer = DynamicPricingOptimizer()

class MarketContext(BaseModel):
    hour_of_day: int
    is_weekend: int
    traffic_index: float
    competitor_price: float
    occupancy_ratio: float
    min_price: float = 5.0
    max_price: float = 50.0

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DynamicPricingAPI"}

@app.post("/recommend-price")
def get_price_recommendation(context: MarketContext):
    market_data = {
        "hour_of_day": context.hour_of_day,
        "is_weekend": context.is_weekend,
        "traffic_index": context.traffic_index,
        "competitor_price": context.competitor_price,
        "occupancy_ratio": context.occupancy_ratio
    }
    
    result = optimizer.optimize_price(
        context=market_data,
        p_min=context.min_price,
        p_max=context.max_price
    )
    return {
        "status": "success",
        "market_context": market_data,
        "pricing_recommendation": result
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
