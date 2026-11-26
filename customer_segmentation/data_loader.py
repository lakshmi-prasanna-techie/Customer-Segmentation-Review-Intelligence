from pathlib import Path
import numpy as np
import pandas as pd
from config import RANDOM_STATE


def load_or_generate_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        print(f"[data_loader] Loading dataset from: {csv_path}")
        return pd.read_csv(csv_path)

    print("[data_loader] Generating synthetic multi-persona customer & review dataset...")
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = 2400

    advocate_reviews = [
        "Amazing quality and fast delivery, I love ordering here and recommend it to everyone",
        "Excellent customer support and smooth checkout experience, worth every penny",
        "Best purchase this year, reliable shipping and perfect product condition",
    ]
    mainstream_reviews = [
        "Product works as expected, delivery was okay and price is fair",
        "Decent quality overall, nothing extraordinary but satisfied with the order",
        "Good item for everyday use, arrived on time without major issues",
    ]
    at_risk_reviews = [
        "Package arrived late and the item was damaged, very disappointed and want a refund",
        "Terrible support experience, defective product and slow response to cancel order",
        "Broken on arrival, poor packaging and worst delivery delay I have seen",
    ]
    dormant_reviews = [
        "Bought once a long time ago, product was fine",
        "No recent orders, item was okay when purchased",
        "Average experience from months back",
    ]

    records = []
    personas = ["advocate", "mainstream", "at_risk", "dormant"]
    weights = [0.28, 0.34, 0.20, 0.18]

    for i in range(n_samples):
        p = rng.choice(personas, p=weights)
        if p == "advocate":
            recency = int(rng.integers(1, 28))
            freq = int(rng.integers(14, 38))
            aov = float(rng.gamma(4.0, 38.0))
            rev_cnt = int(rng.integers(5, 22))
            rating = float(np.clip(rng.normal(4.75, 0.25), 1.0, 5.0))
            text = rng.choice(advocate_reviews)
        elif p == "mainstream":
            recency = int(rng.integers(20, 75))
            freq = int(rng.integers(5, 16))
            aov = float(rng.gamma(3.0, 28.0))
            rev_cnt = int(rng.integers(1, 10))
            rating = float(np.clip(rng.normal(3.95, 0.45), 1.0, 5.0))
            text = rng.choice(mainstream_reviews)
        elif p == "at_risk":
            recency = int(rng.integers(45, 160))
            freq = int(rng.integers(2, 12))
            aov = float(rng.gamma(2.8, 30.0))
            rev_cnt = int(rng.integers(2, 14))
            rating = float(np.clip(rng.normal(2.05, 0.55), 1.0, 5.0))
            text = rng.choice(at_risk_reviews)
        else:
            recency = int(rng.integers(120, 260))
            freq = int(rng.integers(1, 4))
            aov = float(rng.gamma(2.2, 22.0))
            rev_cnt = int(rng.integers(0, 3))
            rating = float(np.clip(rng.normal(3.4, 0.6), 1.0, 5.0))
            text = rng.choice(dormant_reviews)

        records.append({
            "customer_id": f"CUST_{i:04d}",
            "recency_days": recency,
            "order_frequency": freq,
            "avg_order_value": round(aov, 2),
            "review_count": rev_cnt,
            "avg_star_rating": round(rating, 2),
            "review_text": text,
        })

    return pd.DataFrame(records)
