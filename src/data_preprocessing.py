# ===============================================
# src/data_preprocessing.py
# Person 1: Data & Preprocessing Lead
# Project: Do Delays Damage Loyalty?
# ===============================================

import os
import pandas as pd

# Get the repository root directory (parent of src/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")  # lowercase 'raw' to match your structure
PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "processed", "clean_orders.csv")

def load_raw_data(path=RAW_DIR):
    """
    Load the 4 key Olist CSVs needed for this project.
    """
    print("📂 Loading raw CSVs from:", path)
    
    # Check if directory exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data directory not found: {path}")
    
    # Load CSV files with proper path joining
    orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"))
    items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))
    
    print("✅ Raw data loaded successfully!")
    return orders, reviews, items, customers

def clean_and_merge_data(path=RAW_DIR):
    """
    Merge datasets and create key features:
    - delivery_delay (days late/early)
    - shipping_time (days from purchase to delivery)
    - delay_flag ("Late"/"On-time"/"Early")
    """
    print("🛠️ Starting cleaning and merging...")
    
    orders, reviews, items, customers = load_raw_data(path)
    
    # Merge orders + reviews
    df = orders.merge(reviews[['order_id','review_score']], on='order_id', how='left')
    
    # Merge customer info
    df = df.merge(customers[['customer_id','customer_state']], on='customer_id', how='left')
    
    # Aggregate items: sum of price and freight_value
    items_agg = items.groupby('order_id').agg(
        total_price=('price','sum'),
        total_freight=('freight_value','sum')
    ).reset_index()
    df = df.merge(items_agg, on='order_id', how='left')
    
    print(f"🔗 Merged datasets. Shape: {df.shape}")
    
    # Convert timestamps
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    
    # Drop rows without actual delivery date
    before = df.shape[0]
    df = df.dropna(subset=['order_delivered_customer_date'])
    after = df.shape[0]
    print(f"🧹 Dropped {before-after} rows without delivery date")
    
    # Feature engineering
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['shipping_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['delay_flag'] = df['delivery_delay'].apply(
        lambda x: "Late" if x > 0 else ("On-time" if x == 0 else "Early")
    )
    
    # Keep relevant columns
    final_cols = [
        'order_id','customer_id','customer_state',
        'order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date',
        'delivery_delay','shipping_time','delay_flag','review_score',
        'total_price','total_freight'
    ]
    df = df[final_cols]
    
    print("✨ Feature engineering complete!")
    return df

def save_processed_data(output=PROCESSED_PATH):
    """
    Save cleaned dataset to processed folder
    """
    df = clean_and_merge_data()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    df.to_csv(output, index=False)
    print(f"✅ Saved processed dataset to {output}. Rows: {df.shape[0]}, Columns: {df.shape[1]}")

def load_processed_data(path=PROCESSED_PATH):
    """
    Load processed dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run save_processed_data() first.")
    
    print(f"📂 Loading processed dataset from {path}")
    return pd.read_csv(path)

# If run as main, generate processed dataset
if __name__ == "__main__":
    save_processed_data()