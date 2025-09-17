import os
import pandas as pd
from dash import Dash
import layouts
import callbacks

# ----------------------------
# Load CSV safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # dashboard/
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "Processed", "clean_orders.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# ----------------------------
# Create Dash App
# ----------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "E-commerce Analytics Dashboard"

# Deployment server (Gunicorn, Render, etc.)
server = app.server  

# Layout + Callbacks
app.layout = layouts.create_layout(df)
callbacks.register_callbacks(app, df)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)   # ✅ Dash 3.2 style
