from dash import dcc, html

def create_layout(df):
    # Pre-filter valid columns
    categorical_cols = [col for col in df.select_dtypes(include="object").columns 
                        if df[col].nunique() < 50]  # avoid IDs
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [col for col in df.columns if "date" in col.lower() or "timestamp" in col.lower()]

    return html.Div([
        html.H1("📊 E-commerce Analytics Dashboard", className="header"),

        dcc.Tabs([

            # ---------------- EDA SECTION ----------------
            dcc.Tab(label="📊 Exploratory Data Analysis (EDA)", children=[
                html.Div([
                    html.H3("EDA Tools", className="section-title"),
                    dcc.Tabs([

                        # --- Bar Plot ---
                        dcc.Tab(label="Bar Plot", children=[
                            html.Div([
                                html.H4("Categorical Distribution", className="sub-title"),
                                dcc.Dropdown(
                                    id="eda_bar_dropdown",
                                    options=[{"label": col, "value": col} 
                                             for col in categorical_cols],
                                    placeholder="Select a categorical column"
                                ),
                                dcc.Graph(id="eda_bar_plot")
                            ], className="tab_content"),
                        ]),

                        # --- Histogram ---
                        dcc.Tab(label="Histogram", children=[
                            html.Div([
                                html.H4("Numeric Distribution", className="sub-title"),
                                dcc.Dropdown(
                                    id="eda_hist_dropdown",
                                    options=[{"label": col, "value": col} 
                                             for col in numeric_cols],
                                    placeholder="Select a numeric column"
                                ),
                                dcc.Graph(id="eda_hist_plot")
                            ], className="tab_content"),
                        ]),

                        # --- Correlation ---
                        dcc.Tab(label="Correlation Heatmap", children=[
                            html.Div([
                                html.H4("Correlation Matrix", className="sub-title"),
                                dcc.Graph(id="eda_corr_heatmap")
                            ], className="tab_content"),
                        ]),

                        # --- Time Trends ---
                        dcc.Tab(label="Trends Over Time", children=[
                            html.Div([
                                html.H4("Order Trends Over Time", className="sub-title"),
                                dcc.Dropdown(
                                    id="eda_time_dropdown",
                                    options=[{"label": col, "value": col} 
                                             for col in date_cols],
                                    placeholder="Select a date column"
                                ),
                                dcc.Graph(id="eda_time_plot")
                            ], className="tab_content"),
                        ]),

                    ])
                ])
            ]),

            # ---------------- HYPOTHESIS SECTION ----------------
            dcc.Tab(label="📈 Hypothesis Testing", children=[
                html.Div([
                    html.H3("Hypothesis: Do late deliveries affect review scores?", className="section-title"),
                    dcc.Graph(id="hypothesis_plot"),
                    html.Div(id="hypothesis_result", className="stats-box")
                ], className="tab_content")
            ]),

            # ---------------- MODELING SECTION ----------------
            dcc.Tab(label="🤖 Modeling", children=[
                html.Div([
                    html.H3("Logistic Regression Model", className="section-title"),
                    dcc.Graph(id="model_feature_importance"),
                    html.Div(id="model_accuracy", className="stats-box")
                ], className="tab_content")
            ]),

        ])
    ], className="main_container")
