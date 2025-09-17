from dash import Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
  

# ---------- helpers ----------
def _style(fig, title=None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font_color="#f5f5f5",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def empty_fig(kind="bar", title=""):
    """Return a tiny valid figure so px doesn't crash when no data."""
    if kind == "bar":
        fig = px.bar(x=["No data"], y=[0], title=title)
    elif kind == "hist":
        fig = px.histogram(x=[0], title=title)
    elif kind == "line":
        fig = px.line(x=[0], y=[0], title=title)
    elif kind == "box":
        fig = px.box(x=["No data"], y=[0], title=title)
    elif kind == "heatmap":
        fig = px.imshow([[0]], text_auto=True, title=title)
    else:
        fig = px.scatter(x=[0], y=[0], title=title)
    return _style(fig)


# ---------- callbacks ----------
def register_callbacks(app, df):

    # ====== EDA: BAR PLOT ======
    @app.callback(
        Output("eda_bar_plot", "figure"),
        Input("eda_bar_dropdown", "value")
    )
    def update_bar(col):
        if not col or col not in df.columns:
            return empty_fig("bar", "Select a categorical column")

        try:
            vc = df[col].astype("category").value_counts(dropna=False).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count", title=f"Bar Plot of {col}")
            fig.update_xaxes(type="category")
            return _style(fig)
        except Exception as e:
            return empty_fig("bar", f"Error plotting bar chart: {e}")

    # ====== EDA: HISTOGRAM ======
    @app.callback(
        Output("eda_hist_plot", "figure"),
        Input("eda_hist_dropdown", "value")
    )
    def update_hist(col):
        if not col or col not in df.columns:
            return empty_fig("hist", "Select a numeric column")

        try:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                return empty_fig("hist", f"No valid numeric data in {col}")

            fig = px.histogram(x=s, nbins=30, title=f"Histogram of {col}")
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(title=col)
            fig.update_yaxes(title="count")
            return _style(fig)
        except Exception as e:
            return empty_fig("hist", f"Error plotting histogram: {e}")

    # ====== EDA: CORRELATION HEATMAP ======
    @app.callback(
        Output("eda_corr_heatmap", "figure"),
        Input("eda_hist_dropdown", "value")  # dummy trigger
    )
    def update_corr(_):
        try:
            num_df = df.select_dtypes(include=np.number)
            if num_df.empty:
                return empty_fig("heatmap", "No numeric columns for correlation")

            corr = num_df.corr(numeric_only=True).round(2)
            fig = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="Viridis",
                showscale=True,
            )
            fig.update_layout(title="Correlation Heatmap")
            return _style(fig)
        except Exception as e:
            return empty_fig("heatmap", f"Correlation error: {e}")

    # ====== EDA: TRENDS OVER TIME ======
    @app.callback(
        Output("eda_time_plot", "figure"),
        Input("eda_time_dropdown", "value")
    )
    def update_time(col):
        if not col or col not in df.columns:
            return empty_fig("line", "Select a date column")

        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            dfg = (
                pd.DataFrame({col: dt})
                .dropna()
                .assign(day=lambda d: d[col].dt.to_period("D").dt.to_timestamp())
                .groupby("day")
                .size()
                .reset_index(name="count")
            )
            if dfg.empty:
                return empty_fig("line", f"No valid dates in {col}")

            fig = px.line(dfg, x="day", y="count", markers=True,
                          title=f"Orders Over Time ({col})")
            fig.update_xaxes(title="date")
            fig.update_yaxes(title="orders")
            return _style(fig)
        except Exception as e:
            return empty_fig("line", f"Trend error: {e}")

    # ====== HYPOTHESIS: Late vs not-Late on review_score ======
    @app.callback(
        [Output("hypothesis_plot", "figure"),
         Output("hypothesis_result", "children")],
        Input("eda_bar_dropdown", "value")  # dummy trigger
    )
    def hypothesis_test(_):
        if ("delay_flag" not in df.columns) or ("review_score" not in df.columns):
            return empty_fig("box", "Missing columns"), "Need delay_flag and review_score."

        try:
            late = df.loc[df["delay_flag"].astype(str) == "Late", "review_score"].astype(float).dropna()
            other = df.loc[df["delay_flag"].astype(str) != "Late", "review_score"].astype(float).dropna()

            if late.empty or other.empty:
                return empty_fig("box", "Not enough data to test"), "Not enough data in one of the groups."

            # unpack tuple: (t_stat, p_val, dfree)
            t_stat, p_val, dfree = sm.stats.ttest_ind(late, other, alternative="two-sided")

            fig = px.box(df, x="delay_flag", y="review_score", points="all",
                         title="Review Score vs Delivery Status")
            _style(fig)

            verdict = "Different" if p_val < 0.05 else "No significant difference"
            result = f"T-stat: {t_stat:.2f} | P-value: {p_val:.4f} → {verdict}."
            return fig, result
        except Exception as e:
            return empty_fig("box", "Hypothesis test error"), f"Error: {e}"

    # ====== MODELING: Predict 'Late' ======
    @app.callback(
        [Output("model_feature_importance", "figure"),
         Output("model_accuracy", "children")],
        Input("eda_bar_dropdown", "value")  # dummy trigger
    )
    def run_model(_):
        needed = ["delay_flag", "shipping_time", "total_price", "total_freight", "review_score"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return empty_fig("bar", "Missing columns for model"), f"Missing: {', '.join(missing)}"

        try:
            data = df[needed].copy()
            y = (data["delay_flag"].astype(str) == "Late").astype(int)

            X = data[["shipping_time", "total_price", "total_freight", "review_score"]].apply(
                pd.to_numeric, errors="coerce"
            )
            d = pd.concat([X, y], axis=1).dropna()
            X, y = d[X.columns], d[y.name]

            if y.nunique() < 2 or X.empty:
                return empty_fig("bar", "Model cannot train"), "Target has one class or no valid rows."

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.coef_[0]
            }).sort_values("Importance", ascending=False)

            fig = px.bar(coef_df, x="Feature", y="Importance",
                         title="Logistic Regression Coefficients")
            return _style(fig), f"Model Accuracy: {acc*100:.2f}%"
        except Exception as e:
            return empty_fig("bar", "Model error"), f"Error training model: {e}"
