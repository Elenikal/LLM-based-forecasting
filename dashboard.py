"""
dashboard.py
============
Plotly Dash dashboard for:
  "Reading the Fed: LLM-Augmented Forecasting of U.S. GDP and Financial Indicators"

Usage:
  python dashboard.py
  Then open http://127.0.0.1:8050

Data requirements (in ./outputs/ folder):
  results_main.csv
  results_subperiod.csv
  llm_scores.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load data ──────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "outputs"

try:
    main_df   = pd.read_csv(DATA / "results_main.csv")
    sub_df    = pd.read_csv(DATA / "results_subperiod.csv")
    scores_df = pd.read_csv(DATA / "llm_scores.csv", index_col=0)
except FileNotFoundError as e:
    raise SystemExit(f"\nMissing data file: {e}\nRun the pipeline first: python run.py --live\n")

# ── Constants ──────────────────────────────────────────────────────────────────
VAR_LABELS = {
    "GDPC1_gr":     "Real GDP growth",
    "CPIAUCSL_yoy": "CPI inflation",
    "PCEPILFE_yoy": "Core PCE inflation",
    "UNRATE":       "Unemployment rate",
    "FEDFUNDS":     "Federal funds rate",
    "T10Y2Y":       "10Y-2Y spread",
    "BAA10Y":       "BAA-10Y spread",
    "VIXCLS_log":   "log(VIX)",
}

REGIME_BANDS = [
    {"x0": 2008.75, "x1": 2009.5,  "color": "rgba(231,76,60,0.10)",  "label": "GFC"},
    {"x0": 2020.0,  "x1": 2020.5,  "color": "rgba(241,196,15,0.18)", "label": "COVID"},
    {"x0": 2021.5,  "x1": 2023.5,  "color": "rgba(230,126,34,0.12)", "label": "Inflation surge"},
]

# Colour palette — pure hex only, no alpha appended
COLORS_FOMC = ["#1a4e8c", "#c0392b", "#0f6e56", "#d68910", "#533fad"]
COLORS_BB   = ["#5b9bd5", "#e07070", "#3fbf9f", "#f5b942", "#a080e0"]

def hex_to_rgba(h, a=0.10):
    """Convert #rrggbb to rgba(r,g,b,a) — works in Plotly 6."""
    r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"

def q2num(q):
    yr, qt = str(q).split("Q")
    return int(yr) + (int(qt) - 1) / 4

scores_df["_x"] = [q2num(q) for q in scores_df.index]

# ── Shared styles ──────────────────────────────────────────────────────────────
CARD = {
    "background":   "#ffffff",
    "borderRadius": "10px",
    "padding":      "20px 24px",
    "boxShadow":    "0 1px 6px rgba(0,0,0,0.08)",
    "marginBottom": "20px",
}

def kpi_card(value, label, color):
    return html.Div(style={**CARD, "borderTop": f"4px solid {color}", "padding": "16px 20px"}, children=[
        html.Div(value, style={"fontSize": "26px", "fontWeight": "bold",
                               "color": color, "lineHeight": "1.1"}),
        html.Div(label, style={"fontSize": "12px", "color": "#555",
                               "marginTop": "4px", "lineHeight": "1.4"}),
    ])

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Reading the Fed")
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Georgia, serif", "background": "#f4f6f9", "minHeight": "100vh"},
    children=[

        # Header
        html.Div(style={
            "background": "linear-gradient(135deg, #1a4e8c 0%, #0f6e56 100%)",
            "padding": "28px 40px 22px", "marginBottom": "28px",
        }, children=[
            html.H1("Reading the Fed",
                    style={"color": "white", "margin": "0 0 4px",
                           "fontSize": "30px", "fontWeight": "bold"}),
            html.P("LLM-Augmented Forecasting of U.S. GDP and Financial Indicators",
                   style={"color": "rgba(255,255,255,0.85)", "margin": "0",
                          "fontSize": "14px", "fontStyle": "italic"}),
            html.P("Out-of-sample: 2020Q1-2024Q4  |  FOMC minutes + Beige Books -> Claude API -> BVAR",
                   style={"color": "rgba(255,255,255,0.6)", "margin": "5px 0 0",
                          "fontSize": "12px"}),
        ]),

        html.Div(style={"maxWidth": "1360px", "margin": "0 auto",
                        "padding": "0 28px 50px"}, children=[

            # KPI row
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "repeat(4, 1fr)",
                            "gap": "16px", "marginBottom": "24px"}, children=[
                kpi_card("16 / 16", "Variables where BVAR+LLM beats BVAR",      "#1a4e8c"),
                kpi_card("-50%",    "Max RMSE gain (GDP, inflation surge, h=1)", "#0f6e56"),
                kpi_card("10 / 16", "Statistically significant improvements",    "#533fad"),
                kpi_card("69 qtrs", "Quarters with LLM-scored Fed documents",    "#d68910"),
            ]),

            # Tab panel
            html.Div(style=CARD, children=[
                dcc.Tabs(id="tabs", value="scores",
                         colors={"border": "#dee2e6", "primary": "#1a4e8c",
                                 "background": "#f9fafb"},
                         children=[
                    dcc.Tab(label="LLM Scores Over Time", value="scores",
                            style={"fontFamily": "Georgia, serif", "fontSize": "13px"}),
                    dcc.Tab(label="RMSE Comparison",      value="rmse",
                            style={"fontFamily": "Georgia, serif", "fontSize": "13px"}),
                    dcc.Tab(label="Sub-Period Heatmap",   value="heatmap",
                            style={"fontFamily": "Georgia, serif", "fontSize": "13px"}),
                ]),
                html.Div(id="tab-content", style={"marginTop": "16px"}),
            ]),
        ]),
    ]
)


# ── Tab router ────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "scores":  return scores_tab()
    if tab == "rmse":    return rmse_tab()
    if tab == "heatmap": return heatmap_tab()
    return html.Div()


# ── TAB 1: LLM Scores ─────────────────────────────────────────────────────────
def scores_tab():
    return html.Div([
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                        "marginBottom": "14px", "alignItems": "center"}, children=[
            html.Label("Source:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.RadioItems(
                id="score-source",
                options=[
                    {"label": "FOMC minutes", "value": "fomc"},
                    {"label": "Beige Book",   "value": "bb"},
                    {"label": "Both",         "value": "both"},
                ],
                value="fomc", inline=True,
                style={"fontSize": "13px"},
                inputStyle={"marginRight": "5px"},
                labelStyle={"marginRight": "18px"},
            ),
        ]),
        dcc.Graph(id="scores-chart", style={"height": "560px"},
                  config={"displayModeBar": False}),
        html.P(
            "Each panel shows one LLM sentiment dimension extracted by Claude. "
            "Red shading = GFC, yellow = COVID shock, orange = inflation surge.",
            style={"fontSize": "11px", "color": "#888", "marginTop": "8px",
                   "fontStyle": "italic"}
        ),
    ])


@app.callback(Output("scores-chart", "figure"), Input("score-source", "value"))
def update_scores(source):
    prefix_map = {
        "fomc": ["fomc_minutes_"],
        "bb":   ["beige_book_"],
        "both": ["fomc_minutes_", "beige_book_"],
    }
    prefixes  = prefix_map.get(source, ["fomc_minutes_"])
    dim_names = ["growth_expectations", "inflation_concern", "labor_market",
                 "credit_conditions",   "policy_uncertainty"]
    titles    = ["Growth expectations", "Inflation concern", "Labor market",
                 "Credit conditions",   "Policy uncertainty"]

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.05)

    x = scores_df["_x"].values

    for row, (dim, title) in enumerate(zip(dim_names, titles), start=1):
        xref = "x" if row == 1 else f"x{row}"
        yref = "y" if row == 1 else f"y{row}"

        # Regime shading on every panel
        for band in REGIME_BANDS:
            fig.add_shape(
                type="rect",
                xref=xref, yref=yref,
                x0=band["x0"], x1=band["x1"], y0=-2.4, y1=2.4,
                fillcolor=band["color"], line_width=0, layer="below",
            )
            fig.add_annotation(
                x=(band["x0"] + band["x1"]) / 2, y=2.05,
                xref=xref, yref=yref,
                text=band["label"], showarrow=False,
                font={"size": 8, "color": "#aaa"},
            )

        for pf in prefixes:
            col_name = pf + dim
            if col_name not in scores_df.columns:
                continue
            clr_list = COLORS_FOMC if "fomc" in pf else COLORS_BB
            clr      = clr_list[row - 1]
            fill_clr = hex_to_rgba(clr, 0.10)
            src      = "FOMC" if "fomc" in pf else "BB"

            # Mask to only available (non-NaN) quarters — critical for Beige Book
            y_all  = scores_df[col_name].values
            mask   = ~pd.isna(y_all.astype(float))
            x_plot = x[mask]
            y_plot = y_all[mask].astype(float)

            fig.add_trace(go.Scatter(
                x=x_plot, y=y_plot,
                mode="lines+markers",
                name=f"{src}",
                line={"color": clr, "width": 1.8},
                marker={"size": 3, "color": clr},
                fill="tozeroy",
                fillcolor=fill_clr,
                connectgaps=False,
                showlegend=False,   # subplot titles label each panel — no legend needed
            ), row=row, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="#bbb",
                      line_width=0.8, row=row, col=1)
        fig.update_yaxes(range=[-2.4, 2.4], tickvals=[-2, -1, 0, 1, 2],
                         tickfont={"size": 9}, row=row, col=1)

    # Set x-axis range to start where the selected source has data
    if source == "bb":
        x_start = 2011.0   # Beige Book available from 2011Q1
    else:
        x_start = 2007.5   # FOMC minutes from 2007Q4

    year_ticks = [y for y in range(2008, 2025, 2) if y >= x_start]
    fig.update_xaxes(tickvals=year_ticks,
                     ticktext=[str(y) for y in year_ticks],
                     range=[x_start, 2025.0])

    fig.update_layout(
        height=600,
        margin={"l": 55, "r": 20, "t": 40, "b": 30},
        plot_bgcolor="white", paper_bgcolor="white",
        font={"family": "Georgia, serif", "size": 10},
        showlegend=False,
        hovermode="x unified",
    )
    return fig


# ── TAB 2: RMSE Comparison ────────────────────────────────────────────────────
def rmse_tab():
    return html.Div([
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                        "marginBottom": "14px", "alignItems": "center"}, children=[
            html.Label("Horizon:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.RadioItems(
                id="rmse-horizon",
                options=[
                    {"label": "h = 1 quarter",  "value": 1},
                    {"label": "h = 2 quarters", "value": 2},
                    {"label": "Both",            "value": 0},
                ],
                value=0, inline=True,
                style={"fontSize": "13px"},
                inputStyle={"marginRight": "5px"},
                labelStyle={"marginRight": "18px"},
            ),
        ]),
        dcc.Graph(id="rmse-chart", style={"height": "500px"},
                  config={"displayModeBar": False}),
        html.P(
            "RMSE ratios vs AR(4) baseline — below 1.0 beats the AR. "
            "Dark green = BVAR+LLM wins over BVAR. "
            "Stars = DM significance (* 10%, ** 5%, *** 1%).",
            style={"fontSize": "11px", "color": "#888", "marginTop": "8px",
                   "fontStyle": "italic"}
        ),
    ])


@app.callback(Output("rmse-chart", "figure"), Input("rmse-horizon", "value"))
def update_rmse(h_val):
    vars_ord = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy", "UNRATE",
                "FEDFUNDS", "T10Y2Y", "BAA10Y", "VIXCLS_log"]
    labels   = [VAR_LABELS[v] for v in vars_ord]
    horizons = [1, 2] if h_val == 0 else [h_val]

    fig = make_subplots(rows=1, cols=len(horizons),
                        subplot_titles=[f"h = {h}" for h in horizons],
                        shared_yaxes=False)

    for col_idx, h in enumerate(horizons, start=1):
        sub    = main_df[main_df["horizon"] == h].set_index("variable")
        bvar_r = [float(sub.loc[v, "BVAR_ratio"])     if v in sub.index else None for v in vars_ord]
        llm_r  = [float(sub.loc[v, "BVAR_LLM_ratio"]) if v in sub.index else None for v in vars_ord]
        sigs   = [str(sub.loc[v, "BVAR_LLM_dm_sig"])  if v in sub.index else ""   for v in vars_ord]

        llm_colors = [
            "#0f6e56" if (lr is not None and br is not None and lr < br) else "#a8d5c4"
            for lr, br in zip(llm_r, bvar_r)
        ]

        fig.add_trace(go.Bar(
            x=labels, y=bvar_r, name="BVAR / AR",
            marker_color="#1a4e8c", opacity=0.75,
            offsetgroup="bvar", showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        fig.add_trace(go.Bar(
            x=labels, y=llm_r, name="BVAR+LLM / AR",
            marker_color=llm_colors,
            offsetgroup="llm",
            text=[s if s not in ("", "nan") else "" for s in sigs],
            textposition="outside",
            textfont={"color": "#c0392b", "size": 11},
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        fig.add_hline(y=1.0, line_dash="dash", line_color="black",
                      line_width=1.5, row=1, col=col_idx)

        fig.update_yaxes(title_text="RMSE ratio (vs AR)", rangemode="tozero",
                         row=1, col=col_idx)
        fig.update_xaxes(tickangle=-35, tickfont={"size": 9}, row=1, col=col_idx)

    fig.update_layout(
        barmode="group",
        height=520,
        margin={"l": 55, "r": 20, "t": 50, "b": 100},
        plot_bgcolor="white", paper_bgcolor="white",
        font={"family": "Georgia, serif", "size": 10},
        legend={"orientation": "h", "y": 1.08, "font": {"size": 10}},
    )
    return fig


# ── TAB 3: Sub-Period Heatmap ─────────────────────────────────────────────────
def heatmap_tab():
    return html.Div([
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                        "marginBottom": "14px", "alignItems": "center"}, children=[
            html.Label("Horizon:", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.RadioItems(
                id="heat-horizon",
                options=[{"label": "h = 1", "value": 1},
                         {"label": "h = 2", "value": 2}],
                value=1, inline=True,
                style={"fontSize": "13px"},
                inputStyle={"marginRight": "5px"},
                labelStyle={"marginRight": "18px"},
            ),
            html.Label("vs. baseline:", style={"fontWeight": "bold", "fontSize": "13px",
                                               "marginLeft": "20px"}),
            dcc.RadioItems(
                id="heat-metric",
                options=[
                    {"label": "BVAR",  "value": "bvar"},
                    {"label": "AR(4)", "value": "ar"},
                ],
                value="bvar", inline=True,
                style={"fontSize": "13px"},
                inputStyle={"marginRight": "5px"},
                labelStyle={"marginRight": "18px"},
            ),
        ]),
        dcc.Graph(id="heat-chart", style={"height": "440px"},
                  config={"displayModeBar": False}),
        html.P(
            "Cell values = (BVAR+LLM RMSE - baseline RMSE) / baseline RMSE x 100. "
            "Green = LLM improves accuracy; red = worsens. "
            "Sub-periods: COVID (2020Q1-2021Q2), Inflation surge (2021Q3-2023Q2), "
            "Soft landing (2023Q3-2024Q4).",
            style={"fontSize": "11px", "color": "#888", "marginTop": "8px",
                   "fontStyle": "italic"}
        ),
    ])


@app.callback(
    Output("heat-chart", "figure"),
    Input("heat-horizon", "value"),
    Input("heat-metric",  "value"),
)
def update_heatmap(h, metric):
    periods  = ["COVID shock", "Inflation surge", "Soft landing"]
    vars_ord = ["GDPC1_gr", "CPIAUCSL_yoy", "PCEPILFE_yoy", "UNRATE",
                "FEDFUNDS", "T10Y2Y", "BAA10Y", "VIXCLS_log"]
    labels   = [VAR_LABELS[v] for v in vars_ord]
    baseline = "BVAR" if metric == "bvar" else "AR"

    matrix   = np.full((len(vars_ord), len(periods)), np.nan)
    ann_text = [["" for _ in periods] for _ in vars_ord]

    for j, period in enumerate(periods):
        psub      = sub_df[(sub_df["period"] == period) & (sub_df["horizon"] == h)]
        base_rmse = psub[psub["model"] == baseline].set_index("variable")["rmse"]
        llm_rmse  = psub[psub["model"] == "BVAR_LLM"].set_index("variable")["rmse"]
        for i, v in enumerate(vars_ord):
            if v in base_rmse.index and v in llm_rmse.index:
                bv = float(base_rmse[v])
                lm = float(llm_rmse[v])
                if bv > 0:
                    pct = (lm - bv) / bv * 100
                    matrix[i, j] = pct
                    ann_text[i][j] = f"{pct:+.0f}%"

    fig = go.Figure(go.Heatmap(
        z=matrix, x=periods, y=labels,
        text=ann_text,
        texttemplate="%{text}",
        textfont={"size": 12, "family": "Georgia, serif"},
        colorscale=[
            [0.0,  "#1a8c4e"],
            [0.45, "#d5f5e3"],
            [0.5,  "#fdfefe"],
            [0.55, "#fde8d8"],
            [1.0,  "#c0392b"],
        ],
        zmid=0, zmin=-65, zmax=25,
        colorbar={
            "title": {"text": "% RMSE change", "font": {"size": 11}},
            "tickfont": {"size": 9},
            "thickness": 16,
        },
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        title={
            "text": f"BVAR+LLM vs {baseline} — RMSE change by sub-period (h={h})",
            "font": {"size": 13, "family": "Georgia, serif"},
            "x": 0.5, "xanchor": "center",
        },
        height=460,
        margin={"l": 160, "r": 100, "t": 55, "b": 60},
        font={"family": "Georgia, serif", "size": 10},
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Reading the Fed — Research Dashboard")
    print("  Open: http://127.0.0.1:8050")
    print("="*55 + "\n")
    import os

    #app.run(debug=False, port=8050)

    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))