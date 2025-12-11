import base64
import ast
import pandas as pd
from PIL import Image
from io import BytesIO
import os

from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ------------------------------------------------------
# Helper: encode image as base64 for Dash
# ------------------------------------------------------
def pil_to_base64(img_path):
    img = Image.open(img_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}", img.size


# ------------------------------------------------------
# Helpers: load results and summary
# ------------------------------------------------------
def get_results_path(baseline):
    return (
        "./baseline_results/results_answers_iter1.csv"
        if baseline
        else "./incontext_lrn_results/results_iter2.csv"
    )

def analyze_results(baseline):
    if baseline:
        import analyze_baseline
        return analyze_baseline.main(get_results_path(baseline))
    else:
        import analyze_coding
        return analyze_coding.main(get_results_path(baseline), "./incontext_lrn_results/eval_done.csv")

def process_results(df):
    df["image"] = df["image"].apply(lambda x: ','.join(x))
    return df
    
# ------------------------------------------------------
# Dash App
# ------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container(
    [
        html.H2("VLM Evaluation Results Viewer"),
        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    [
                        # -------------------------------
                        # Baseline toggle added here
                        # -------------------------------
                        dbc.RadioItems(
                            id="baseline-toggle",
                            options=[
                                {"label": "Baseline", "value": True},
                                {"label": "Coding", "value": False},
                            ],
                            value=False,
                            inline=True,
                        ),

                        html.Br(),

                        dbc.RadioItems(
                            id="mode",
                            options=[
                                {"label": "View Results", "value": "results"},
                                {"label": "View Summary", "value": "summary"},
                            ],
                            value="results",
                            inline=True,
                        ),
                        html.Br(),

                        dash_table.DataTable(
                            id="result-table",
                            data=[],
                            columns=[],
                            page_size=20,
                            row_selectable="single",
                            cell_selectable=False,
                            style_table={
                                "height": "80vh",
                                "overflowY": "scroll",
                                "width": "100%",
                                "minWidth": "100%",
                            },
                            style_cell={
                                "whiteSpace": "normal",
                                "textAlign": "left",
                                "fontSize": 14,
                                "maxWidth": "600px",
                            },
                        ),
                    ],
                    width=10,
                ),

                dbc.Col(
                    [
                        html.H4("Details"),
                        html.Div(id="details-panel"),
                    ],
                    width=2,
                ),
            ]
        ),
    ],
    fluid=True,
)

# ------------------------------------------------------
# Load data for mode + baseline toggle
# ------------------------------------------------------
@app.callback(
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Input("mode", "value"),
    Input("baseline-toggle", "value"),
)
def update_table(mode, baseline):
    summary_df, results_df = analyze_results(baseline)
    results_df = process_results(results_df)
    table_df = summary_df if mode == "summary" else results_df
    return (
        table_df.to_dict("records"),
        [{"name": c, "id": c} for c in table_df.columns],
    )

# ------------------------------------------------------
# Display clicked row details
# ------------------------------------------------------
@app.callback(
    Output("details-panel", "children"),
    Input("result-table", "selected_rows"),
    Input("mode", "value"),
    Input("baseline-toggle", "value"),
)
def display_details(selected_rows, mode, baseline):
    if mode == "summary":
        return html.Div("Select 'View Results' to see row details.")
    if not selected_rows:
        return html.Div("Click a row to display images and details.")

    _, results_df = analyze_results(baseline)
    results_df = process_results(results_df)

    idx = selected_rows[0]
    row = results_df.iloc[idx]

    cards = []
    for img_path in row["image"].split(","):
        img_b64, size = pil_to_base64(img_path)
        width, height = size
        cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(f"Image: {img_path}"),
                    dbc.CardBody(
                        [
                            html.Img(
                                src=img_b64,
                                style={"maxWidth": "100%", "height": "auto"},
                            ),
                            html.P(f"Size: {width} x {height}"),
                        ]
                    ),
                ],
                style={"marginBottom": "15px"},
            )
        )

    return html.Div(cards)


# ------------------------------------------------------
# Run
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
