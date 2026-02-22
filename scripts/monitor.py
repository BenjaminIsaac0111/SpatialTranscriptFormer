import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import argparse
import os

import flask
import glob

# Set up argument parsing for the target run directory
parser = argparse.ArgumentParser(description="Real-time Training Monitor")
parser.add_argument(
    "--run-dir",
    type=str,
    required=True,
    help="Path to the experiment run directory containing training_log.csv",
)
parser.add_argument(
    "--port", type=int, default=8050, help="Port to host the dashboard on"
)
parser.add_argument(
    "--interval", type=int, default=5000, help="Log polling interval in milliseconds"
)
args = parser.parse_args()

log_path = os.path.join(args.run_dir, "training_log.csv")

# Configure Flask to serve images from the run directory
server = flask.Flask(__name__)


@server.route("/images/<path:filename>")
def serve_image(filename):
    return flask.send_from_directory(os.path.abspath(args.run_dir), filename)


# Initialize Dash application
app = dash.Dash(
    __name__,
    server=server,
    title=f"Training Monitor - {os.path.basename(args.run_dir)}",
)

# Define application layout
app.layout = html.Div(
    style={"fontFamily": "sans-serif", "padding": "20px"},
    children=[
        html.H1(f"Real-time Training Monitor: {os.path.basename(args.run_dir)}"),
        html.Div(
            id="last-updated",
            style={"color": "gray", "fontStyle": "italic", "marginBottom": "10px"},
        ),
        html.Button(
            "Pause Updates",
            id="pause-button",
            n_clicks=0,
            style={
                "marginBottom": "20px",
                "padding": "10px",
                "fontSize": "16px",
                "cursor": "pointer",
                "backgroundColor": "#f0f0f0",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
            },
        ),
        html.Div(
            [
                html.Label(
                    "Smoothing Window (Epochs):",
                    style={"fontWeight": "bold", "marginRight": "10px"},
                ),
                dcc.Slider(
                    id="smoothing-slider",
                    min=1,
                    max=50,
                    step=1,
                    value=1,
                    marks={1: "1 (None)", 10: "10", 25: "25", 50: "50"},
                ),
            ],
            style={
                "marginBottom": "20px",
                "padding": "10px",
                "backgroundColor": "#f9f9f9",
                "borderRadius": "5px",
            },
        ),
        html.Div(
            [
                # Graph for Losses
                html.Div(
                    [dcc.Graph(id="live-loss-graph", animate=False)],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                # Graph for Metrics
                html.Div(
                    [dcc.Graph(id="live-metric-graph", animate=False)],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.H3("Latest Inference Plot (Truth vs Pred)"),
                html.Div(id="image-container"),
            ],
            style={
                "marginTop": "40px",
                "textAlign": "center",
                "backgroundColor": "#1a1a2e",
                "padding": "20px",
                "borderRadius": "10px",
            },
        ),
        # Hidden interval component for polling
        dcc.Interval(
            id="interval-component",
            interval=args.interval,  # in milliseconds
            n_intervals=0,
            disabled=False,
        ),
    ],
)


@app.callback(
    Output("image-container", "children"), [Input("interval-component", "n_intervals")]
)
def update_image(n):
    search_pattern = os.path.join(args.run_dir, "*.png")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return html.P(
            "No inference plots found yet. Make sure to run training with --plot-pathways.",
            style={"color": "red"},
        )

    # Get the newest file
    latest_file = max(list_of_files, key=os.path.getmtime)
    filename = os.path.basename(latest_file)

    # Force reload by appending modifying timestamp query
    mtime = os.path.getmtime(latest_file)
    url = f"/images/{filename}?t={mtime}"

    return html.Img(src=url, style={"maxWidth": "100%", "height": "auto"})


@app.callback(
    [
        Output("live-loss-graph", "figure"),
        Output("live-metric-graph", "figure"),
        Output("last-updated", "children"),
    ],
    [Input("interval-component", "n_intervals"), Input("smoothing-slider", "value")],
)
def update_graphs(n, smoothing_window):
    if not os.path.exists(log_path):
        return (
            dash.no_update,
            dash.no_update,
            "Waiting for training_log.csv to be created...",
        )

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        return dash.no_update, dash.no_update, f"Error reading log file: {str(e)}"

    if df.empty or "epoch" not in df.columns:
        return (
            dash.no_update,
            dash.no_update,
            "Log file is empty or missing 'epoch' column.",
        )

    # Subplot 1: Losses
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    loss_traces = []
    for col in loss_cols:
        y_data = df[col]
        if smoothing_window and smoothing_window > 1:
            y_data = y_data.rolling(window=smoothing_window, min_periods=1).mean()

        loss_traces.append(go.Scatter(x=df["epoch"], y=y_data, mode="lines", name=col))

    loss_layout = go.Layout(
        title="Training vs Validation Loss",
        xaxis=dict(title="Epoch"),
        yaxis=dict(
            title="Loss", type="log"
        ),  # Log scale often helps with early MSE spikes
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Subplot 2: Interpretability Metrics
    metric_cols = [c for c in df.columns if c not in loss_cols and c != "epoch"]
    metric_traces = []
    for col in metric_cols:
        y_data = df[col]
        if smoothing_window and smoothing_window > 1:
            y_data = y_data.rolling(window=smoothing_window, min_periods=1).mean()

        metric_traces.append(
            go.Scatter(x=df["epoch"], y=y_data, mode="lines", name=col)
        )

    metric_layout = go.Layout(
        title="Interpretability Metrics (MAE, PCC, Correlation)",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Score / Error"),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    last_epoch = df["epoch"].iloc[-1]
    update_text = f"Last updated: Epoch {last_epoch} (Polled automatically)"

    return (
        {"data": loss_traces, "layout": loss_layout},
        {"data": metric_traces, "layout": metric_layout},
        update_text,
    )


@app.callback(
    [
        Output("interval-component", "disabled"),
        Output("pause-button", "children"),
        Output("pause-button", "style"),
    ],
    [Input("pause-button", "n_clicks")],
)
def toggle_pause(n_clicks):
    base_style = {
        "marginBottom": "20px",
        "padding": "10px",
        "fontSize": "16px",
        "cursor": "pointer",
        "borderRadius": "5px",
        "border": "1px solid #ccc",
    }
    if n_clicks % 2 == 1:
        # Paused state
        active_style = {
            **base_style,
            "backgroundColor": "#ffcccc",
            "borderColor": "#ff0000",
        }
        return True, "Resume Updates", active_style
    # Active state
    active_style = {**base_style, "backgroundColor": "#f0f0f0"}
    return False, "Pause Updates", active_style


if __name__ == "__main__":
    print(f"Tracking log at: {log_path}")
    print(f"Starting dashboard on http://127.0.0.1:{args.port}/")
    # Turn off debug to prevent double-reloading the data parser during polling
    app.run(debug=False, port=args.port)
