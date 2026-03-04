import dash
from dash import dcc, html


def create_layout(args):
    """Generates the main dashboard layout."""
    import os

    run_dir = getattr(args, "run_dir", None)
    run_name = os.path.basename(run_dir) if run_dir else "Multiple Runs"

    return html.Div(
        style={
            "fontFamily": "Inter, Roboto, sans-serif",
            "padding": "20px",
            "backgroundColor": "#0f172a",
            "color": "#f1f5f9",
            "minHeight": "100vh",
        },
        children=[
            # Header
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "20px",
                },
                children=[
                    html.H1(
                        (
                            f"Training Monitor: {run_name}"
                            if getattr(args, "run_dir", None)
                            else "Training Monitor: Compare Runs"
                        ),
                        style={"color": "#38bdf8", "margin": "0"},
                    ),
                    html.Div(
                        id="last-updated",
                        style={"color": "#94a3b8", "fontStyle": "italic"},
                    ),
                ],
            ),
            # Controls
            html.Div(
                style={
                    "display": "flex",
                    "gap": "20px",
                    "padding": "20px",
                    "backgroundColor": "#1e293b",
                    "borderRadius": "12px",
                    "marginBottom": "20px",
                    "alignItems": "center",
                    "flexWrap": "wrap",  # Allow wrapping if many controls
                },
                children=[
                    html.Button(
                        "Pause Updates",
                        id="pause-button",
                        n_clicks=0,
                        style={
                            "padding": "10px 20px",
                            "fontSize": "14px",
                            "cursor": "pointer",
                            "backgroundColor": "#2563eb",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontWeight": "bold",
                            "transition": "background-color 0.2s",
                        },
                    ),
                    html.Div(
                        style={"flex": "2", "minWidth": "250px"},
                        children=[
                            html.Label(
                                "Select Runs:",
                                style={
                                    "fontWeight": "bold",
                                    "color": "#cbd5e1",
                                    "display": "block",
                                    "marginBottom": "5px",
                                },
                            ),
                            dcc.Dropdown(
                                id="run-selector",
                                options=[],  # Populated by callback
                                value=[],
                                multi=True,
                                style={"color": "black"},
                                placeholder="Select runs to compare...",
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1", "minWidth": "200px"},
                        children=[
                            html.Label(
                                "Smoothing (Epochs):",
                                style={
                                    "fontWeight": "bold",
                                    "color": "#cbd5e1",
                                    "display": "block",
                                    "marginBottom": "5px",
                                },
                            ),
                            dcc.Slider(
                                id="smoothing-slider",
                                min=1,
                                max=50,
                                step=1,
                                value=1,
                                marks={
                                    i: {"label": str(i), "style": {"color": "#cbd5e1"}}
                                    for i in [1, 10, 25, 50]
                                },
                            ),
                        ],
                    ),
                    html.Button(
                        "Export Data",
                        id="export-button",
                        n_clicks=0,
                        style={
                            "padding": "10px 20px",
                            "fontSize": "14px",
                            "cursor": "pointer",
                            "backgroundColor": "#10b981",  # Emerald
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontWeight": "bold",
                        },
                    ),
                    dcc.Download(
                        id="download-data"
                    ),  # Component to handle the actual file download
                ],
            ),
            # KPI Cards (Top Row)
            html.Div(
                id="kpi-cards",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                    "gap": "20px",
                    "marginBottom": "20px",
                },
                # Children populated by callback
            ),
            # Charts
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px",
                    "marginBottom": "30px",
                },
                children=[
                    dcc.Graph(
                        id="live-loss-graph",
                        animate=False,
                        style={
                            "height": "400px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                    dcc.Graph(
                        id="live-pcc-graph",
                        animate=False,
                        style={
                            "height": "400px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                    dcc.Graph(
                        id="live-variance-graph",
                        animate=False,
                        style={
                            "height": "400px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                    dcc.Graph(
                        id="live-lr-graph",
                        animate=False,
                        style={
                            "height": "400px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                ],
            ),
            # Hardware Resource Charts
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                    "gap": "20px",
                    "marginBottom": "30px",
                },
                children=[
                    dcc.Graph(
                        id="live-cpu-graph",
                        animate=False,
                        style={
                            "height": "300px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                    dcc.Graph(
                        id="live-ram-graph",
                        animate=False,
                        style={
                            "height": "300px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                    dcc.Graph(
                        id="live-gpu-graph",
                        animate=False,
                        style={
                            "height": "300px",
                            "borderRadius": "12px",
                            "overflow": "hidden",
                        },
                    ),
                ],
            ),
            # Image Preview Section
            html.Div(
                style={
                    "backgroundColor": "#1e293b",
                    "padding": "30px",
                    "borderRadius": "12px",
                },
                children=[
                    html.H2(
                        "Spatial Predictions",
                        style={"color": "#e2e8f0", "marginTop": "0"},
                    ),
                    # Controls for image
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "20px",
                            "marginBottom": "20px",
                        },
                        children=[
                            html.Div(
                                style={"flex": "1"},
                                children=[
                                    html.Label(
                                        "Sample ID:",
                                        style={
                                            "color": "#cbd5e1",
                                            "display": "block",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="sample-dropdown",
                                        options=[],  # Populated dynamically
                                        style={
                                            "color": "black"
                                        },  # Text color inside dropdown
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"flex": "1"},
                                children=[
                                    html.Label(
                                        "Epoch:",
                                        style={
                                            "color": "#cbd5e1",
                                            "display": "block",
                                            "marginBottom": "5px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="epoch-dropdown",
                                        options=[],  # Populated dynamically based on sample
                                        style={"color": "black"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="image-container",
                        style={
                            "textAlign": "center",
                            "minHeight": "400px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "backgroundColor": "#0f172a",
                            "borderRadius": "8px",
                        },
                    ),
                ],
            ),
            # Polling Interval
            dcc.Interval(
                id="interval-component",
                interval=args.interval,
                n_intervals=0,
                disabled=False,
            ),
        ],
    )


def create_kpi_card(title, value, subtitle=""):
    """Helper to create a stylized KPI card."""
    from dash import html

    return html.Div(
        style={
            "backgroundColor": "#1e293b",
            "padding": "20px",
            "borderRadius": "12px",
            "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
            "borderLeft": "4px solid #38bdf8",
        },
        children=[
            html.H3(
                title,
                style={
                    "margin": "0 0 10px 0",
                    "color": "#94a3b8",
                    "fontSize": "14px",
                    "textTransform": "uppercase",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "28px",
                    "fontWeight": "bold",
                    "color": "#f8fafc",
                    "marginBottom": "5px",
                },
            ),
            (
                html.Div(subtitle, style={"fontSize": "12px", "color": "#64748b"})
                if subtitle
                else None
            ),
        ],
    )
