import os
import dash
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import html, dcc
import plotly.graph_objs as go

from .data_access import get_training_data, get_available_images
from .layout import create_kpi_card


def register_callbacks(app, args):
    """Register all Dashboard callbacks."""

    from .data_access import get_available_runs

    @app.callback(
        [Output("run-selector", "options"), Output("run-selector", "value")],
        [Input("interval-component", "n_intervals")],
        [State("run-selector", "value")],
    )
    def update_run_selector(n, current_selected):
        runs = get_available_runs(args)
        options = [{"label": r["name"], "value": r["name"]} for r in runs]

        # If no runs selected, default to the first one
        if not current_selected and runs:
            return options, [runs[0]["name"]]
        return options, dash.no_update

    @app.callback(
        Output("download-data", "data"),
        Input("export-button", "n_clicks"),
        [State("run-selector", "value")],
        prevent_initial_call=True,
    )
    def export_data(n_clicks, selected_runs):
        data_dict = get_training_data(args, selected_runs)
        if not data_dict:
            return dash.no_update

        # Combine into one exportable CSV
        combined = []
        for run_name, df in data_dict.items():
            df_copy = df.copy()
            df_copy.insert(0, "run_name", run_name)
            combined.append(df_copy)

        final_df = pd.concat(combined, ignore_index=True)
        return dcc.send_data_frame(final_df.to_csv, "training_metrics.csv", index=False)

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
            "padding": "10px 20px",
            "fontSize": "14px",
            "cursor": "pointer",
            "border": "none",
            "borderRadius": "8px",
            "fontWeight": "bold",
            "transition": "background-color 0.2s",
        }
        if n_clicks % 2 == 1:
            active_style = {
                **base_style,
                "backgroundColor": "#ef4444",
                "color": "white",
            }
            return True, "Resume Updates", active_style

        active_style = {**base_style, "backgroundColor": "#2563eb", "color": "white"}
        return False, "Pause Updates", active_style

    @app.callback(
        [Output("sample-dropdown", "options"), Output("sample-dropdown", "value")],
        [Input("interval-component", "n_intervals"), Input("run-selector", "value")],
        [State("sample-dropdown", "value")],
    )
    def update_sample_dropdown(n, selected_runs, current_val):
        images = get_available_images(args, selected_runs)
        samples = sorted(list(set(img["sample"] for img in images)))
        options = [{"label": s, "value": s} for s in samples]

        default_val = (
            current_val if current_val in samples else (samples[0] if samples else None)
        )
        return options, default_val

    @app.callback(
        [Output("epoch-dropdown", "options"), Output("epoch-dropdown", "value")],
        [
            Input("sample-dropdown", "value"),
            Input("interval-component", "n_intervals"),
            Input("run-selector", "value"),
        ],
        [State("epoch-dropdown", "value")],
    )
    def update_epoch_dropdown(selected_sample, n, selected_runs, current_epoch):
        if not selected_sample:
            return [], None

        images = get_available_images(args, selected_runs)

        # Since we might compare across runs, an epoch might be available for multiple runs
        epochs = sorted(
            list(
                set(img["epoch"] for img in images if img["sample"] == selected_sample)
            ),
            reverse=True,
        )
        options = [{"label": f"Epoch {e}", "value": e} for e in epochs]

        default_val = (
            current_epoch
            if current_epoch in epochs
            else (epochs[0] if epochs else None)
        )
        return options, default_val

    @app.callback(
        Output("image-container", "children"),
        [
            Input("sample-dropdown", "value"),
            Input("epoch-dropdown", "value"),
            Input("run-selector", "value"),
        ],
    )
    def display_image(sample, epoch, selected_runs):
        if not sample or not epoch:
            return html.Div(
                "Select a sample and epoch to view predictions.",
                style={"color": "#64748b", "padding": "50px"},
            )

        images = get_available_images(args, selected_runs)

        # We might have matches from multiple runs. We'll show them side by side.
        matches = [
            img for img in images if img["sample"] == sample and img["epoch"] == epoch
        ]

        if not matches:
            return html.Div("Image not found.", style={"color": "#ef4444"})

        # If it's a multi-run directory, we need to map the image path correctly to a Flask route.
        # But our Flask route expects only filenames and serves from args.run_dir.
        # This will be tricky if we have an `--runs-dir`. We will need to update the server route.
        # For now, we'll construct the HTML to expect a new API.

        children = []
        for match in matches:
            url = f"/images/{match['run_name']}/{match['filename']}?t={match['mtime']}"
            children.append(
                html.Div(
                    [
                        html.H4(
                            match["run_name"],
                            style={"color": "#38bdf8", "textAlign": "center"},
                        ),
                        html.Img(
                            src=url,
                            style={
                                "maxWidth": "100%",
                                "height": "auto",
                                "objectFit": "contain",
                                "borderRadius": "8px",
                                "marginBottom": "20px",
                            },
                        ),
                    ],
                    style={"flex": "1", "minWidth": "300px", "padding": "10px"},
                )
            )

        return html.Div(
            children=children,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "20px",
                "width": "100%",
                "justifyContent": "center",
            },
        )

    def _make_traces(data_dict, cols, smoothing_window):
        """Create Plotly traces for the given columns across multiple runs."""
        traces = []

        # Color palette for runs
        colors = ["#38bdf8", "#fb7185", "#a3e635", "#c084fc", "#facc15", "#2dd4bf"]

        for r_idx, (run_name, df) in enumerate(data_dict.items()):
            color = colors[r_idx % len(colors)]

            for c_idx, col in enumerate(cols):
                if col not in df.columns:
                    continue
                y_data = df[col].dropna()
                epochs = df.loc[y_data.index, "epoch"]
                if smoothing_window and smoothing_window > 1:
                    y_data = y_data.rolling(
                        window=smoothing_window, min_periods=1
                    ).mean()

                # Use solid lines for primary metric, dashed for secondary if multiple cols
                dash_style = "solid" if c_idx == 0 else "dash"

                label_name = (
                    f"{run_name}"
                    if len(cols) == 1
                    else f"{run_name} ({col.replace('_', ' ').title()})"
                )

                traces.append(
                    go.Scatter(
                        x=epochs,
                        y=y_data,
                        mode="lines",
                        name=label_name,
                        line=dict(width=2.5, color=color, dash=dash_style),
                        showlegend=True,
                    )
                )
        return traces

    @app.callback(
        [
            Output("live-loss-graph", "figure"),
            Output("live-pcc-graph", "figure"),
            Output("live-variance-graph", "figure"),
            Output("live-lr-graph", "figure"),
            Output("live-cpu-graph", "figure"),
            Output("live-ram-graph", "figure"),
            Output("live-gpu-graph", "figure"),
            Output("last-updated", "children"),
            Output("kpi-cards", "children"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("smoothing-slider", "value"),
            Input("run-selector", "value"),
        ],
    )
    def update_metrics(n, smoothing_window, selected_runs):
        empty = dash.no_update
        data_dict = get_training_data(args, selected_runs)

        if not data_dict:
            return (
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                "Waiting for training data...",
                [html.Div("No data yet or no runs selected", style={"color": "white"})],
            )

        # Common layout styles for dark mode charts
        layout_defaults = dict(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#1e293b",
            font=dict(color="#cbd5e1"),
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        # Separate base yaxis style to mix in
        yaxis_base = dict(gridcolor="#334155", zerolinecolor="#334155")

        # Losses
        loss_cols = ["train_loss", "val_loss"]
        loss_fig = go.Figure(data=_make_traces(data_dict, loss_cols, smoothing_window))
        loss_fig.update_layout(
            title="Loss Landscape",
            yaxis_type="log",
            yaxis_title="Loss (Log Scale)",
            yaxis=yaxis_base,
            xaxis_title="Epoch",
            **layout_defaults,
        )

        # Correlation / Errors
        corr_cols = ["val_pcc", "val_mae"]
        pcc_fig = go.Figure(data=_make_traces(data_dict, corr_cols, smoothing_window))
        pcc_fig.update_layout(
            title="Validation Metrics",
            yaxis_title="Score",
            yaxis=yaxis_base,
            xaxis_title="Epoch",
            **layout_defaults,
        )

        # Variance
        var_cols = ["pred_variance"]
        var_fig = go.Figure(data=_make_traces(data_dict, var_cols, smoothing_window))
        var_fig.update_layout(
            title="Prediction Variance",
            yaxis_type="log",
            yaxis_title="Variance",
            yaxis=yaxis_base,
            xaxis_title="Epoch",
            **layout_defaults,
        )

        # Learning Rate
        lr_cols = ["lr"]
        lr_fig = go.Figure(data=_make_traces(data_dict, lr_cols, smoothing_window))
        lr_fig.update_layout(
            title="Learning Rate",
            yaxis_type="log",
            yaxis_title="LR",
            yaxis=yaxis_base,
            xaxis_title="Epoch",
            **layout_defaults,
        )

        # Hardware Metrics
        cpu_cols = ["sys_cpu_percent"]
        cpu_fig = go.Figure(data=_make_traces(data_dict, cpu_cols, smoothing_window))
        cpu_fig.update_layout(
            title="CPU Usage",
            yaxis_title="%",
            yaxis={**yaxis_base, "range": [0, 100]},
            xaxis_title="Epoch",
            **layout_defaults,
        )

        ram_cols = ["sys_ram_percent"]
        ram_fig = go.Figure(data=_make_traces(data_dict, ram_cols, smoothing_window))
        ram_fig.update_layout(
            title="RAM Usage",
            yaxis_title="%",
            yaxis={**yaxis_base, "range": [0, 100]},
            xaxis_title="Epoch",
            **layout_defaults,
        )

        gpu_cols = ["sys_gpu_mem_mb"]
        gpu_fig = go.Figure(data=_make_traces(data_dict, gpu_cols, smoothing_window))
        gpu_fig.update_layout(
            title="GPU Memory",
            yaxis_title="MB",
            yaxis=yaxis_base,
            xaxis_title="Epoch",
            **layout_defaults,
        )

        # KPI Data - only show for the first selected run (or if only 1 run, that run)
        # to avoid blowing up the UI with 20 cards.
        target_run_name = list(data_dict.keys())[0] if data_dict else None
        kpi_elements = []
        update_text = "Data Loaded"

        if target_run_name:
            df = data_dict[target_run_name]
            last_row = df.iloc[-1]
            last_epoch = int(last_row["epoch"])

            run_lbl = f"{target_run_name} @ " if len(data_dict) > 1 else ""

            if "train_loss" in df.columns:
                kpi_elements.append(
                    create_kpi_card(
                        "Train Loss",
                        f"{last_row['train_loss']:.4f}",
                        f"{run_lbl}Epoch {last_epoch}",
                    )
                )
            if "val_loss" in df.columns:
                kpi_elements.append(
                    create_kpi_card(
                        "Val Loss",
                        f"{last_row['val_loss']:.4f}",
                        f"{run_lbl}Epoch {last_epoch}",
                    )
                )
            if "val_pcc" in df.columns:
                kpi_elements.append(
                    create_kpi_card(
                        "Val PCC",
                        f"{last_row['val_pcc']:.4f}",
                        f"{run_lbl}Epoch {last_epoch}",
                    )
                )
            if "lr" in df.columns:
                kpi_elements.append(
                    create_kpi_card("Learning Rate", f"{last_row['lr']:.2e}")
                )

            update_text = f"Last updated: {target_run_name} Epoch {last_epoch} (Live)"

        return (
            loss_fig,
            pcc_fig,
            var_fig,
            lr_fig,
            cpu_fig,
            ram_fig,
            gpu_fig,
            update_text,
            kpi_elements,
        )
