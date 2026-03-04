import dash
import flask
import os
import argparse
import logging
from .layout import create_layout
from .callbacks import register_callbacks

# Configure Python logging (app level)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Global references (assigned during init)
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)


def init_app(args):
    """Initialize layout, routes, and callbacks with the given arguments."""

    # Configure Flask to serve images from the run directory
    @server.route("/images/<run_name>/<path:filename>")
    @server.route("/images/<path:filename>")
    def serve_image(filename, run_name=None):
        if run_name and getattr(args, "runs_dir", None):
            directory = os.path.join(os.path.abspath(args.runs_dir), run_name)
        elif getattr(args, "run_dir", None):
            directory = os.path.abspath(args.run_dir)
        else:
            return "Not found", 404

        return flask.send_from_directory(directory, filename)

    app.title = (
        f"Training Monitor - {os.path.basename(args.run_dir)}"
        if getattr(args, "run_dir", None)
        else "Training Monitor: Compare Runs"
    )
    app.layout = create_layout(args)
    register_callbacks(app, args)

    return app
