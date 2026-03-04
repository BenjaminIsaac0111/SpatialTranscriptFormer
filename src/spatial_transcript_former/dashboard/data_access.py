import os
import sqlite3
import pandas as pd
import glob
import logging
from threading import Lock

# Simple thread-safe cache for images to avoid globbing disc constantly
_image_cache = {"last_check": 0, "images": []}
_cache_lock = Lock()


import os
import sqlite3
import pandas as pd
import glob
import logging
from threading import Lock

# Simple thread-safe cache for images to avoid globbing disc constantly
_image_cache = {"last_check": 0, "images": []}
_cache_lock = Lock()


def get_available_runs(args):
    """Returns a list of dicts with name and path for available runs."""
    runs = []
    if getattr(args, "run_dir", None):
        if os.path.exists(args.run_dir):
            runs.append(
                {
                    "name": os.path.basename(os.path.normpath(args.run_dir)),
                    "path": args.run_dir,
                }
            )

    if getattr(args, "runs_dir", None) and os.path.exists(args.runs_dir):
        # Scan immediate subdirectories
        for entry in os.scandir(args.runs_dir):
            if entry.is_dir() and not entry.name.startswith("."):  # Ignore hidden
                runs.append({"name": entry.name, "path": entry.path})

    # Sort runs alphabetically by name for consistency
    runs.sort(key=lambda x: x["name"])
    return runs


def get_db_path(run_dir):
    return os.path.join(run_dir, "training_logs.sqlite")


def _fetch_run_metrics(run_dir):
    """Fetch all rows from the metrics table in the SQLite database for a single run."""
    db_path = get_db_path(run_dir)

    if not os.path.exists(db_path):
        # Fallback to CSV if DB doesn't exist yet (for backwards compat)
        csv_path = os.path.join(run_dir, "training_log.csv")
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                logging.error(f"Failed to read CSV: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
            )
            if not cursor.fetchone():
                return pd.DataFrame()

            query = "SELECT * FROM metrics ORDER BY epoch ASC"
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        logging.error(f"Database error reading metrics: {e}")
        return pd.DataFrame()


def get_training_data(args, selected_runs=None):
    """
    Fetch metric data across selected runs.
    If selected_runs is None, defaults to all available runs.
    Returns: Dict mapping run_name -> DataFrame.
    """
    all_runs = get_available_runs(args)
    if selected_runs:
        runs_to_fetch = [r for r in all_runs if r["name"] in selected_runs]
    else:
        runs_to_fetch = all_runs

    data_dict = {}
    for run in runs_to_fetch:
        df = _fetch_run_metrics(run["path"])
        if not df.empty:
            data_dict[run["name"]] = df

    return data_dict


def get_available_images(args, selected_runs=None, cache_ttl=10):
    """Scans for inference plot images and extracts metadata, caching results across runs."""
    import time

    with _cache_lock:
        now = time.time()
        # Need to ensure cache keying is valid, but keeping simple for now
        # Refresh if TTL expired or cache is totally empty
        if now - _image_cache["last_check"] < cache_ttl and _image_cache["images"]:
            all_imgs = _image_cache["images"]
        else:
            all_runs = get_available_runs(args)
            parsed_images = []

            for run in all_runs:
                search_pattern = os.path.join(run["path"], "*.png")
                files = glob.glob(search_pattern)

                for file in files:
                    basename = os.path.basename(file)
                    # Expected format: SAMPLEID_epoch_NUM.png
                    try:
                        parts = basename.replace(".png", "").split("_epoch_")
                        if len(parts) == 2:
                            sample_id = parts[0]
                            epoch = int(parts[1])
                            mtime = os.path.getmtime(file)
                            parsed_images.append(
                                {
                                    "filename": basename,
                                    "run_name": run["name"],
                                    "run_path": run["path"],
                                    "sample": sample_id,
                                    "epoch": epoch,
                                    "mtime": mtime,
                                }
                            )
                    except Exception:
                        pass  # Skip improperly named files

            # Sort by epoch descending then run alphabetical
            parsed_images.sort(key=lambda x: (x["epoch"], x["run_name"]), reverse=True)

            _image_cache["last_check"] = now
            _image_cache["images"] = parsed_images
            all_imgs = parsed_images

    # Filter by selected runs after cache fetch
    if selected_runs:
        return [img for img in all_imgs if img["run_name"] in selected_runs]
    return all_imgs
