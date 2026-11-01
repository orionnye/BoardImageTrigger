"""Load config.json, creating it with defaults if missing."""
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

DEFAULTS = {
    "camera_index": 0,
    "frame_width": 1920,
    "frame_height": 1080,
    "lock_camera_auto": True,
    "present_threshold": 0.05,
    "empty_threshold": 0.03,
    "still_threshold": 0.003,
    "stable_frames": 15,
    "pixel_diff_threshold": 25,
    "background_adapt_rate": 0.02,
    "background_learn_frames": 30,
    "detect_width": 640,
    "edge_margin_px": 4,
    "merge_px": 15,
    "min_object_frac": 0.002,
    "crop_padding_px": 40,
    "max_long_side_px": 1800,
    "jpeg_quality": 92,
    "output_dir": "captures",
    "camera_retry_limit": 30,
    "preview_max_width": 1280,
    "debug_view": False,
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return defaults overlaid with values from the file. Missing file is created."""
    cfg = dict(DEFAULTS)
    if not path.exists():
        path.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
        log.info("Created default config at %s", path)
        return cfg

    try:
        user = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        log.error("config.json is not valid JSON (%s); using defaults", e)
        return cfg

    for key, value in user.items():
        if key not in DEFAULTS:
            log.warning("Unknown config key ignored: %s", key)
            continue
        problem = _check(key, value)
        if problem:
            log.warning("Config %s=%r %s; using default %r", key, value, problem, DEFAULTS[key])
            continue
        cfg[key] = int(value) if isinstance(DEFAULTS[key], int) and not isinstance(value, bool) \
            else value

    if cfg["empty_threshold"] >= cfg["present_threshold"]:
        log.warning("empty_threshold should be below present_threshold; using defaults for both")
        cfg["empty_threshold"] = DEFAULTS["empty_threshold"]
        cfg["present_threshold"] = DEFAULTS["present_threshold"]
    return cfg


FRACTIONS = {"present_threshold", "empty_threshold", "still_threshold",
             "background_adapt_rate", "min_object_frac"}


def _check(key, value) -> str | None:
    """Return why a value is unusable, or None if it's fine."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return None if isinstance(value, bool) else "must be true or false"
    if isinstance(default, str):
        return None if isinstance(value, str) and value else "must be a non-empty string"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be a number"
    if isinstance(default, int) and not float(value).is_integer():
        return "must be a whole number"
    if key in FRACTIONS and not 0 < value < 1:
        return "must be between 0 and 1"
    if key == "jpeg_quality" and not 1 <= value <= 100:
        return "must be 1-100"
    if key not in FRACTIONS and key != "camera_index" and value <= 0:
        return "must be positive"
    if key == "camera_index" and value < 0:
        return "must be 0 or more"
    return None


def resolve_output_dir(cfg: dict, path: Path = CONFIG_PATH) -> Path:
    """Output dir is relative to the config file, not the launch directory."""
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        out = path.parent / out
    out.mkdir(parents=True, exist_ok=True)
    return out
