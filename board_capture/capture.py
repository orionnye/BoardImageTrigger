"""Crop, resize, and atomically save a photo."""
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def crop_and_resize(frame: np.ndarray, bbox, cfg: dict) -> np.ndarray:
    """Crop to bbox + padding (full frame if bbox is None), then cap the long side."""
    img = frame
    if bbox is not None:
        x, y, w, h = bbox
        p = cfg["crop_padding_px"]
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(fw, x + w + p), min(fh, y + h + p)
        img = frame[y0:y1, x0:x1]

    h, w = img.shape[:2]
    limit = cfg["max_long_side_px"]
    if max(h, w) > limit:
        s = limit / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def remove_stale_temp_files(out_dir: Path) -> int:
    """Delete half-written files left behind by a crash during a save."""
    stale = list(out_dir.glob("*.jpg.tmp"))
    for f in stale:
        f.unlink(missing_ok=True)
    return len(stale)


def save_atomic(img: np.ndarray, out_dir: Path, quality: int) -> Path:
    """Write to a temp name then rename, so watchers never see a partial file."""
    name = "board_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
    final = out_dir / name
    tmp = out_dir / (name + ".tmp")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    return final
