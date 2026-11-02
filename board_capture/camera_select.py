"""Startup camera picker: find connected cameras, show a snapshot of each, let the user choose."""
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

import ui
from camera import backend

log = logging.getLogger(__name__)

PROBE_WARMUP_FRAMES = 5      # let auto-exposure settle a little before the snapshot
RESCAN_SECONDS = 3           # how often to look again when no camera is found
TILE_W, TILE_H = 400, 225


@dataclass
class FoundCamera:
    index: int
    snapshot: np.ndarray
    width: int
    height: int


def _quiet_opencv():
    """Silence OpenCV's warnings for camera indices that don't exist."""
    logging_mod = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging_mod is not None:
        logging_mod.setLogLevel(logging_mod.LOG_LEVEL_ERROR)


def probe_cameras(cfg: dict, on_progress=None) -> list[FoundCamera]:
    """Try camera indices 0..camera_scan_max-1 and return the ones that give a frame."""
    _quiet_opencv()
    found = []
    for i in range(cfg["camera_scan_max"]):
        if on_progress:
            on_progress(i)
        cap = cv2.VideoCapture(i, backend())
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
            frame = None
            for _ in range(PROBE_WARMUP_FRAMES):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
            if frame is not None:
                found.append(FoundCamera(i, frame, frame.shape[1], frame.shape[0]))
                log.info("Found camera %d (%dx%d)", i, frame.shape[1], frame.shape[0])
        finally:
            cap.release()
    return found


class CameraPicker:
    """Selection state and screen. No camera or window access, so it can be tested."""

    def __init__(self, found: list[FoundCamera], default_index: int, timeout_s: float, now: float):
        self.found = found
        indices = [c.index for c in found]
        self.selected = indices.index(default_index) if default_index in indices else 0
        self.timeout_s = timeout_s
        self.deadline = now + timeout_s if timeout_s > 0 else None

    def handle_key(self, key: int, now: float):
        """Return the chosen camera index, "quit", "rescan", or None to keep waiting."""
        if key >= 0:
            ch = chr(key).lower()
            if ch == "q":
                return "quit"
            if ch == "s":
                return "rescan"
            if ch.isdigit() and 1 <= int(ch) <= len(self.found):
                return self.found[int(ch) - 1].index
            if key in (13, 10):                     # Enter
                return self.found[self.selected].index
            if key == 9 or ch == " ":               # Tab / space: move highlight, stop countdown
                self.selected = (self.selected + 1) % len(self.found)
                self.deadline = None
        if self.deadline is not None and now >= self.deadline:
            return self.found[self.selected].index
        return None

    def render(self, now: float) -> np.ndarray:
        cols = 1 if len(self.found) == 1 else 2 if len(self.found) <= 4 else 3
        rows = -(-len(self.found) // cols)
        gap, head, foot = 24, 90, 70
        w = cols * TILE_W + (cols + 1) * gap
        h = head + rows * (TILE_H + 40 + gap) + foot
        img = np.full((h, max(w, 900), 3), ui.DARK, np.uint8)
        w = img.shape[1]
        ui.put(img, "SELECT CAMERA", (gap, 50), ui.WHITE, 1.1, 2)

        x_start = (w - (cols * TILE_W + (cols - 1) * gap)) // 2
        for n, cam in enumerate(self.found):
            r, c = divmod(n, cols)
            x0, y0 = x_start + c * (TILE_W + gap), head + r * (TILE_H + 40 + gap)
            tile = ui.fit_inside(cam.snapshot, TILE_W, TILE_H)
            img[y0:y0 + TILE_H, x0:x0 + TILE_W] = tile
            chosen = n == self.selected
            col = ui.CYAN if chosen else ui.GREY
            cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + TILE_W + 2, y0 + TILE_H + 2), col,
                          4 if chosen else 1)
            ui.put(img, f"[{n + 1}]  Camera {cam.index}   {cam.width}x{cam.height}",
                   (x0, y0 + TILE_H + 28), ui.WHITE if chosen else ui.GREY, 0.6)

        if self.deadline is not None:
            left = max(0, int(self.deadline - now + 0.999))
            status = f"Starting with Camera {self.found[self.selected].index} in {left}s"
        else:
            status = "Countdown stopped"
        ui.put(img, status, (gap, h - 40), ui.YELLOW, 0.6)
        keys = "[1-9] choose  [Tab] move  [Enter] start  [S] rescan  [Q] quit"
        ui.put(img, keys, (w - ui.text_width(keys, 0.5) - gap, h - 40), ui.GREY, 0.5)
        return img


def select_camera(cfg: dict, show, read_key, window_open):
    """Run the picker. Returns a camera index, or None if the user quit.

    show(img), read_key(ms) and window_open() are the main window's helpers.
    """
    while True:
        found = probe_cameras(cfg, on_progress=lambda i: (show(ui.message_screen(
            "Looking for cameras...", [f"Checking camera {i + 1} of {cfg['camera_scan_max']}"])),
            read_key(1)))

        if not found:
            log.warning("No cameras found")
            retry_at = time.monotonic() + RESCAN_SECONDS
            while time.monotonic() < retry_at:
                show(ui.message_screen("NO CAMERAS FOUND", [
                    "Plug in a camera. Checking again automatically...", "[Q] quit"], ui.RED))
                key = read_key(50)
                if key in (ord("q"), ord("Q")) or not window_open():
                    return None
            continue

        if len(found) == 1:
            log.info("One camera found; using camera %d", found[0].index)
            return found[0].index

        picker = CameraPicker(found, cfg["camera_index"], cfg["camera_select_timeout_s"],
                              time.monotonic())
        while True:
            show(picker.render(time.monotonic()))
            key = read_key(50)
            if not window_open():
                return None
            result = picker.handle_key(key, time.monotonic())
            if result == "quit":
                return None
            if result == "rescan":
                break
            if result is not None:
                log.info("Camera %d selected", result)
                return result
