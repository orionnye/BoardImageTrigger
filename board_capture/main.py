"""Board Capture: watch the tray, take one cropped photo per board, save it to a folder.

Usage:  python main.py [--camera N] [--config PATH] [--debug]

At startup a camera picker is shown when more than one camera is connected
(skipped with --camera N, or camera_select_on_boot=false in config.json).

Keys (preview window focused):
  R = reject the last photo (deletes it; the board is captured again once still)
  B = re-learn background
  L = swap the operator screen to a large live view
  D = toggle debug view
  H = help
  Q = quit
"""
import argparse
import logging
import logging.handlers
import time
from pathlib import Path

import cv2
import numpy as np

import debug_view
import ui
from camera import Camera
from camera_select import select_camera
from capture import crop_and_resize, remove_stale_temp_files, save_atomic
from config import CONFIG_PATH, load_config, resolve_output_dir, save_camera_index
from detector import Detector, State, StateMachine

WINDOW = "Board Capture"
RECONNECT_SECONDS = 3

log = logging.getLogger("board_capture")


class App:
    """Per-frame logic, independent of where frames and keys come from."""

    def __init__(self, cfg: dict, config_path: Path = CONFIG_PATH):
        self.cfg = cfg
        self.out_dir = resolve_output_dir(cfg, config_path)
        if n := remove_stale_temp_files(self.out_dir):
            log.warning("Removed %d half-written file(s) from a previous crash", n)
        self.det = Detector(cfg)
        self.sm = StateMachine(cfg)
        self.m = None
        self.last_path = None      # last saved photo
        self.last_bbox = None      # where it was cropped (full-res coords)
        self.last_save_t = None
        self.thumb = None
        self.session_count = 0
        self.debug = cfg["debug_view"]
        self.show_help = False
        self.live_large = False    # operator screen shows the last photo large by default
        self.fps = 0.0
        self._last_t = None

    def step(self, frame: np.ndarray, key: int = -1, now: float | None = None) -> np.ndarray:
        """Process one frame and one keypress; return the image to display."""
        now = time.monotonic() if now is None else now
        if self._last_t is not None and now > self._last_t:
            self.fps = 0.9 * self.fps + 0.1 / (now - self._last_t)
        self._last_t = now

        self.handle_key(key)

        m = self.det.process(frame, adapt=self.sm.state == State.ARMED_EMPTY)
        self.m = m
        if self.sm.update(m) == "capture":
            self._save(frame, m.bbox, now)
            if self.sm.state == State.REVIEW and self.last_path:
                log.warning("Possible obstruction in %s - press R to reject", self.last_path.name)

        if self.debug:
            return debug_view.render(self, frame, now)
        return ui.render_operator(self, frame, now)

    def handle_key(self, key):
        if key < 0:
            return
        ch = chr(key).lower()
        if ch == "r" and self.sm.reject() and self.last_path:
            self.last_path.unlink(missing_ok=True)
            log.info("Rejected: deleted %s", self.last_path.name)
            self.session_count -= 1
            self.last_path = self.thumb = self.last_bbox = self.last_save_t = None
        elif ch == "d":
            self.debug = not self.debug
        elif ch == "l":
            self.live_large = not self.live_large
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "b":
            self.relearn()

    def relearn(self):
        log.info("Re-learning background")
        self.det.relearn()
        self.sm.relearn()

    def _save(self, frame, bbox, now):
        img = crop_and_resize(frame, bbox, self.cfg)
        try:
            path = save_atomic(img, self.out_dir, self.cfg["jpeg_quality"])
        except OSError as e:
            log.error("Could not save photo: %s", e)
            self.sm.reject()       # try again once the board is still
            return
        self.last_path = path
        self.thumb = img
        self.last_bbox = bbox
        self.last_save_t = now
        self.session_count += 1
        log.info("Saved %s (%dx%d)", path.name, img.shape[1], img.shape[0])


# ---- window helpers ---------------------------------------------------------

def read_key(delay_ms: int) -> int:
    """Wait for a key; return its ASCII code, or -1. Strips modifier bits some platforms add."""
    key = cv2.waitKey(delay_ms)
    return -1 if key < 0 else key & 0xFF


def window_open() -> bool:
    """False once the user closes the window with its close button."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def show(img):
    cv2.imshow(WINDOW, img)


# ---- setup ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Detect a board on the tray and save one photo of it.")
    p.add_argument("--camera", type=int, help="camera index (skips the camera picker)")
    p.add_argument("--config", type=Path, default=CONFIG_PATH, help="path to config.json")
    p.add_argument("--debug", action="store_true", help="start in the debug view")
    return p.parse_args()


def setup_logging(log_dir: Path):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    log_dir.mkdir(parents=True, exist_ok=True)
    file = logging.handlers.RotatingFileHandler(log_dir / "board_capture.log",
                                                maxBytes=1_000_000, backupCount=3)
    file.setFormatter(fmt)
    root.addHandler(file)


def main():
    args = parse_args()
    setup_logging(args.config.resolve().parent / "logs")
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.debug:
        cfg["debug_view"] = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    if args.camera is None and cfg["camera_select_on_boot"]:
        chosen = select_camera(cfg, show, read_key, window_open)
        if chosen is None:
            cv2.destroyAllWindows()
            return
        if chosen != cfg["camera_index"]:
            save_camera_index(chosen, args.config)
        cfg["camera_index"] = chosen

    show(ui.message_screen("Starting camera...", [f"Camera {cfg['camera_index']}"]))
    read_key(1)

    cam = Camera(cfg)
    cam.open()
    app = App(cfg, args.config)
    log.info("Saving photos to %s", app.out_dir)
    key, next_reconnect = -1, time.monotonic() + RECONNECT_SECONDS

    try:
        while True:
            frame = cam.read()
            if frame is not None:
                show(app.step(frame, key))
                key = read_key(1)
            else:
                if cam.failed:
                    show(ui.message_screen("CAMERA DISCONNECTED", [
                        "Check the USB cable. Reconnecting automatically...", "[Q] quit"], ui.RED))
                    if time.monotonic() >= next_reconnect:
                        cam.release()
                        if cam.open():
                            log.info("Camera reconnected")
                            app.relearn()   # the view may have shifted
                        next_reconnect = time.monotonic() + RECONNECT_SECONDS
                pressed = read_key(30)
                key = pressed if pressed >= 0 else key   # keep a key pressed during an outage
            if key in (ord("q"), ord("Q")) or not window_open():
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        log.info("Stopped. %d photo(s) saved this session.", app.session_count)


if __name__ == "__main__":
    main()
