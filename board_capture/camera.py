"""Camera open, settings, and frame reads. Works on Windows, macOS and Linux."""
import logging
import sys
import time

import cv2

log = logging.getLogger(__name__)

WARMUP_FRAMES = 30


def _backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


class Camera:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cap = None
        self.failures = 0

    @property
    def failed(self) -> bool:
        """True once consecutive read failures reach the configured limit."""
        return self.failures >= self.cfg["camera_retry_limit"]

    def open(self) -> bool:
        cfg = self.cfg
        self.cap = cv2.VideoCapture(cfg["camera_index"], _backend())
        if not self.cap.isOpened():
            log.error("Could not open camera %s", cfg["camera_index"])
            self.failures = cfg["camera_retry_limit"]
            return False

        # MJPG lets most USB webcams deliver 1080p at a usable frame rate.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])

        for _ in range(WARMUP_FRAMES):
            self.cap.read()

        if cfg["lock_camera_auto"]:
            self._lock_auto()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera %s opened at %dx%d", cfg["camera_index"], w, h)
        if (w, h) != (cfg["frame_width"], cfg["frame_height"]):
            log.warning("Requested %dx%d, camera gave %dx%d",
                        cfg["frame_width"], cfg["frame_height"], w, h)
        self.failures = 0
        return True

    def _lock_auto(self):
        # Manual-exposure value differs by backend: DirectShow uses 0.25, V4L2 uses 1.
        manual_exposure = 0.25 if sys.platform.startswith("win") else 1
        attempts = [
            ("auto-exposure", cv2.CAP_PROP_AUTO_EXPOSURE, manual_exposure),
            ("auto-focus", cv2.CAP_PROP_AUTOFOCUS, 0),
            ("auto white balance", cv2.CAP_PROP_AUTO_WB, 0),
        ]
        for name, prop, value in attempts:
            ok = self.cap.set(prop, value)
            if not ok:
                log.warning("Camera refused to lock %s; continuing", name)
            else:
                log.info("Locked %s", name)

    def read(self):
        """Return a BGR frame, or None on failure. Never raises."""
        if self.cap is None:
            self.failures += 1
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.failures += 1
            if self.failures == self.cfg["camera_retry_limit"]:
                log.error("Camera read failed %d times in a row", self.failures)
            time.sleep(0.03)
            return None
        self.failures = 0
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
