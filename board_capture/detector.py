"""Background model, object finding, and per-frame presence / stillness measurements."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from tracker import Track, Tracker

MOVING_FRAC = 0.02   # share of an object's box that must change frame-to-frame to count as moving
LEARN_MOTION_FACTOR = 3   # movement above still_threshold x this restarts background learning


@dataclass
class Measurement:
    present_frac: float          # fraction of pixels changed vs. empty background
    motion_frac: float           # fraction of pixels changed vs. previous frame
    bbox: tuple | None           # (x, y, w, h) of the target object, full-res coords
    touches_edge: bool           # target reaches the frame edge (likely a hand/arm)
    objects: list = field(default_factory=list)   # all tracked objects (detection coords)
    target: Track | None = None


class Detector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.background = None   # float32, detection-size grayscale
        self.prev = None
        self._learn_acc = None
        self._learn_count = 0
        self.learn_restarts = 0
        self.debug = None
        self.tracker = Tracker()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k = max(1, int(cfg["merge_px"]))
        self._merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ---- preprocessing -------------------------------------------------

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.cfg["detect_width"] / w
        small = cv2.resize(frame, (self.cfg["detect_width"], round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _diff_mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, self.cfg["pixel_diff_threshold"], 255, cv2.THRESH_BINARY)
        return cv2.dilate(mask, self._kernel, iterations=2)

    # ---- background learning -------------------------------------------

    @property
    def learning(self) -> bool:
        return self.background is None

    @property
    def learn_progress(self) -> float:
        return self._learn_count / self.cfg["background_learn_frames"]

    def relearn(self):
        self.learn_restarts = 0
        self.background = None
        self.prev = None
        self._learn_acc = None
        self._learn_count = 0
        self.tracker.reset()

    def _feed_learning(self, gray: np.ndarray):
        if self._learn_acc is None:
            self._learn_acc = np.zeros(gray.shape, np.float32)
        self._learn_acc += gray
        self._learn_count += 1
        if self._learn_count >= self.cfg["background_learn_frames"]:
            self.background = self._learn_acc / self._learn_count
            self._learn_acc = None

    # ---- per-frame -----------------------------------------------------

    def process(self, frame: np.ndarray, adapt: bool) -> Measurement | None:
        """Measure one frame. Returns None while the background is still being learned.

        adapt: blend this frame into the background (only when the tray is empty).
        """
        gray = self._prepare(frame)
        if self.learning:
            # Something moved while learning: start over so it isn't baked into the background.
            if self.prev is not None and self._learn_count > 0:
                moved = cv2.countNonZero(self._diff_mask(gray, self.prev)) / gray.size
                if moved > self.cfg["still_threshold"] * LEARN_MOTION_FACTOR:
                    self._learn_acc, self._learn_count = None, 0
                    self.learn_restarts += 1
            self._feed_learning(gray)
            self.prev = gray
            return None

        bg = cv2.convertScaleAbs(self.background)
        bg_mask = self._diff_mask(gray, bg)
        present_frac = cv2.countNonZero(bg_mask) / bg_mask.size

        motion_frac = 0.0
        motion_mask = np.zeros_like(bg_mask)
        if self.prev is not None:
            motion_mask = self._diff_mask(gray, self.prev)
            motion_frac = cv2.countNonZero(motion_mask) / motion_mask.size
        self.prev = gray

        # Close small gaps so a board split by dark parts reads as one solid object.
        objects_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, self._merge_kernel)
        tracks = self.tracker.update(self._find_objects(objects_mask, motion_mask))
        target = self.tracker.choose_target()

        bbox, touches_edge = None, False
        if target is not None:
            mh, mw = bg_mask.shape
            sx, sy = frame.shape[1] / mw, frame.shape[0] / mh
            x, y, w, h = target.bbox
            bbox = (round(x * sx), round(y * sy), round(w * sx), round(h * sy))
            touches_edge = target.touches_edge

        # Intermediate images, kept for the debug view (detection-size coords).
        self.debug = {"background": bg, "bg_mask": bg_mask,
                      "objects_mask": objects_mask, "motion_mask": motion_mask}

        # Only absorb frames that themselves look empty, so an arriving board never ghosts in.
        if adapt and present_frac < self.cfg["empty_threshold"]:
            cv2.accumulateWeighted(gray, self.background, self.cfg["background_adapt_rate"])

        return Measurement(present_frac, motion_frac, bbox, touches_edge, tracks, target)

    def _find_objects(self, mask: np.ndarray, motion_mask: np.ndarray) -> list[Track]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mh, mw = mask.shape
        min_area = self.cfg["min_object_frac"] * mask.size
        m = self.cfg["edge_margin_px"]
        objects = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            edge = x <= m or y <= m or x + w >= mw - m or y + h >= mh - m
            moving = cv2.countNonZero(motion_mask[y:y + h, x:x + w]) > MOVING_FRAC * w * h
            objects.append(Track(0, c, (x, y, w, h), area, edge, moving))
        return objects


# ---- state machine ------------------------------------------------------

class State:
    LEARNING = "LEARNING"
    ARMED_EMPTY = "ARMED_EMPTY"
    SETTLING = "SETTLING"
    REVIEW = "REVIEW"          # photo saved, but something touched the frame edge
    CAPTURED = "CAPTURED"


SETTLING_EMPTY_FRAMES = 3   # empty frames while settling before going back to armed


class StateMachine:
    """Decides when to take a photo. update() returns "capture" when one should be saved."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = State.LEARNING
        self._still = 0
        self._empty = 0

    @property
    def still_progress(self) -> float:
        """0..1 toward a capture while settling."""
        return min(1.0, self._still / self.cfg["stable_frames"])

    @property
    def empty_progress(self) -> float:
        """0..1 toward re-arming after a capture."""
        return min(1.0, self._empty / self.cfg["stable_frames"])

    def _go(self, state):
        self.state = state
        self._still = self._empty = 0

    def update(self, m: Measurement | None):
        cfg = self.cfg
        if m is None:                       # background still being learned
            self.state = State.LEARNING
            return None
        if self.state == State.LEARNING:
            self._go(State.ARMED_EMPTY)

        empty = m.present_frac < cfg["empty_threshold"]
        self._empty = self._empty + 1 if empty else 0

        if self.state == State.ARMED_EMPTY:
            if m.present_frac > cfg["present_threshold"]:
                self._go(State.SETTLING)

        elif self.state == State.SETTLING:
            if self._empty >= SETTLING_EMPTY_FRAMES:
                self._go(State.ARMED_EMPTY)
                return None
            still = m.motion_frac < cfg["still_threshold"] and not empty
            self._still = self._still + 1 if still else 0
            if self._still >= cfg["stable_frames"]:
                # Save either way; a suspected hand only adds a warning the operator can reject.
                self._go(State.REVIEW if m.touches_edge else State.CAPTURED)
                return "capture"

        elif self.state in (State.REVIEW, State.CAPTURED):
            if self._empty >= cfg["stable_frames"]:
                self._go(State.ARMED_EMPTY)

        return None

    # ---- operator actions ----------------------------------------------

    def reject(self) -> bool:
        """Discard the last photo and wait for the board to settle again."""
        if self.state not in (State.REVIEW, State.CAPTURED):
            return False
        self._go(State.SETTLING)
        return True

    def relearn(self):
        self._go(State.LEARNING)
