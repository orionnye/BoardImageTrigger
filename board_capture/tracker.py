"""Follow separate objects across frames and pick one target to photograph."""
from dataclasses import dataclass

import numpy as np

MATCH_IOU = 0.3      # min box overlap to count as the same object as last frame
SWITCH_RATIO = 1.2   # a new object must be this much bigger to steal the target
LOST_FRAMES = 5      # an object that vanishes mid-frame is remembered this long, then dropped


@dataclass
class Track:
    id: int
    contour: np.ndarray      # outline, detection-size coords
    bbox: tuple              # (x, y, w, h), detection-size coords
    area: float              # pixels, detection size
    touches_edge: bool       # reaches the hand zone at the frame border
    moving: bool             # changed since the previous frame
    age: int = 1             # frames seen
    missed: int = 0          # consecutive frames not seen (only for remembered objects)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []    # objects visible this frame
        self._lost: list[Track] = []     # recently vanished, kept briefly to survive flicker
        self.target_id = None
        self._next_id = 1

    def reset(self):
        self.tracks, self._lost, self.target_id = [], [], None

    def update(self, objects: list[Track]) -> list[Track]:
        """objects: this frame's detections (id ignored). Returns tracks with stable ids."""
        known = self.tracks + self._lost
        pairs = sorted(((_iou(t.bbox, o.bbox), ti, oi)
                        for ti, t in enumerate(known)
                        for oi, o in enumerate(objects)), reverse=True)
        used_t, used_o = set(), set()
        for iou, ti, oi in pairs:
            if iou < MATCH_IOU:
                break
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            objects[oi].id = known[ti].id
            objects[oi].age = known[ti].age + 1
        for oi, o in enumerate(objects):
            if oi not in used_o:
                o.id = self._next_id
                self._next_id += 1

        # Forget objects that left: immediately if last seen at the frame edge (went
        # off-screen), otherwise after LOST_FRAMES frames without being seen again.
        lost = []
        for ti, t in enumerate(known):
            if ti in used_t or t.touches_edge:
                continue
            t.missed += 1
            if t.missed <= LOST_FRAMES:
                lost.append(t)
        self._lost = lost
        self.tracks = objects
        return self.tracks

    @property
    def remembered(self) -> int:
        """Total objects held in memory (visible + recently vanished)."""
        return len(self.tracks) + len(self._lost)

    def choose_target(self) -> Track | None:
        """Largest object wins. Edge-touching objects (usually arms) only win if nothing
        else is in view. The current target is kept unless another is clearly bigger."""
        if not self.tracks:
            self.target_id = None
            return None
        inner = [t for t in self.tracks if not t.touches_edge]
        pool = inner or self.tracks
        best = max(pool, key=lambda t: t.area)
        current = next((t for t in pool if t.id == self.target_id), None)
        if current is not None and best.area < current.area * SWITCH_RATIO:
            best = current
        self.target_id = best.id
        return best
