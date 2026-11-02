"""Behaviour tests using simulated camera frames. No camera needed.

Run from the board_capture folder:  python -m unittest discover tests
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_select import CameraPicker, FoundCamera  # noqa: E402
from capture import crop_and_resize, remove_stale_temp_files, save_atomic  # noqa: E402
from config import DEFAULTS, load_config, save_camera_index  # noqa: E402
from detector import State  # noqa: E402
from main import App  # noqa: E402
from tracker import Track, Tracker  # noqa: E402

W, H = 960, 540
_rng = np.random.default_rng(0)
_NOISE = [_rng.normal(0, 3, (H, W, 3)).astype(np.int16) for _ in range(4)]


class Scene:
    """Draws simple frames: a plain tray, a board, a small item, an arm from the right edge."""

    def __init__(self):
        self.n = 0

    def frame(self, board=False, dx=0, item=False, arm=False, wiggle=False, light=120):
        self.n += 1
        f = np.clip(_NOISE[self.n % 4] + int(light), 0, 255).astype(np.uint8)
        if board:
            cv2.rectangle(f, (250 + dx, 150), (500 + dx, 350), (30, 90, 30), -1)
            cv2.rectangle(f, (250 + dx, 240), (500 + dx, 248), (120, 120, 120), -1)  # split line
        if item:
            cv2.circle(f, (700, 400), 35, (40, 40, 160), -1)
        if arm:
            dy = (self.n % 4) * 6 if wiggle else 0
            cv2.rectangle(f, (450, 230 + dy), (W, 290 + dy), (150, 170, 210), -1)
        return f


def make_app(tmp):
    cfg = dict(DEFAULTS, output_dir=tmp, debug_view=True)
    return App(cfg)


class AppFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = make_app(self.tmp)
        self.scene = Scene()
        self.run_frames(35)                       # learn background + arm

    def run_frames(self, n, key=-1, **kw):
        for i in range(n):
            self.app.step(self.scene.frame(**kw), key if i == 0 else -1)

    def photos(self):
        return sorted(Path(self.tmp).glob("*.jpg"))

    def test_one_photo_per_board(self):
        self.assertEqual(self.app.sm.state, State.ARMED_EMPTY)
        self.run_frames(10, board=True, arm=True, wiggle=True)   # carried in by hand
        self.run_frames(14, board=True)
        self.assertEqual(self.photos(), [], "captured before 15 still frames")
        self.run_frames(2, board=True)
        self.assertEqual(len(self.photos()), 1)
        self.assertEqual(self.app.sm.state, State.CAPTURED)

    def test_nudge_does_not_retake_and_removal_rearms(self):
        self.run_frames(20, board=True)
        self.run_frames(20, board=True, dx=10)
        self.assertEqual(len(self.photos()), 1)
        self.run_frames(20)
        self.assertEqual(self.app.sm.state, State.ARMED_EMPTY)

    def test_no_capture_while_hand_moving(self):
        self.run_frames(60, board=True, arm=True, wiggle=True)
        self.assertEqual(self.photos(), [])
        self.assertEqual(self.app.sm.state, State.SETTLING)

    def test_obstruction_saves_with_warning_then_reject(self):
        self.run_frames(20, board=True, arm=True)              # arm over board, still
        self.assertEqual(self.app.sm.state, State.REVIEW)
        self.assertEqual(len(self.photos()), 1)
        self.run_frames(10, board=True, arm=True, wiggle=True)  # movement doesn't reject
        self.assertEqual(self.app.sm.state, State.REVIEW)
        self.run_frames(1, key=ord("r"), board=True)
        self.assertEqual(self.photos(), [])
        self.assertEqual(self.app.sm.state, State.SETTLING)
        self.run_frames(20, board=True)
        self.assertEqual(len(self.photos()), 1)
        self.assertEqual(self.app.sm.state, State.CAPTURED)

    def test_largest_object_is_cropped(self):
        self.run_frames(5, board=True, item=True, arm=True, wiggle=True)
        self.run_frames(20, board=True, item=True)
        img = cv2.imread(str(self.photos()[0]))
        h, w = img.shape[:2]
        # Board is 250x200 at 960 wide = 500x400 at full res, plus padding; item excluded.
        self.assertLess(w, 700)
        self.assertLess(h, 600)
        self.assertEqual(self.app.m.target.touches_edge, False)

    def test_slow_lighting_drift_never_triggers(self):
        for i in range(600):
            self.app.step(self.scene.frame(light=120 + 30 * i / 600))
        self.assertEqual(self.app.sm.state, State.ARMED_EMPTY)
        self.assertEqual(self.photos(), [])

    def test_session_count_tracks_saves_and_rejects(self):
        self.run_frames(20, board=True)
        self.assertEqual(self.app.session_count, 1)
        self.run_frames(1, key=ord("r"), board=True)
        self.assertEqual(self.app.session_count, 0)

    def test_operator_and_debug_views_render_every_state(self):
        for debug in (False, True):
            self.app.debug = debug
            self.app.show_help = debug
            self.app.live_large = debug
            for kw in ({}, {"board": True, "arm": True, "wiggle": True}, {"board": True}):
                img = self.app.step(self.scene.frame(**kw))
                self.assertEqual(img.ndim, 3)

    def test_movement_during_learning_restarts_it(self):
        app = make_app(self.tmp)
        for i in range(20):
            app.step(self.scene.frame(board=True, dx=i * 5))
        self.assertTrue(app.det.learning)
        self.assertGreater(app.det.learn_restarts, 0)

    def test_relearn_in_both_views_survives_learning_finishing(self):
        for debug in (True, False):
            self.app.debug = debug
            self.run_frames(20, board=True)                 # something on the tray
            self.run_frames(1, key=ord("b"))
            self.run_frames(40)                             # learning completes mid-run
            self.assertEqual(self.app.sm.state, State.ARMED_EMPTY)

    def test_relearn_key(self):
        self.run_frames(1, key=ord("b"))
        self.assertEqual(self.app.sm.state, State.LEARNING)


class TrackerTests(unittest.TestCase):
    @staticmethod
    def obj(x, y=100, w=50, h=50):
        edge = x <= 4 or x + w >= 636
        return Track(0, np.zeros((1, 1, 2), np.int32), (x, y, w, h), w * h, edge, False)

    def test_forgets_objects_that_leave_via_edge(self):
        t = Tracker()
        t.update([self.obj(100), self.obj(300)])
        second = t.tracks[1].id
        for x in range(310, 600, 10):
            t.update([self.obj(100), self.obj(x)])
        self.assertEqual(t.tracks[1].id, second)
        t.update([self.obj(100)])
        self.assertEqual(t.remembered, 1)

    def test_survives_flicker_then_forgets(self):
        t = Tracker()
        t.update([self.obj(100)])
        first = t.tracks[0].id
        t.update([])
        t.update([])
        t.update([self.obj(102)])
        self.assertEqual(t.tracks[0].id, first)
        for _ in range(6):
            t.update([])
        self.assertEqual(t.remembered, 0)

    def test_target_prefers_largest_inner_object(self):
        t = Tracker()
        arm = Track(0, None, (500, 0, 140, 300), 140 * 300, True, True)
        t.update([self.obj(100, w=80, h=80), self.obj(300, w=40, h=40), arm])
        self.assertEqual(t.choose_target().area, 80 * 80)


class CaptureTests(unittest.TestCase):
    def test_crop_pads_clamps_and_limits_size(self):
        cfg = dict(DEFAULTS)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        self.assertEqual(crop_and_resize(frame, (10, 10, 100, 100), cfg).shape[:2], (150, 150))
        self.assertEqual(max(crop_and_resize(frame, None, cfg).shape[:2]), 1800)

    def test_atomic_save_and_stale_cleanup(self):
        out = Path(tempfile.mkdtemp())
        (out / "board_x.jpg.tmp").write_bytes(b"partial")
        self.assertEqual(remove_stale_temp_files(out), 1)
        path = save_atomic(np.zeros((10, 10, 3), np.uint8), out, 90)
        self.assertTrue(path.exists())
        self.assertEqual(list(out.glob("*.tmp")), [])


class ConfigTests(unittest.TestCase):
    def load(self, data):
        p = Path(tempfile.mkdtemp()) / "config.json"
        p.write_text(json.dumps(data) if not isinstance(data, str) else data)
        return load_config(p)

    def test_missing_file_created(self):
        p = Path(tempfile.mkdtemp()) / "config.json"
        self.assertEqual(load_config(p), DEFAULTS)
        self.assertTrue(p.exists())

    def test_bad_values_fall_back_to_defaults(self):
        cfg = self.load({"present_threshold": "high", "stable_frames": 2.5,
                         "jpeg_quality": 500, "lock_camera_auto": 1, "stable_frames_x": 3})
        for k in ("present_threshold", "stable_frames", "jpeg_quality", "lock_camera_auto"):
            self.assertEqual(cfg[k], DEFAULTS[k])

    def test_good_values_kept(self):
        cfg = self.load({"stable_frames": 20.0, "present_threshold": 0.08})
        self.assertEqual(cfg["stable_frames"], 20)
        self.assertIsInstance(cfg["stable_frames"], int)
        self.assertEqual(cfg["present_threshold"], 0.08)

    def test_invalid_json_uses_defaults(self):
        self.assertEqual(self.load("{not json"), DEFAULTS)


class CameraPickerTests(unittest.TestCase):
    def picker(self, default=2, timeout=10):
        snap = np.zeros((90, 160, 3), np.uint8)
        found = [FoundCamera(i, snap, 160, 90) for i in (0, 2, 3)]
        return CameraPicker(found, default, timeout, now=0.0)

    def test_highlights_saved_camera_and_auto_starts_after_timeout(self):
        p = self.picker(default=2)
        self.assertIsNone(p.handle_key(-1, now=9.9))
        self.assertEqual(p.handle_key(-1, now=10.0), 2)

    def test_number_keys_pick_by_position(self):
        self.assertEqual(self.picker().handle_key(ord("3"), now=0), 3)
        self.assertIsNone(self.picker().handle_key(ord("9"), now=0))

    def test_tab_moves_highlight_and_stops_countdown(self):
        p = self.picker(default=2)
        p.handle_key(9, now=0)
        self.assertIsNone(p.handle_key(-1, now=100))
        self.assertEqual(p.handle_key(13, now=100), 3)

    def test_quit_rescan_and_unknown_default(self):
        p = self.picker(default=7)          # saved camera no longer connected
        self.assertEqual(p.selected, 0)
        self.assertEqual(p.handle_key(ord("q"), now=0), "quit")
        self.assertEqual(p.handle_key(ord("s"), now=0), "rescan")
        self.assertEqual(p.render(now=1).ndim, 3)

    def test_zero_timeout_waits_forever(self):
        self.assertIsNone(self.picker(timeout=0).handle_key(-1, now=1e6))

    def test_save_camera_index_keeps_other_settings(self):
        p = Path(tempfile.mkdtemp()) / "config.json"
        p.write_text(json.dumps({"camera_index": 0, "jpeg_quality": 80}))
        save_camera_index(3, p)
        self.assertEqual(json.loads(p.read_text()), {"camera_index": 3, "jpeg_quality": 80})


if __name__ == "__main__":
    unittest.main()
