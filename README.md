# Board Capture

Watches a fixed overhead webcam, notices when a board has been placed on the tray and is
still, and saves **one** cropped photo of it to a folder. Another tool can pick the photos
up from there.

Runs on Windows, macOS and Linux. Needs only Python, OpenCV and NumPy, and works offline.

## Quick start

1. Install **Python 3.11 or newer** ([python.org](https://www.python.org/downloads/)).
   On Windows, tick "Add Python to PATH" during install.
2. Plug in the webcam and point it straight down at the empty tray.
3. Start the tool:
   - **Windows:** double-click `run_windows.bat`
   - **macOS / Linux:** run `./run_mac.sh`

   The first run sets up a private Python environment, which takes a minute.
4. If more than one camera is connected, pick one on the **Select camera** screen (see below).
5. Keep the tray clear for about a second while it learns the empty background.

Photos are saved to `board_capture/captures/`, named `board_YYYYMMDD_HHMMSS_mmm.jpg`.
Each file is written under a temporary name and then renamed, so a program watching the
folder never sees a half-written image.

## Choosing a camera

At startup the tool looks for connected cameras and shows a snapshot from each:

- Press **1-9** to start with that camera, or **Enter** for the highlighted one.
- **Tab** moves the highlight (and stops the countdown). **S** scans again. **Q** quits.
- The choice is saved to `config.json` and highlighted next time. If nobody presses
  anything, it starts with that camera after 10 seconds, so an unattended station
  still boots on its own.
- With only one camera connected, the screen is skipped.
- With no camera connected, it shows **NO CAMERAS FOUND** and keeps checking.

OpenCV can't read camera names, so cameras are listed by number with a snapshot.
To skip the picker, start with `--camera N` or set `camera_select_on_boot` to `false`.

## Using it

The main area shows the **last saved photo**, exactly as it was cropped. The live camera
is a small inset in the bottom-right corner (press `L` to swap them). The coloured banner
at the top always says what to do next:

| Banner | Meaning |
|---|---|
| **LEARNING** (yellow) | Learning the empty tray. Keep it clear. Restarts if anything moves. |
| **READY** (blue) | Waiting for a board. |
| **HOLD STILL** (yellow) | Board seen. Move hands away; the bar fills as it settles. |
| **SAVED** (green) | Photo saved. Remove the board to arm for the next one. |
| **CHECK PHOTO** (red) | Photo saved, but something touched the frame edge (often a hand). Press **R** if it should be thrown away. |

Rules:
- One photo per board. Nudging the board after the photo does nothing.
- The tool re-arms only once the tray is empty again.
- If several objects are on the tray, the **largest** one is photographed. Objects
  touching the frame edge (usually an arm) are only chosen if nothing else is in view.

### Keys (click the window first)

| Key | Action |
|---|---|
| `R` | Reject the last photo: deletes it, and retakes once the board is still |
| `L` | Swap between the large last-photo view (default) and a large live view |
| `B` | Re-learn the empty background (clear the tray first) |
| `D` | Toggle the debug view |
| `H` | Show / hide help |
| `Q` | Quit (closing the window also quits) |

### If the camera is unplugged
The window shows **CAMERA DISCONNECTED** and keeps retrying every few seconds. When the
camera comes back, the tool re-learns the background automatically.

## Setting up a new station

1. Use a matte, plain backdrop with steady lighting. Avoid sunlight that changes during the day.
2. Make sure a board placed on the tray sits fully inside the frame with a small margin.
   Anything touching the frame edge is treated as a possible hand.
3. Run with `--debug` (or press `D`) to see what the detector sees:
   - **Live view:** the target object in bold, ignored objects in grey, movement in magenta,
     the edge "hand zone" in red, and the crop area as a dashed box.
   - **Side panels:** the learned background, objects found, and frame-to-frame movement.
   - **Gauges:** live numbers against each threshold, to help tune `config.json`.
4. Adjust `board_capture/config.json` if needed (see below) and restart.

### Command-line options

```
run_windows.bat --camera 1        # use camera 1 and skip the camera picker
run_windows.bat --debug           # start in the debug view
run_windows.bat --config D:\station2\config.json
```

## Configuration

`board_capture/config.json` is created with defaults on first run. Invalid values are
logged and replaced with the default.

| Key | Default | Purpose |
|---|---|---|
| `camera_index` | 0 | Which webcam (updated when you pick one at startup) |
| `camera_select_on_boot` | true | Show the camera picker when more than one camera is connected |
| `camera_scan_max` | 5 | How many camera numbers (0, 1, 2...) the picker checks |
| `camera_select_timeout_s` | 10 | Seconds before the picker starts with the highlighted camera (0 = wait) |
| `frame_width`, `frame_height` | 1920, 1080 | Requested resolution |
| `lock_camera_auto` | true | Try to lock auto-exposure, focus and white balance |
| `present_threshold` | 0.05 | Share of the frame that must change for "something is on the tray" |
| `empty_threshold` | 0.03 | Below this the tray counts as empty (must be under `present_threshold`) |
| `still_threshold` | 0.003 | Max frame-to-frame change that still counts as "not moving" |
| `stable_frames` | 15 | Still frames needed before a photo (and empty frames before re-arming) |
| `pixel_diff_threshold` | 25 | Brightness difference (0-255) for a pixel to count as changed |
| `background_adapt_rate` | 0.02 | How fast the background absorbs slow lighting drift while empty |
| `background_learn_frames` | 30 | Frames averaged when learning the background |
| `detect_width` | 640 | Width frames are shrunk to for detection (speed vs. detail) |
| `edge_margin_px` | 4 | Width of the edge "hand zone", in detection pixels |
| `merge_px` | 15 | Gaps smaller than this are closed so one board reads as one object |
| `min_object_frac` | 0.002 | Changed areas smaller than this share of the frame are ignored |
| `crop_padding_px` | 40 | Margin around the board in the saved photo |
| `max_long_side_px` | 1800 | Saved photos are shrunk so the long side is at most this |
| `jpeg_quality` | 92 | JPEG quality (1-100) |
| `output_dir` | `captures` | Photo folder, relative to `config.json` |
| `camera_retry_limit` | 30 | Failed reads before showing the disconnected screen |
| `preview_max_width` | 1280 | Width of the operator view |
| `debug_view` | false | Start in the debug view |

### Tuning tips
- **Triggers with nothing on the tray:** raise `pixel_diff_threshold` or `present_threshold`,
  and check that the camera accepted the exposure lock (see the log).
- **Never gets past HOLD STILL:** raise `still_threshold` a little. Camera noise or
  flicker can keep the motion reading above it.
- **Board shows up as several objects:** raise `merge_px`.
- **Crop too tight or too loose:** change `crop_padding_px`.

## Logs

`board_capture/logs/board_capture.log` records saves, rejects, camera problems and config
warnings. It rotates at 1 MB and keeps 3 old files.

## Development

```
cd board_capture
../.venv/bin/python -m unittest discover tests     # Windows: ..\.venv\Scripts\python
```

The tests use simulated camera frames, so no camera is needed.

| File | Role |
|---|---|
| `main.py` | Entry point, main loop, keys, camera reconnect |
| `camera_select.py` | Startup camera picker |
| `camera.py` | Opening the camera, settings, frame reads |
| `detector.py` | Background model, object finding, capture state machine |
| `tracker.py` | Following objects between frames, choosing the target |
| `capture.py` | Crop, resize, safe save |
| `ui.py` | Operator screen and shared drawing helpers |
| `debug_view.py` | Debug view |
| `config.py` | Loading and checking `config.json` |
