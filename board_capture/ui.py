"""Operator screen plus drawing helpers shared with the debug view."""
import cv2
import numpy as np

from detector import State

GREEN, RED, YELLOW = (60, 200, 60), (60, 60, 230), (40, 210, 235)
WHITE, GREY, DARK, BLACK = (255, 255, 255), (140, 140, 140), (28, 28, 28), (0, 0, 0)
CYAN, MAGENTA, ORANGE, BLUE = (230, 200, 0), (220, 60, 220), (0, 150, 255), (200, 140, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX

FLASH_SECONDS = 0.6
BANNER_H = 64
FOOTER_H = 44

# Per state: (headline, instruction, banner colour)
STATE_TEXT = {
    State.LEARNING: ("LEARNING", "Keep the tray clear while the empty background is learned", YELLOW),
    State.ARMED_EMPTY: ("READY", "Place a board on the tray", BLUE),
    State.SETTLING: ("HOLD STILL", "Move hands away and let the board settle", YELLOW),
    State.REVIEW: ("CHECK PHOTO", "Saved, but something touched the edge.  [R] reject and retake", RED),
    State.CAPTURED: ("SAVED", "Remove the board for the next one   ([R] reject)", GREEN),
}

HELP_LINES = [
    "HOW IT WORKS",
    "1. Keep the tray clear at startup while the background is learned.",
    "2. Place a board and move your hands away.",
    "3. When the board is still, one photo is cropped and saved automatically.",
    "4. Remove the board. The tool re-arms for the next one.",
    "",
    "If a hand or object touched the frame edge, the photo is still saved",
    "but marked CHECK PHOTO. Press R if it should be thrown away.",
    "",
    "KEYS",
    "R   reject the last photo (deletes it, retakes once the board is still)",
    "L   swap to a large live camera view (and back)",
    "B   re-learn the empty background (clear the tray first)",
    "D   toggle the debug view",
    "H   show / hide this help",
    "Q   quit",
]


def state_text(app):
    """(headline, instruction, colour) for the current state."""
    title, instruction, col = STATE_TEXT[app.sm.state]
    if app.sm.state == State.LEARNING and app.det.learn_restarts:
        instruction = "Movement seen, starting over - keep the tray clear"
    return title, instruction, col


# ---- drawing helpers ------------------------------------------------------

def put(img, msg, org, color=WHITE, scale=0.5, thick=1):
    """Text with a 1px drop shadow (same thickness, so spacing matches)."""
    x, y = org
    cv2.putText(img, msg, (x + 1, y + 1), FONT, scale, BLACK, thick, cv2.LINE_AA)
    cv2.putText(img, msg, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def text_width(msg, scale, thick=1) -> int:
    return cv2.getTextSize(msg, FONT, scale, thick)[0][0]


def blend_rect(img, p0, p1, color, alpha):
    x0, y0 = max(0, p0[0]), max(0, p0[1])
    x1, y1 = min(img.shape[1], p1[0]), min(img.shape[0], p1[1])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0, dst=roi)


def dashed_rect(img, p0, p1, color, dash=12, thick=2):
    (x0, y0), (x1, y1) = p0, p1
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, thick)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, thick)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, thick)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, thick)


def fit_width(img, width):
    """Resize to an exact width, keeping aspect. Returns (image, scale)."""
    h, w = img.shape[:2]
    s = width / w
    if s == 1:
        return img.copy(), 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (width, round(h * s)), interpolation=interp), s


def flash_border(img, app, now):
    """Brief coloured border right after a photo is saved."""
    if app.last_save_t is None or now - app.last_save_t > FLASH_SECONDS:
        return
    col = RED if app.sm.state == State.REVIEW else GREEN
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 14)


def help_overlay(img):
    h, w = img.shape[:2]
    pw, ph = min(820, w - 40), 40 + 28 * len(HELP_LINES)
    x0, y0 = (w - pw) // 2, max(10, (h - ph) // 2)
    blend_rect(img, (x0, y0), (x0 + pw, y0 + ph), DARK, 0.92)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), GREY, 1)
    for i, line in enumerate(HELP_LINES):
        heading = line.isupper()
        put(img, line, (x0 + 24, y0 + 36 + i * 28), CYAN if heading else WHITE,
            0.6 if heading else 0.55)


def message_screen(title, lines, color=YELLOW, size=(1280, 720)):
    """Full-window message, e.g. starting up or camera lost."""
    w, h = size
    img = np.full((h, w, 3), DARK, np.uint8)
    put(img, title, ((w - text_width(title, 1.2, 2)) // 2, h // 2 - 30), color, 1.2, 2)
    for i, line in enumerate(lines):
        put(img, line, ((w - text_width(line, 0.6)) // 2, h // 2 + 20 + i * 30), WHITE, 0.6)
    return img


# ---- operator screen --------------------------------------------------------

def fit_inside(img, w, h):
    """Scale to fit within w x h (no cropping), centred on a dark canvas."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    canvas = np.full((h, w, 3), DARK, np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas


def live_view(app, frame, width):
    """Camera frame at the given width, with the target outline or the last crop area."""
    cfg, m, state = app.cfg, app.m, app.sm.state
    view, s = fit_width(frame, width)
    vh = view.shape[0]
    thick = 2 if width >= 640 else 1
    captured = state in (State.REVIEW, State.CAPTURED)
    bbox = app.last_bbox if captured else (m.bbox if m and state == State.SETTLING else None)
    if bbox:
        x, y, w, h = bbox
        p = cfg["crop_padding_px"] if captured else 0
        p0 = (max(0, round((x - p) * s)), max(0, round((y - p) * s)))
        p1 = (min(width - 1, round((x + w + p) * s)), min(vh - 1, round((y + h + p) * s)))
        if captured:
            dashed_rect(view, p0, p1, RED if state == State.REVIEW else GREEN,
                        dash=max(4, width // 100), thick=thick)
        else:
            cv2.rectangle(view, p0, p1, RED if m.touches_edge else YELLOW, thick)
    return view


def photo_view(app, w, h):
    """Last saved photo, large. Placeholder until the first photo exists."""
    if app.thumb is None:
        panel = np.full((h, w, 3), DARK, np.uint8)
        for i, (line, scale, col) in enumerate([
                ("No photo yet", 1.0, WHITE),
                ("The cropped photo appears here once a board is captured", 0.6, GREY)]):
            put(panel, line, ((w - text_width(line, scale, 2 if i == 0 else 1)) // 2,
                              h // 2 - 10 + i * 40), col, scale, 2 if i == 0 else 1)
        return panel
    panel = fit_inside(app.thumb, w, h)
    label = f"LAST PHOTO  {app.last_path.name}" if app.last_path else "LAST PHOTO"
    blend_rect(panel, (0, 0), (text_width(label, 0.55) + 24, 32), BLACK, 0.6)
    put(panel, label, (12, 22), CYAN, 0.55)
    return panel


def inset(main, small, label, color):
    """Paste a small view into the bottom-right corner of main."""
    mh, mw = main.shape[:2]
    sh, sw = small.shape[:2]
    if sh > mh - 40 or sw > mw - 24:
        return
    x0, y0 = mw - sw - 12, mh - sh - 12
    main[y0:y0 + sh, x0:x0 + sw] = small
    cv2.rectangle(main, (x0 - 2, y0 - 2), (x0 + sw + 1, y0 + sh + 1), color, 2)
    put(main, label, (x0, y0 - 8), color, 0.45)


# ---- operator screen --------------------------------------------------------

def render_operator(app, frame, now):
    cfg, sm, state = app.cfg, app.sm, app.sm.state
    width = cfg["preview_max_width"]
    fh, fw = frame.shape[:2]
    height = round(width * fh / fw)
    small_w = width // 5
    live_col = {State.SETTLING: YELLOW, State.REVIEW: RED}.get(state, WHITE)

    if app.live_large:
        view = live_view(app, frame, width)
        if app.thumb is not None:
            inset(view, fit_width(app.thumb, small_w)[0], "last photo", WHITE)
    else:
        view = photo_view(app, width, height)
        inset(view, live_view(app, frame, small_w), "LIVE", live_col)
    if state == State.REVIEW:
        cv2.rectangle(view, (0, 0), (width - 1, height - 1), RED, 4)
    flash_border(view, app, now)

    # Banner.
    title, instruction, col = state_text(app)
    banner = np.full((BANNER_H, width, 3), col, np.uint8)
    cv2.putText(banner, title, (16, 44), FONT, 1.1, BLACK, 3, cv2.LINE_AA)
    cv2.putText(banner, instruction, (30 + text_width(title, 1.1, 3), 40), FONT, 0.65, BLACK, 2,
                cv2.LINE_AA)
    progress = {State.LEARNING: app.det.learn_progress, State.SETTLING: sm.still_progress}
    if state in progress:
        cv2.rectangle(banner, (0, BANNER_H - 8), (int(width * progress[state]), BANNER_H), BLACK, -1)

    # Footer.
    footer = np.full((FOOTER_H, width, 3), DARK, np.uint8)
    put(footer, f"Photos this session: {app.session_count}", (14, 28), WHITE, 0.6)
    keys = "[R] reject  [L] large live  [B] background  [D] debug  [H] help  [Q] quit"
    put(footer, keys, (width - text_width(keys, 0.5) - 14, 28), GREY, 0.5)

    out = np.vstack([banner, view, footer])
    if app.show_help:
        help_overlay(out)
    return out
