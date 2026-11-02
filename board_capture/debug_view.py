"""Debug view: shows what the detector sees and what it is waiting for.

Layout
  +-----------------------------+-----------+
  | live view + markers         | background|
  |                             | changed   |
  |                             | motion    |
  +-----------------------------+-----------+
  | checklist | gauges | state pipeline | last photo |
  +-----------------------------------------+
"""
import cv2
import numpy as np

from detector import State
from ui import (CYAN, DARK, GREEN, GREY, MAGENTA, ORANGE, RED, WHITE, YELLOW, blend_rect,
                dashed_rect, flash_border, help_overlay, put, state_text, text_width)

MAIN_W = 1100
SIDE_W = 360
STRIP_H = 190

PIPELINE = [State.LEARNING, State.ARMED_EMPTY, State.SETTLING, State.REVIEW, State.CAPTURED]


def gauge(img, x, y, w, label, value, vmax, markers, good):
    """Horizontal bar. markers: [(value, label)]. good: colour the fill green."""
    h = 14
    put(img, label, (x, y - 5), WHITE, 0.45)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * min(value / vmax, 1.0)) if vmax > 0 else 0
    cv2.rectangle(img, (x, y), (x + fill, y + h), GREEN if good else ORANGE, -1)
    for i, (mv, ml) in enumerate(markers):
        mx = x + int(w * min(mv / vmax, 1.0))
        cv2.line(img, (mx, y - 3), (mx, y + h + 3), WHITE, 2)
        # Alternate label sides so close markers don't overlap.
        tw = text_width(ml, 0.38)
        put(img, ml, (mx - tw - 3 if i % 2 == 0 else mx + 3, y + h + 14), GREY, 0.38)
    cv2.rectangle(img, (x, y), (x + w, y + h), GREY, 1)


def side_panel(img, label, h):
    panel = cv2.resize(img, (SIDE_W, h), interpolation=cv2.INTER_NEAREST)
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    put(panel, label, (8, 20), WHITE, 0.5)
    cv2.rectangle(panel, (0, 0), (SIDE_W - 1, h - 1), (70, 70, 70), 1)
    return panel


def tint_mask(mask, color):
    out = np.full((*mask.shape, 3), 18, np.uint8)
    out[mask > 0] = color
    return out


def render(app, frame, now):
    cfg, det, sm, m = app.cfg, app.det, app.sm, app.m
    state = sm.state
    src = frame
    fh, fw = src.shape[:2]
    main_h = round(MAIN_W * fh / fw)
    s = MAIN_W / fw                       # full-res -> main view
    main = cv2.resize(src, (MAIN_W, main_h), interpolation=cv2.INTER_AREA)
    # Only draw detector data that belongs to this frame's measurement.
    dbg = det.debug if m is not None else None

    # ---- markers on the live view ---------------------------------------
    if dbg is not None:
        dh, dw = dbg["bg_mask"].shape
        k = MAIN_W / dw                   # detection size -> main view

        # Movement telegraph: magenta where pixels changed since the last frame.
        mm = cv2.resize(dbg["motion_mask"], (MAIN_W, main_h), interpolation=cv2.INTER_NEAREST)
        main[mm > 0] = (main[mm > 0] * 0.5 + np.array(MAGENTA) * 0.5).astype(np.uint8)

        # Hand zone: anything reaching this border counts as a possible obstruction.
        e = max(3, round((cfg["edge_margin_px"] + 1) * k))
        edge_col = RED if any(t.touches_edge for t in m.objects) else (90, 90, 200)
        for p0, p1 in [((0, 0), (MAIN_W, e)), ((0, main_h - e), (MAIN_W, main_h)),
                       ((0, 0), (e, main_h)), ((MAIN_W - e, 0), (MAIN_W, main_h))]:
            blend_rect(main, p0, p1, edge_col, 0.45)
        put(main, "HAND ZONE", (e + 6, main_h - e - 8), edge_col, 0.45)

        # Objects: the target in bold, everything else greyed out and labelled ignored.
        target_id = m.target.id if m.target else None
        for t in sorted(m.objects, key=lambda t: t.id == target_id):
            c = (t.contour * k).astype(np.int32)
            x, y = round(t.bbox[0] * k), round(t.bbox[1] * k)
            tag = " moving" if t.moving else ""
            if t.id == target_id:
                col = RED if t.touches_edge else GREEN
                cv2.drawContours(main, [c], -1, col, 3)
                put(main, f"#{t.id} TARGET{tag}", (x + 4, y + 18), col, 0.5)
            else:
                cv2.drawContours(main, [c], -1, GREY, 2)
                put(main, f"#{t.id} ignored{tag}", (x + 4, y + 18), GREY, 0.45)

    captured = state in (State.REVIEW, State.CAPTURED)
    bbox = app.last_bbox if captured else (m.bbox if m else None)
    if bbox and state not in (State.ARMED_EMPTY, State.LEARNING):
        x, y, w, h = bbox
        p = cfg["crop_padding_px"]
        x0, y0 = max(0, round((x - p) * s)), max(0, round((y - p) * s))
        x1, y1 = min(MAIN_W - 1, round((x + w + p) * s)), min(main_h - 1, round((y + h + p) * s))
        dashed_rect(main, (x0, y0), (x1, y1), CYAN)
        put(main, "CROP", (x0 + 4, max(14, y0 - 6)), CYAN, 0.5)

        # Countdown ring to capture while settling.
        if state == State.SETTLING:
            frac = sm.still_progress
            cx, cy = x1 - 22, y0 + 22
            cv2.circle(main, (cx, cy), 16, (60, 60, 60), 4)
            cv2.ellipse(main, (cx, cy), (16, 16), -90, 0, 360 * frac, GREEN, 4)

    banner_col = {State.REVIEW: RED, State.CAPTURED: GREEN, State.SETTLING: YELLOW,
                  State.LEARNING: YELLOW}.get(state, WHITE)
    blend_rect(main, (0, 0), (MAIN_W, 34), (0, 0, 0), 0.55)
    put(main, f"{state}: {state_text(app)[1]}", (10, 23), banner_col, 0.55, 1)
    if state == State.LEARNING:
        blend_rect(main, (0, 34), (int(MAIN_W * det.learn_progress), 40), YELLOW, 1.0)
    if state == State.REVIEW:
        cv2.rectangle(main, (0, 0), (MAIN_W - 1, main_h - 1), RED, 6)

    # ---- side panels ----------------------------------------------------
    ph = main_h // 3
    if dbg is not None:
        changed = tint_mask(dbg["objects_mask"], (30, 90, 30))
        changed[dbg["bg_mask"] > 0] = GREEN
        for t in m.objects:
            x, y, w, h = t.bbox
            is_target = m.target is not None and t.id == m.target.id
            cv2.rectangle(changed, (x, y), (x + w, y + h), WHITE if is_target else GREY,
                          2 if is_target else 1)
        panels = [
            side_panel(dbg["background"], "Background (empty tray)", ph),
            side_panel(changed, f"Objects ({len(m.objects)}) - box = target", ph),
            side_panel(tint_mask(dbg["motion_mask"], MAGENTA), "Moving (frame to frame)", ph),
        ]
    else:
        blank = np.full((ph, SIDE_W, 3), DARK, np.uint8)
        panels = [side_panel(blank, t, ph) for t in ("Background: learning...", "", "")]
    side = np.vstack(panels)
    side = np.vstack([side, np.full((main_h - side.shape[0], SIDE_W, 3), DARK, np.uint8)])

    # ---- bottom strip ---------------------------------------------------
    W = MAIN_W + SIDE_W
    strip = np.full((STRIP_H, W, 3), DARK, np.uint8)
    put(strip, "LOOKING FOR", (14, 24), GREY, 0.5)
    if m is not None:
        present = m.present_frac > cfg["present_threshold"]
        still = m.motion_frac < cfg["still_threshold"]
        checks = [
            (present, "Something on the tray"),
            (still, f"Nothing moving  ({round(sm.still_progress * cfg['stable_frames'])}"
                    f"/{cfg['stable_frames']} still frames)"),
            (not m.touches_edge, "Clear of the hand zone"),
        ]
        for i, (ok, label) in enumerate(checks):
            y = 52 + i * 30
            cv2.rectangle(strip, (14, y - 14), (32, y + 4), GREEN if ok else RED, -1 if ok else 2)
            put(strip, label, (42, y), WHITE if ok else GREY, 0.5)

        gx, gw = 420, 330
        gauge(strip, gx, 40, gw, f"changed vs background  {m.present_frac:.3f}",
              m.present_frac, max(0.25, cfg["present_threshold"] * 4),
              [(cfg["empty_threshold"], "empty"), (cfg["present_threshold"], "present")], present)
        gauge(strip, gx, 92, gw, f"motion  {m.motion_frac:.4f}",
              m.motion_frac, cfg["still_threshold"] * 10, [(cfg["still_threshold"], "still")], still)
        if state in (State.CAPTURED, State.REVIEW):
            gauge(strip, gx, 144, gw, "tray clear -> re-arm",
                  sm.empty_progress, 1.0, [], True)
        else:
            gauge(strip, gx, 144, gw, "hold still -> capture",
                  sm.still_progress, 1.0, [], True)

    # State pipeline chips.
    px, py = 800, 30
    put(strip, "STATE", (px, py - 8), GREY, 0.5)
    for i, st in enumerate(PIPELINE):
        y = py + i * 28
        active = st == state
        cv2.rectangle(strip, (px, y), (px + 150, y + 22), banner_col if active else (55, 55, 55),
                      -1 if active else 1)
        put(strip, st, (px + 8, y + 16), (0, 0, 0) if active else GREY, 0.45)

    # Last photo.
    lx = 980
    put(strip, "LAST PHOTO", (lx, 24), GREY, 0.5)
    if app.thumb is not None:
        th = STRIP_H - 60
        t = cv2.resize(app.thumb, (round(th * app.thumb.shape[1] / app.thumb.shape[0]), th))
        t = t[:, : W - lx - 14]
        strip[34:34 + t.shape[0], lx:lx + t.shape[1]] = t
        put(strip, app.last_path.name, (lx, STRIP_H - 10), CYAN, 0.42)
    put(strip, f"[R]eject [B]ackground [D]ebug [H]elp [Q]uit   {app.fps:4.1f} fps",
        (14, STRIP_H - 12), GREY, 0.42)

    flash_border(main, app, now)
    out = np.vstack([np.hstack([main, side]), strip])
    if app.show_help:
        help_overlay(out)
    return out
