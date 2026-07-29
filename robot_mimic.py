#!/usr/bin/env python3
"""
Hand Tracking Playground — Python version
=========================================

Real-time webcam hand tracking built on the MediaPipe Hand Landmarker task
(Google AI Edge). Detects up to two hands, draws all 21 landmarks per hand
with per-finger colors, shows handedness (left/right) and a scale-invariant
pinch indicator, and prints FPS.

Docs: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

Setup
-----
    pip install mediapipe opencv-python

Run
---
    python hand_tracking.py

The script downloads the official model bundle (hand_landmarker.task, ~7 MB)
to the current folder on first run. Press Q or Esc to quit, M to toggle
mirroring, S to toggle the skeleton.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# The 21 landmarks, indexed 0-20 (see the official landmark diagram):
# 0 WRIST, 1-4 thumb (CMC, MCP, IP, TIP), 5-8 index (MCP, PIP, DIP, TIP),
# 9-12 middle, 13-16 ring, 17-20 pinky.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
FINGERTIPS = (4, 8, 12, 16, 20)

# Per-finger colors (BGR for OpenCV)
FINGER_COLORS = {
    "thumb":  ((78, 193, 242), (1, 2, 3, 4)),
    "index":  ((92, 81, 232), (5, 6, 7, 8)),
    "middle": ((190, 192, 91), (9, 10, 11, 12)),
    "ring":   ((242, 111, 125), (13, 14, 15, 16)),
    "pinky":  ((224, 123, 224), (17, 18, 19, 20)),
}

# Skeleton bone connections (landmark index pairs)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)

PINCH_RATIO = 0.35  # thumb-index distance / palm size; lower = stricter


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ensure_model() -> str:
    """Download the official model bundle on first run."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def color_of(idx: int):
    """Marker color for a landmark index."""
    for bgr, ids in FINGER_COLORS.values():
        if idx in ids:
            return bgr
    return (255, 255, 255)  # wrist


def is_pinching(lms) -> bool:
    """Scale-invariant pinch: thumb-tip to index-tip distance divided by
    palm size (wrist to middle-finger MCP), so it works at any distance
    from the camera."""
    dx, dy = lms[THUMB_TIP].x - lms[INDEX_TIP].x, lms[THUMB_TIP].y - lms[INDEX_TIP].y
    px, py = lms[WRIST].x - lms[MIDDLE_MCP].x, lms[WRIST].y - lms[MIDDLE_MCP].y
    pinch_d = (dx * dx + dy * dy) ** 0.5
    palm_d = (px * px + py * py) ** 0.5
    return palm_d > 0 and pinch_d / palm_d < PINCH_RATIO


def draw_hand(frame, lms, label: str, pinch: bool):
    """Draw bones, joints, and a per-hand info tag onto the BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (235, 235, 235), 2, cv2.LINE_AA)

    for i, p in enumerate(pts):
        r = 7 if i in FINGERTIPS else 9 if i == WRIST else 4
        cv2.circle(frame, p, r, color_of(i), -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, (20, 20, 20), 1, cv2.LINE_AA)

    tag = f"{label}  {'PINCH' if pinch else 'open'}"
    tx, ty = pts[WRIST][0] - 30, pts[WRIST][1] + 34
    cv2.putText(frame, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 255, 120) if pinch else (255, 255, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # 1) Create the Hand Landmarker in VIDEO mode
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # 2) Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (device 0). "
                         "Close other apps using the camera and try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mirror, show_skeleton = True, True
    t_prev, fps = time.time(), 0.0
    print("Running. Q/Esc: quit | M: mirror | S: skeleton")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        # 3) Detect: BGR -> RGB, wrap as mp.Image, monotonic timestamp in ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        # 4) Draw results
        if show_skeleton and result.hand_landmarks:
            for lms, handed in zip(result.hand_landmarks, result.handedness):
                label = handed[0].category_name
                if mirror:  # labels refer to the un-mirrored image
                    label = {"Left": "Right", "Right": "Left"}.get(label, label)
                draw_hand(frame, lms, label, is_pinching(lms))

        # FPS (exponential smoothing)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(frame, f"{fps:5.1f} fps", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:5.1f} fps", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Tracking Playground (Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            mirror = not mirror
        elif key == ord("s"):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
