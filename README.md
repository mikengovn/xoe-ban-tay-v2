# Robot Mimic — Hand Tracking Playground v2

A split-screen app: the **left panel** (top panel on phones) shows real-time webcam hand tracking, and the **right panel** (bottom on phones) shows a friendly white robot — rounded head, dark visor, glowing cyan eyes, antennas — that **mirrors your hand gestures live**: arm position, wrist rotation, and the curl of all five fingers. A Python desktop version of the tracker (`robot_mimic.py`) is included. Both are powered by the same model: the **MediaPipe Hand Landmarker** task from Google AI Edge.

---

## Credits: the Hand Landmarker model

All hand detection and tracking is performed by the **[MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)** task, created by **Google AI Edge (MediaPipe Solutions)**.

- **Model bundle:** [`HandLandmarker (full)`, float16](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) — one `.task` file packaging a **palm detection model** (finds hands in the frame) and a **hand landmark model** (predicts precise keypoints on the cropped hand). Trained on ~30K real-world images plus synthetic hand renders.
- **Model card:** [Hand Tracking (Lite/Full) with Fairness](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf)
- **Runtimes:** [`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision) (WebAssembly, browser) and the [`mediapipe`](https://pypi.org/project/mediapipe/) Python package.
- **Licenses:** MediaPipe code samples and models — Apache License 2.0; documentation and images — Creative Commons Attribution 4.0.
- Pipeline efficiency detail: in VIDEO mode the expensive palm detector only re-runs when tracking is lost; otherwise the previous frame's landmarks localize the hand for the next frame.

### The 21 hand landmarks

The model outputs 21 keypoints per hand as normalized coordinates (x, y in 0–1, plus depth z), together with world coordinates and handedness (left/right with confidence).

![The 21 hand landmarks detected by the MediaPipe Hand Landmarker](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png)

*Image © Google, from the [Hand Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

| Index | Landmark | Index | Landmark |
| --- | --- | --- | --- |
| 0 | WRIST | 11 | MIDDLE_FINGER_DIP |
| 1 | THUMB_CMC | 12 | MIDDLE_FINGER_TIP |
| 2 | THUMB_MCP | 13 | RING_FINGER_MCP |
| 3 | THUMB_IP | 14 | RING_FINGER_PIP |
| 4 | THUMB_TIP | 15 | RING_FINGER_DIP |
| 5 | INDEX_FINGER_MCP | 16 | RING_FINGER_TIP |
| 6 | INDEX_FINGER_PIP | 17 | PINKY_MCP |
| 7 | INDEX_FINGER_DIP | 18 | PINKY_PIP |
| 8 | INDEX_FINGER_TIP | 19 | PINKY_DIP |
| 9 | MIDDLE_FINGER_MCP | 20 | PINKY_TIP |
| 10 | MIDDLE_FINGER_PIP | | |

(MCP = knuckle, PIP/DIP = finger joints, IP = thumb joint, TIP = fingertip.)

---

## How the code makes hands trackable

Same four-step pipeline as before; the web app adds a fifth step that retargets the pose onto the robot.

### Step 1 — Load the model

The WebAssembly runtime is fetched with `FilesetResolver.forVisionTasks()`, then the landmarker is created from the official model URL with a GPU delegate and a CPU fallback:

```js
const vision = await FilesetResolver.forVisionTasks(WASM_URL);
const landmarker = await HandLandmarker.createFromOptions(vision, {
  baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" }, // falls back to CPU
  runningMode: "VIDEO",
  numHands: 2,
});
```

The Python version builds the same options via `vision.HandLandmarkerOptions` and auto-downloads the model on first run. `VIDEO` running mode makes the task exploit temporal continuity instead of running full detection every frame.

### Step 2 — Open the camera

Web: `getUserMedia({ video })` into a `<video>` element (requires `localhost`/HTTPS). Python: `cv2.VideoCapture(0)`.

### Step 3 — Detect landmarks every frame

Each new video frame is passed to `detectForVideo(video, timestamp)` (web) or `detect_for_video(mp_image, ms)` (Python). The result contains per hand: `landmarks` (21 normalized points), `world_landmarks`, and `handedness`.

### Step 4 — Left panel: skeleton overlay

Normalized landmarks are converted to pixels (`x * width`, `y * height`), with x flipped in mirror mode, and the skeleton is drawn from a fixed table of bone connections with joints color-coded per finger. Left/right labels are also flipped in mirror mode, since the model labels the raw image.

### Step 5 — Right panel: retargeting the pose onto the robot

This is the new part. Each detected hand is first **distilled into a compact pose signal** — everything the robot needs, nothing it doesn't:

```text
pose = { x, y            wrist position, normalized 0–1 (mirrored)
         rot             wrist rotation = atan2(middleMCP − wrist) vs "fingers up"
         curls[5]        per-finger curl in 0–1
         pinch }         thumb-tip↔index-tip / palm-size < 0.35
```

**Per-finger curl** is computed from joint angles, so it is independent of hand size and distance from the camera. For each finger chain (e.g. index = landmarks 5→6→7→8), the deviation from straight is summed at the two middle joints and normalized:

```text
curl = clamp( (π − angle@PIP) + (π − angle@DIP) ) / 2.4, 0..1 )   (thumb: /1.4)
```

An extended finger deviates ~0 rad → curl 0; a fist bends each joint ~70–90° → curl ≈ 1.

**Hand-to-arm assignment:** each hand claims the robot arm on its side of the screen (wrist x < 0.5 → robot's left arm), so the robot behaves like a mirror; with two hands, both arms animate independently.

**Fixed arm posture:** the shoulder→elbow→hand chain is a constant pose — elbows bent with each forearm raised upward in front of the body, like a "ready to mimic" stance. The hands are shown palm-forward toward the viewer, and the finger layout is mirrored per arm so both **thumbs point inward toward each other** while both **pinkies point outward**. The upper arms and elbows never move; all expressiveness is concentrated in the wrists and fingers, which keeps the motion readable and robotic.

**Wrist rotation sync:** the user's wrist angle (`atan2` of wrist→middle-knuckle, measured in mirrored coordinates so it matches what you see) is blended onto the robot's wrist with shortest-arc angle interpolation (so a tilt across the ±180° seam never causes a spin) and clamped to about ±72°, which keeps the palm facing the screen at all times.

**Articulated fingers:** the robot hand is a palm plate plus five segmented fingers (thumb 2 segments, others 3). Each segment rotates by `curl × bend-per-segment` relative to the previous one, folding toward the palm, so a real fist folds the robot's fingers over the palm, and a single raised index finger raises exactly one robot finger.

**Smoothness** comes from exponential smoothing (lerp) applied to every channel — wrist rotation, curls, pinch — with a snappy factor (~0.3/frame) while tracking and a gentle one (~0.08) when a hand disappears and the wrist and fingers relax back to an idle pose. The robot also has a life of its own: idle breathing bob, scheduled blinking, eyes that glance toward your hands, antenna tips that glow while mirroring, and a smile + "^ ^" eyes when you pinch.

---

## Files

| File | What it is |
| --- | --- |
| `index.html` | The complete split-screen app in one self-contained file (HTML + CSS + JS inline): tracking panel + robot mirror. Serve from `localhost`/HTTPS for camera access; panels stack vertically on phones. |
| `robot_mimic.py` | The Python desktop tracker (`pip install mediapipe opencv-python`, then `python robot_mimic.py`). Auto-downloads the model; Q quits, M toggles mirror, S toggles skeleton. |
| `README.md` | This document. |
| `LICENSE` | MIT license for this project's own code. |

---

## Ideas to take the robot further

- **Gesture commands** — recognize poses (open palm = wave hello, fist = power-down animation, thumbs-up = robot dances) using the same curl vector as a tiny feature space, or swap in MediaPipe's Gesture Recognizer task which adds a gesture classifier on top of the same landmarks.
- **Depth and reach** — use the landmark z values or `world_landmarks` to move the robot's arm toward/away from the "camera," scaling the hand to fake perspective.
- **A 3D robot** — replace the 2D canvas with a rigged glTF robot in three.js and drive its skeleton bones from the same pose signal; the retargeting math (curl + IK) carries over directly.
- **Emotions and memory** — let the robot get "bored" when ignored, excited when both hands appear, or mimic with a deliberate delay to play copy-cat games.
- **Control real hardware** — the pose signal is just numbers; stream `{rot, curls[5]}` over WebSerial/WebBluetooth to a servo-driven robot hand (e.g. an Arduino with 5 servos) for a physical mirror.
- **Two-player mode** — send one user's pose over WebRTC so the robot on your screen mirrors a friend's hand remotely.
- **Record & replay** — log pose signals with timestamps and replay them, turning gestures into reusable robot animations.

---

## Licenses

App code in this repository: MIT. MediaPipe Hand Landmarker model, runtimes, and sample code: Apache 2.0, © Google. Landmark diagram image: © Google, CC BY 4.0. The robot mimic idea is made by Michael Ngo.
