"""
Real-Time Person Detection with Age, Gender & Emotion Analysis
Uses: YOLOv8 (person detection) + DeepFace (age/gender/emotion)
Author: Generated Project
"""

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from deepface import DeepFace

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0          # 0 = default laptop camera
YOLO_MODEL        = "yolov8n.pt"  # nano = fastest; swap for yolov8s/m/l for accuracy
CONFIDENCE        = 0.50       # YOLO confidence threshold
ANALYZE_EVERY_N   = 10         # Run DeepFace every N frames (performance control)
DISPLAY_WIDTH     = 1280
DISPLAY_HEIGHT    = 720

# Color palette (BGR)
COLORS = {
    "box":      (0, 200, 255),
    "male":     (255, 100,  50),
    "female":   (50,  150, 255),
    "label_bg": (20,   20,  20),
    "text":     (255, 255, 255),
    "fps":      (0,   255, 120),
}

EMOTION_EMOJI = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "surprise": "😲",
    "fear":     "😨",
    "disgust":  "🤢",
    "neutral":  "😐",
}
# ─────────────────────────────────────────────────────────────────────────────


class PersonAnalyzer:
    """
    Runs DeepFace analysis in a background thread so it doesn't block
    the main video loop.
    """

    def __init__(self):
        self.results: dict = {}   # person_id -> analysis dict
        self._lock = threading.Lock()
        self._queue: list = []    # list of (person_id, face_crop)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, person_id: int, face_crop: np.ndarray):
        with self._lock:
            # Keep only latest request per person
            self._queue = [q for q in self._queue if q[0] != person_id]
            self._queue.append((person_id, face_crop.copy()))

    def get(self, person_id: int) -> dict:
        with self._lock:
            return self.results.get(person_id, {})

    def _worker(self):
        while True:
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.pop(0)

            if task is None:
                time.sleep(0.01)
                continue

            person_id, crop = task
            try:
                analysis = DeepFace.analyze(
                    crop,
                    actions=["age", "gender", "emotion", "race"],
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]

                result = {
                    "age":     int(analysis.get("age", 0)),
                    "gender":  analysis.get("dominant_gender", "Unknown"),
                    "emotion": analysis.get("dominant_emotion", "neutral"),
                    "race":    analysis.get("dominant_race", ""),
                    "gender_scores": analysis.get("gender", {}),
                    "emotion_scores": analysis.get("emotion", {}),
                }
                with self._lock:
                    self.results[person_id] = result

            except Exception:
                pass  # Face not clear enough — keep previous result


def draw_label(frame, text, x, y, color, font_scale=0.55, thickness=1):
    """Draw a filled-background label on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  COLORS["label_bg"], cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw 4 lines + 4 corner arcs
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def age_category(age: int) -> str:
    if age < 13:   return "Child"
    if age < 18:   return "Teen"
    if age < 30:   return "Young Adult"
    if age < 50:   return "Adult"
    if age < 65:   return "Middle-Aged"
    return "Senior"


def run():
    print("Loading YOLOv8 model …")
    model = YOLO(YOLO_MODEL)
    # Warm-up
    model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)

    analyzer = PersonAnalyzer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    if not cap.isOpened():
        print(f"❌  Cannot open camera index {CAMERA_INDEX}")
        return

    print("✅  Camera opened. Press  Q  to quit,  S  to save screenshot.")

    frame_idx   = 0
    fps_display = 0.0
    t_prev      = time.time()
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_idx += 1

        # ── FPS ──────────────────────────────────────────────────────────────
        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev      = t_now

        # ── YOLO detection ───────────────────────────────────────────────────
        results   = model(frame, conf=CONFIDENCE, classes=[0], verbose=False)
        boxes     = results[0].boxes  # class 0 = person

        person_count = 0

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf            = float(box.conf[0])
            person_count   += 1

            # ── Face crop (upper 40 % of bounding box) ───────────────────────
            face_h  = int((y2 - y1) * 0.40)
            face_y2 = min(y1 + face_h, frame.shape[0])
            face_x1 = max(x1, 0);  face_x2 = min(x2, frame.shape[1])
            face_crop = frame[y1:face_y2, face_x1:face_x2]

            # Submit for analysis every N frames
            if frame_idx % ANALYZE_EVERY_N == 0 and face_crop.size > 0:
                analyzer.submit(i, face_crop)

            info = analyzer.get(i)

            # ── Choose box color based on gender ────────────────────────────
            gender  = info.get("gender", "")
            box_col = COLORS["male"] if gender == "Man" else \
                      COLORS["female"] if gender == "Woman" else COLORS["box"]

            draw_rounded_rect(frame, (x1, y1), (x2, y2), box_col, thickness=2)

            # ── Overlay info ─────────────────────────────────────────────────
            label_y = y1 - 8
            if label_y < 20:
                label_y = y2 + 20

            if info:
                age     = info.get("age", 0)
                emotion = info.get("emotion", "neutral")
                race    = info.get("race", "")
                emoji   = EMOTION_EMOJI.get(emotion, "")

                g_display = "♂ Male"   if gender == "Man"   else \
                            "♀ Female" if gender == "Woman" else "? Unknown"

                lines = [
                    f"#{i+1}  {g_display}",
                    f"Age: {age} ({age_category(age)})",
                    f"Mood: {emotion.capitalize()} {emoji}",
                    f"Conf: {conf:.0%}",
                ]
                if race:
                    lines.append(f"Race: {race.capitalize()}")

                for j, line in enumerate(lines):
                    draw_label(frame, line, x1, label_y + j * 22,
                               COLORS["text"], font_scale=0.52)
            else:
                draw_label(frame, f"#{i+1}  Analyzing…", x1, label_y,
                           COLORS["text"], font_scale=0.52)

        # ── HUD ──────────────────────────────────────────────────────────────
        hud = f"FPS: {fps_display:.1f}  |  Persons: {person_count}  |  [Q] Quit  [S] Save"
        draw_label(frame, hud, 10, 28, COLORS["fps"], font_scale=0.6, thickness=1)

        cv2.imshow("Person Detector — Age / Gender / Emotion", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸  Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    run()