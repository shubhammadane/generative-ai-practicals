import cv2
import numpy as np
import tensorflow as tf
import time
import os
from collections import deque, Counter
from utils import (
    create_hand_detector, extract_landmarks_from_video, normalize_landmarks,
    gesture_index_to_label, alphabet_index_to_label,
    GESTURE_TO_PHRASE, GESTURE_MODEL_PATH, ALPHABET_MODEL_PATH,
    MIN_DETECTION_CONFIDENCE
)
from text_to_speech import TextToSpeech

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.85   # Minimum prediction confidence to accept
GESTURE_HOLD_FRAMES   = 20     # Frames gesture must be stable before accepting
LETTER_COOLDOWN       = 1.5    # Seconds between adding same letter
SPEAK_PAUSE           = 3.0    # Seconds of no hand before auto-speaking
SMOOTHING_WINDOW      = 10     # Frames for prediction smoothing

# Modes
MODE_ALPHABET = 'ALPHABET'
MODE_GESTURE  = 'GESTURE'


class GestureSpeakApp:

    def __init__(self):
        print("Loading models...")
        self.alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL_PATH)
        print("  Alphabet model loaded ✓")

        self.gesture_model = None
        if os.path.exists(GESTURE_MODEL_PATH):
            self.gesture_model = tf.keras.models.load_model(GESTURE_MODEL_PATH)
            print("  Gesture model loaded ✓")
        else:
            print("  Gesture model not found — GESTURE mode unavailable")

        self.tts = TextToSpeech(rate=150)
        self.detector = create_hand_detector(mode='video')

        # App state
        self.mode         = MODE_ALPHABET
        self.current_word = []
        self.sentence     = []

        # Prediction smoothing
        self.pred_buffer  = deque(maxlen=SMOOTHING_WINDOW)

        # Timing
        self.last_letter_time  = 0
        self.last_gesture_time = time.time()
        self.start_time        = time.time()

        # Stability tracking
        self.stable_label = None
        self.stable_count = 0

        print("\nGestureSpeak ready!")
        print("Controls: M=toggle mode | S=speak | C=clear | Q=quit\n")

    def predict(self, landmarks):
        """Run inference and return (label, confidence)."""
        data = landmarks.reshape(1, -1)
        if self.mode == MODE_ALPHABET:
            preds = self.alphabet_model.predict(data, verbose=0)[0]
        else:
            if self.gesture_model is None:
                return None, 0.0
            preds = self.gesture_model.predict(data, verbose=0)[0]

        confidence = float(np.max(preds))
        index      = int(np.argmax(preds))

        if self.mode == MODE_ALPHABET:
            label = alphabet_index_to_label(index)
        else:
            label = gesture_index_to_label(index)

        return label, confidence

    def smooth_prediction(self, label, confidence):
        """Return most frequent prediction in recent buffer."""
        if confidence >= CONFIDENCE_THRESHOLD:
            self.pred_buffer.append(label)

        if not self.pred_buffer:
            return None, 0.0

        counts = Counter(self.pred_buffer)
        top_label, count = counts.most_common(1)[0]
        return top_label, count / len(self.pred_buffer)

    def handle_confirmed_label(self, label):
        """Process a stable confirmed label into text buffer."""
        now = time.time()
        if now - self.last_letter_time < LETTER_COOLDOWN:
            return

        if self.mode == MODE_GESTURE:
            phrase = GESTURE_TO_PHRASE.get(label, label)
            print(f"Gesture: {label} → '{phrase}'")
            self.tts.speak(phrase)
            self.last_letter_time = now
            return

        # Alphabet mode
        if label == 'space':
            if self.current_word:
                self.sentence.append(''.join(self.current_word))
                self.current_word = []
                print(f"Word added | Sentence: {' '.join(self.sentence)}")
        elif label == 'del':
            if self.current_word:
                self.current_word.pop()
                print(f"Deleted | Word: {''.join(self.current_word)}")
        else:
            self.current_word.append(label)
            print(f"Letter: {label} | Word: {''.join(self.current_word)}")

        self.last_letter_time = now

    def handle_stability(self, label):
        """Only confirm a label after it's held for GESTURE_HOLD_FRAMES."""
        if label == self.stable_label:
            self.stable_count += 1
        else:
            self.stable_label = label
            self.stable_count = 1

        if self.stable_count >= GESTURE_HOLD_FRAMES:
            self.handle_confirmed_label(label)
            self.stable_count = 0

    def speak_sentence(self):
        """Speak and clear current sentence buffer."""
        words = self.sentence + ([(''.join(self.current_word))] if self.current_word else [])
        full  = ' '.join(words).strip()
        if full:
            print(f"\nSpeaking: '{full}'")
            self.tts.speak(full)
            self.sentence     = []
            self.current_word = []

    def check_auto_speak(self):
        """Auto-speak after SPEAK_PAUSE seconds of no hand detected."""
        if time.time() - self.last_gesture_time > SPEAK_PAUSE:
            self.speak_sentence()
            self.last_gesture_time = time.time()

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand skeleton on frame."""
        if not hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        # Draw connections
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        return frame

    def draw_ui(self, frame, label, confidence):
        """Draw all UI elements."""
        h, w = frame.shape[:2]

        # Bottom panel
        cv2.rectangle(frame, (0, h - 130), (w, h), (20, 20, 20), -1)

        # Mode indicator
        mode_color = (0, 200, 255) if self.mode == MODE_ALPHABET else (0, 255, 150)
        cv2.rectangle(frame, (0, 0), (200, 35), (20, 20, 20), -1)
        cv2.putText(frame, f"MODE: {self.mode}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

        # Prediction label
        if label:
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, f"Detected: {label} ({confidence:.0%})",
                        (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Stability bar
        if self.stable_label and self.stable_count > 0:
            bar = int((self.stable_count / GESTURE_HOLD_FRAMES) * 180)
            cv2.rectangle(frame, (w - 200, 10), (w - 20, 32), (50, 50, 50), -1)
            cv2.rectangle(frame, (w - 200, 10), (w - 200 + bar, 32), (0, 220, 100), -1)
            cv2.putText(frame, "Hold", (w - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Current word
        word = ''.join(self.current_word) if self.current_word else '_'
        cv2.putText(frame, f"Word: {word}",
                    (10, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Sentence
        sent = ' '.join(self.sentence)
        cv2.putText(frame, f"Sentence: {sent}",
                    (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Controls
        cv2.putText(frame, "M=mode  S=speak  C=clear  Q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame       = cv2.flip(frame, 1)
            image_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp   = int((time.time() - self.start_time) * 1000)
            frame_count += 1

            label, confidence   = None, 0.0
            hand_landmarks_list = None

            landmarks, result = extract_landmarks_from_video(
                image_rgb, self.detector, timestamp
            )

            if landmarks is not None:
                self.last_gesture_time = time.time()
                hand_landmarks_list    = result.hand_landmarks[0] if result.hand_landmarks else None

                normalized          = normalize_landmarks(landmarks)
                raw_label, raw_conf = self.predict(normalized)
                label, confidence   = self.smooth_prediction(raw_label, raw_conf)

                if label:
                    self.handle_stability(label)
            else:
                self.pred_buffer.clear()
                self.stable_label = None
                self.stable_count = 0
                self.check_auto_speak()

            frame = self.draw_landmarks(frame, hand_landmarks_list)
            frame = self.draw_ui(frame, label, confidence)

            cv2.imshow('GestureSpeak', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                if self.gesture_model:
                    self.mode = MODE_GESTURE if self.mode == MODE_ALPHABET else MODE_ALPHABET
                    self.pred_buffer.clear()
                    print(f"Switched to {self.mode} mode")
                else:
                    print("Gesture model not available. Train gesture model first.")
            elif key == ord('s'):
                self.speak_sentence()
            elif key == ord('c'):
                self.sentence     = []
                self.current_word = []
                print("Cleared.")

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("GestureSpeak closed.")


if __name__ == "__main__":
    app = GestureSpeakApp()
    app.run()
