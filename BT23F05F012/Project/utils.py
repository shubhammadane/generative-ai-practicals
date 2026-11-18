import os
import numpy as np

# ── MediaPipe Tasks API (mediapipe 0.10.x) ─────────────────────────────────────
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GESTURE_DATASET_DIR  = os.path.join(BASE_DIR, 'dataset', 'gestures',
                                    'hagrid-sample-500k-384p', 'hagrid_500k')
ALPHABET_DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'alphabet',
                                    'asl_alphabet_train', 'asl_alphabet_train')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
DATA_DIR    = os.path.join(BASE_DIR, 'data')

GESTURE_MODEL_PATH  = os.path.join(MODEL_DIR, 'gesture_model.h5')
ALPHABET_MODEL_PATH = os.path.join(MODEL_DIR, 'alphabet_model.h5')

GESTURE_X_PATH  = os.path.join(DATA_DIR, 'gesture_X.npy')
GESTURE_Y_PATH  = os.path.join(DATA_DIR, 'gesture_y.npy')
ALPHABET_X_PATH = os.path.join(DATA_DIR, 'alphabet_X.npy')
ALPHABET_Y_PATH = os.path.join(DATA_DIR, 'alphabet_y.npy')

# MediaPipe hand landmarker model path (downloaded automatically)
HAND_LANDMARKER_MODEL = os.path.join(BASE_DIR, 'model', 'hand_landmarker.task')

# ── Gesture Labels & Phrase Mapping ───────────────────────────────────────────
GESTURE_LABELS = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute',
    'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock',
    'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted'
]

GESTURE_TO_PHRASE = {
    'call'           : "Please call me",
    'dislike'        : "No, I disagree",
    'fist'           : "Stop",
    'four'           : "I am hungry",
    'like'           : "Yes, I agree",
    'mute'           : "Please be quiet",
    'ok'             : "Okay, understood",
    'one'            : "I need help",
    'palm'           : "Hello",
    'peace'          : "I am fine, thank you",
    'peace_inverted' : "I don't understand",
    'rock'           : "That is great",
    'stop'           : "Please wait",
    'stop_inverted'  : "I am leaving now",
    'three'          : "I am thirsty",
    'three2'         : "Please repeat that",
    'two_up'         : "Thank you very much",
    'two_up_inverted': "Good morning",
}

# ── Alphabet Labels ────────────────────────────────────────────────────────────
ALPHABET_LABELS = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','space'
]

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_SIZE             = 63
NUM_GESTURE_CLASSES      = len(GESTURE_LABELS)
NUM_ALPHABET_CLASSES     = len(ALPHABET_LABELS)
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5


def download_hand_model():
    """Download MediaPipe hand landmarker model if not present."""
    import urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(HAND_LANDMARKER_MODEL):
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, HAND_LANDMARKER_MODEL)
        print("Model downloaded.")


def create_hand_detector(mode='image'):
    """
    Create a HandLandmarker detector.
    mode: 'image' for static images, 'video' for webcam frames
    """
    download_hand_model()
    running_mode = (VisionTaskRunningMode.IMAGE if mode == 'image'
                    else VisionTaskRunningMode.VIDEO)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(image_rgb_np, detector):
    """
    Extract 21 hand landmark coordinates from an RGB numpy array (static image).
    Returns flat numpy array of 63 values, or None if no hand detected.
    """
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb_np)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)
    return None


def extract_landmarks_from_video(image_rgb_np, detector, timestamp_ms):
    """
    Extract landmarks from a video frame using timestamp.
    Required for VIDEO running mode (webcam).
    """
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb_np)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if result.hand_landmarks:
        landmarks = []
        for lm in result.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32), result
    return None, result


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to wrist (landmark 0).
    Makes model position and scale independent.
    """
    if landmarks is None:
        return None
    lm = landmarks.reshape(21, 3)
    wrist = lm[0].copy()
    lm = lm - wrist
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm = lm / max_val
    return lm.flatten()


def gesture_label_to_index(label):
    return GESTURE_LABELS.index(label)

def gesture_index_to_label(index):
    return GESTURE_LABELS[index]

def alphabet_label_to_index(label):
    return ALPHABET_LABELS.index(label)

def alphabet_index_to_label(index):
    return ALPHABET_LABELS[index]

def ensure_dirs():
    for d in [MODEL_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)
