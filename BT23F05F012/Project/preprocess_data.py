import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import (
    create_hand_detector, extract_landmarks_from_image, normalize_landmarks,
    gesture_label_to_index, alphabet_label_to_index,
    GESTURE_LABELS, ALPHABET_LABELS,
    GESTURE_DATASET_DIR, ALPHABET_DATASET_DIR,
    GESTURE_X_PATH, GESTURE_Y_PATH,
    ALPHABET_X_PATH, ALPHABET_Y_PATH,
    ensure_dirs
)


def process_dataset(dataset_dir, labels, label_to_index_fn,
                    folder_prefix='', max_per_class=500):
    """
    Generic function to process any image dataset.
    Extracts hand landmarks from each image using MediaPipe.

    Args:
        dataset_dir    : Root folder containing class subfolders
        labels         : List of class label strings
        label_to_index_fn: Function to convert label to integer index
        folder_prefix  : Prefix on folder names (e.g. 'train_val_' for HaGRID)
        max_per_class  : Max images to process per class
    """
    detector = create_hand_detector(mode='image')
    X, y = [], []
    skipped = 0

    for label in labels:
        folder_name = folder_prefix + label
        label_dir = os.path.join(dataset_dir, folder_name)

        if not os.path.exists(label_dir):
            print(f"  [SKIP] Folder not found: {label_dir}")
            continue

        image_files = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:max_per_class]

        if not image_files:
            print(f"  [SKIP] No images in: {label}")
            continue

        for img_file in tqdm(image_files, desc=f"  {label:20s}", leave=False):
            img_path = os.path.join(label_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                skipped += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = extract_landmarks_from_image(image_rgb, detector)

            if landmarks is None:
                skipped += 1
                continue

            normalized = normalize_landmarks(landmarks)
            X.append(normalized)
            y.append(label_to_index_fn(label))

    detector.close()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), skipped


def preprocess_gestures(max_per_class=300):
    print("\n" + "="*55)
    print("PHASE 1: Processing HaGRID Gesture Dataset")
    print("="*55)

    X, y, skipped = process_dataset(
        dataset_dir=GESTURE_DATASET_DIR,
        labels=GESTURE_LABELS,
        label_to_index_fn=gesture_label_to_index,
        folder_prefix='train_val_',
        max_per_class=max_per_class
    )

    if len(X) == 0:
        print("[ERROR] No gesture data processed. Check dataset path.")
        return

    np.save(GESTURE_X_PATH, X)
    np.save(GESTURE_Y_PATH, y)

    print(f"\nGesture Preprocessing Done!")
    print(f"  Samples   : {len(X)}")
    print(f"  Skipped   : {skipped}")
    print(f"  Shape     : {X.shape}")
    print(f"  Saved to  : {GESTURE_X_PATH}")


def preprocess_alphabet(max_per_class=500):
    print("\n" + "="*55)
    print("PHASE 2: Processing ASL Alphabet Dataset")
    print("="*55)

    X, y, skipped = process_dataset(
        dataset_dir=ALPHABET_DATASET_DIR,
        labels=ALPHABET_LABELS,
        label_to_index_fn=alphabet_label_to_index,
        folder_prefix='',
        max_per_class=max_per_class
    )

    if len(X) == 0:
        print("[ERROR] No alphabet data processed. Check dataset path.")
        return

    np.save(ALPHABET_X_PATH, X)
    np.save(ALPHABET_Y_PATH, y)

    print(f"\nAlphabet Preprocessing Done!")
    print(f"  Samples   : {len(X)}")
    print(f"  Skipped   : {skipped}")
    print(f"  Shape     : {X.shape}")
    print(f"  Saved to  : {ALPHABET_X_PATH}")


if __name__ == "__main__":
    ensure_dirs()

    print("GestureSpeak - Data Preprocessing")
    print("This will take 15-30 minutes depending on your CPU.\n")

    # Comment out either one if you want to run separately
    preprocess_gestures(max_per_class=300)
    preprocess_alphabet(max_per_class=500)

    print("\n" + "="*55)
    print("All preprocessing complete! Ready to train.")
    print("="*55)
