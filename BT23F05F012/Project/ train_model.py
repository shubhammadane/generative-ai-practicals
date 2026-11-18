import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, avoids DLL issues
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils import (
    GESTURE_X_PATH, GESTURE_Y_PATH, ALPHABET_X_PATH, ALPHABET_Y_PATH,
    GESTURE_MODEL_PATH, ALPHABET_MODEL_PATH,
    NUM_GESTURE_CLASSES, NUM_ALPHABET_CLASSES,
    FEATURE_SIZE, GESTURE_LABELS, ALPHABET_LABELS,
    MODEL_DIR, ensure_dirs
)


def load_data(X_path, y_path, name):
    print(f"\nLoading {name} data...")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {len(np.unique(y))}")
    return X, y


def build_model(num_classes):
    """
    Fully-connected neural network for gesture/alphabet classification.
    Input  : 63 landmark features
    Output : probability over num_classes
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(FEATURE_SIZE,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history, name):
    """Save training accuracy/loss plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title(f'{name} Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title(f'{name} Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = f"{MODEL_DIR}/{name.lower()}_training.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Training plot saved: {save_path}")


def train(X_path, y_path, model_path, num_classes, labels, name):
    X, y = load_data(X_path, y_path, name)

    # One-hot encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    # 70% train / 15% val / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\n  Train : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}")

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]

    print(f"\nTraining {name} model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Final test evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*45}")
    print(f"{name} Model Results:")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Saved to      : {model_path}")
    print(f"{'='*45}")

    plot_history(history, name)
    return test_acc


if __name__ == "__main__":
    ensure_dirs()

    import os

    print("GestureSpeak - Model Training")
    print("="*45)

    # Train alphabet model
    if os.path.exists(ALPHABET_X_PATH):
        train(
            X_path=ALPHABET_X_PATH,
            y_path=ALPHABET_Y_PATH,
            model_path=ALPHABET_MODEL_PATH,
            num_classes=NUM_ALPHABET_CLASSES,
            labels=ALPHABET_LABELS,
            name="Alphabet"
        )
    else:
        print("[SKIP] Alphabet data not found. Run preprocess_data.py first.")

    # Train gesture model (runs after gesture data is ready)
    if os.path.exists(GESTURE_X_PATH):
        train(
            X_path=GESTURE_X_PATH,
            y_path=GESTURE_Y_PATH,
            model_path=GESTURE_MODEL_PATH,
            num_classes=NUM_GESTURE_CLASSES,
            labels=GESTURE_LABELS,
            name="Gesture"
        )
    else:
        print("[SKIP] Gesture data not found. Run preprocess_data.py first.")

    print("\nAll training complete!")
