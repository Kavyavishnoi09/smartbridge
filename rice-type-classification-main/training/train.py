import os
import json

import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------
# Update these paths if needed
# -----------------------------
TRAIN_PATH = os.path.join("data", "train")
VAL_PATH = os.path.join("data", "validation")
MODEL_OUTPUT_PATH = os.path.join("training", "rice.h5")
CLASS_MAP_OUTPUT = os.path.join("training", "class_indices.json")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4


def main():
    if not os.path.isdir(TRAIN_PATH):
        raise FileNotFoundError(f"Training folder not found: {TRAIN_PATH}")
    if not os.path.isdir(VAL_PATH):
        raise FileNotFoundError(f"Validation folder not found: {VAL_PATH}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    num_classes = train_generator.num_classes

    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs("training", exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUTPUT_PATH, monitor="val_accuracy", save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    class_indices = train_generator.class_indices
    with open(CLASS_MAP_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("training", "accuracy_plot.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("training", "loss_plot.png"))
    plt.close()

    print("Training complete.")
    print(f"Best model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Class indices saved to: {CLASS_MAP_OUTPUT}")


if __name__ == "__main__":
    main()
