from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
EPOCHS = 5
BATCH_SIZE = 32
IMG_SIZE = 224
OUTPUT_PATH = REPO_ROOT / "models" / "checkpoints" / "mura_binary.keras"


def _load_mura_frame(data_root: Path, split: str) -> tuple[pd.Series, pd.Series]:
    mura = data_root / "MURA-v1.1"
    studies = pd.read_csv(
        mura / f"{split}_labeled_studies.csv",
        header=None,
        names=["study_dir", "label"],
        dtype={"study_dir": str, "label": int},
    )
    images = pd.read_csv(
        mura / f"{split}_image_paths.csv",
        header=None,
        names=["rel_path"],
        dtype=str,
    )
    images["study_dir"] = images["rel_path"].apply(
        lambda p: f"{Path(p).parent.as_posix()}/"
    )
    merged = images.merge(studies, on="study_dir", how="inner")
    if merged.empty:
        raise SystemExit(
            f"No rows after joining {split} image paths with labeled studies "
            f"(check that paths under {mura} match the CSV prefixes)."
        )
    paths = (data_root / merged["rel_path"].str.strip()).map(lambda p: p.as_posix())
    labels = merged["label"].astype("float32")
    return paths, labels


def _decode_image(path: tf.Tensor, label: tf.Tensor, size: int) -> tuple[tf.Tensor, tf.Tensor]:
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [size, size])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img, label


def _make_dataset(
    paths: list[str],
    labels: list[float],
    *,
    size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 10_000), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: _decode_image(p, y, size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_mura_resnet_model(img_size: int) -> tf.keras.Model:
    """
    Deep CNN: ResNet50V2 feature extractor + batch-normalized MLP head.
    Backbone weights are frozen for this baseline script.
    """
    backbone = tf.keras.applications.ResNet50V2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        name="radiograph_backbone",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.5, name="drop1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dropout(0.35, name="drop2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob_abnormal")(x)

    model = tf.keras.Model(inputs, outputs, name="mura_resnet50v2_binary")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    data_root = DATA_ROOT.resolve()

    train_paths, train_y = _load_mura_frame(data_root, "train")
    val_paths, val_y = _load_mura_frame(data_root, "valid")

    train_ds = _make_dataset(
        train_paths.tolist(),
        train_y.tolist(),
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_ds = _make_dataset(
        val_paths.tolist(),
        val_y.tolist(),
        size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train images: {len(train_paths)}, Valid images: {len(val_paths)}")
    print(f"Architecture: ResNet50V2 (frozen) + dense head | {EPOCHS} epochs")

    model = build_mura_resnet_model(IMG_SIZE)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
