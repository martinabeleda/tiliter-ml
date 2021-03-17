import argparse
import enum
import dataclasses
import logging
from pathlib import Path
from typing import Tuple, Dict

import tensorflow as tf
from tensorflow.keras import Model, layers
import tensorflow_datasets as tfds


# Let's define some config here although this usually would be stored in some kind of config file or server
CONFIG = {
    "mnist": {
        "log_level": "INFO",
        "input_shape": (28, 28, 3),
        "num_classes": 10,
        "batch_size": 64,
        "train_epochs": 2,
        "data_dir": "data/mnist",
    },
    "flowers": {
        "log_level": "INFO",
        "input_shape": (256, 256, 3),
        "num_classes": 5,
        "batch_size": 16,
        "train_epochs": 11,
        "data_dir": "data/flowers",
    },
}


class Datasets(enum.Enum):
    """A class for the supported training datasets"""

    FLOWERS = "flowers"
    MNIST = "mnist"


@dataclasses.dataclass(frozen=True)
class Config:
    """A class for storing the configuration of a training run"""

    log_level: str
    input_shape: Tuple[int, int, int]
    num_classes: int
    batch_size: int
    train_epochs: int
    data_dir: str


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> Model:
    """Builds a simple CNN model for classification

    Args:
        input_shape (Tuple[int, int, int]): The input shape of the model

    Returns:
        Model: An uncompiled Keras model
    """
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu")(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(input_layer, output)
    return model


def build_mnist_dataset(data_dir: str, batch_size: int, split: str = "training") -> tf.data.Dataset:
    """Builds the MNIST training dataset from file with pre-processing

    Args:
        data_dir (str): The parent directory of the dataset
        batch_size (int): The batch size
        split (str, optional): The dataset split. Defaults to "training".

    Returns:
        tf.data.Dataset
    """
    builder = tfds.ImageFolder(data_dir)
    dataset = builder.as_dataset(split=split, shuffle_files=True)

    def scale(element: Dict):
        image = tf.cast(element["image"], tf.float32)
        image = tf.divide(
            image,
            tf.constant(
                [
                    255.0,
                ],
                dtype=tf.float32,
            ),
        )
        return image, element["label"]

    dataset = dataset.map(scale).batch(batch_size)
    return dataset


def build_flowers_dataset(
    data_dir: str, batch_size: int, image_size: Tuple[int, int], split: str = "training", augment: bool = False
) -> tf.data.Dataset:
    """Builds the flowers training dataset from file with pre-processing and optional augmentation

    Args:
        data_dir (str): The parent directory of the dataset
        batch_size (int): The batch size
        image_size (Tuple[int, int]): Height and width of the image
        split (str, optional): The dataset split. Defaults to "training".
        augment (bool, optional): Apply data augmentations Defaults to False.

    Returns:
        tf.data.Dataset
    """
    # Load the dataset from file, fix the seed so we get a deterministic train/test split
    # Note that the spec sheet asked to generate the split using sklearn. I decided to use the native keras function
    # to avoid a bunch of expensive re-packaging of the dataset.
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, batch_size=batch_size, image_size=image_size, seed=1, validation_split=0.2, subset=split
    )
    # Since our images are not consistently sized, let's resize them to the model context
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    logging.debug(f"Dataset has {len(dataset)} members")

    data_augmentation = tf.keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    return dataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", help="The  ID of the config to use", type=str, required=True)
    args = parser.parse_args()

    config = Config(**CONFIG[args.config_id])
    logging.basicConfig(level=logging.getLevelName(config.log_level))
    logging.debug(f"Training config: {config}")

    dataset = Datasets(args.config_id)
    if dataset == Datasets.FLOWERS:
        training_dataset = build_flowers_dataset(
            config.data_dir, config.batch_size, config.input_shape[:2], "training", augment=True
        )
        test_dataset = build_flowers_dataset(config.data_dir, config.batch_size, config.input_shape[:2], "validation")
    elif dataset == Datasets.MNIST:
        training_dataset = build_mnist_dataset(config.data_dir, config.batch_size, "training")
        test_dataset = build_mnist_dataset(config.data_dir, config.batch_size, "testing")
    else:
        raise ValueError(f"Unsupported dataset: {dataset.value}")

    # Build and compile the model
    model = build_model(config.input_shape, config.num_classes)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        model.summary(print_fn=logging.debug)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    logging.info(f"Training with: batch size - {config.batch_size} for {len(training_dataset)} steps")
    try:
        model.fit(
            training_dataset.repeat(),
            epochs=config.train_epochs,
            steps_per_epoch=len(training_dataset),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(filepath="data"),
            ],
        )
    except KeyboardInterrupt:
        logging.warning("Received KeyboardInterrupt, stopping training early")

    logging.info(
        f"Completed training, running evaluation with: batch size - {config.batch_size} for {len(test_dataset)} steps"
    )
    results = model.evaluate(test_dataset, batch_size=config.batch_size, return_dict=True)
    for metric, value in results.items():
        logging.info(f"Test {metric}: {value}")


if __name__ == "__main__":
    main()
