import tensorflow as tf
import matplotlib.pyplot as plt


def prepare_dataset(
    train_dir: str,
    test_dir: str,
    validation_dir: str,
    width: int,
    height: int,
    batch_size: int,
    **kwargs,
):
    augmentation_kwargs = kwargs.get("augmentation_kwargs", {})
    if augmentation_kwargs == {}:
        assert False, f"augmentation_kwargs is {augmentation_kwargs}"

    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=augmentation_kwargs.get("rotation_range", 0),
        width_shift_range=augmentation_kwargs.get("width_shift_range", 0),
        height_shift_range=augmentation_kwargs.get("height_shift_range", 0),
        shear_range=augmentation_kwargs.get("shear_range", 0),
        zoom_range=augmentation_kwargs.get("zoom_range", 0),
        horizontal_flip=augmentation_kwargs.get("horizontal_flip", False),
        vertical_flip=False,
        rescale=1.0 / 255,  # normalize data
    )
    test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,  # normalize data
    )
    validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,  # normalize data
    )

    train_dataset = train_data_generator.flow_from_directory(
        directory=train_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=validation_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    test_dataset = test_data_generator.flow_from_directory(
        directory=test_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    return train_dataset, validation_dataset, test_dataset


def performance_metric(test_acc: float, total_parameter: int):
    return test_acc / (total_parameter / 10**5)


def plot_training_history(history, run_name: str, test_loss=None, test_accuracy=None):
    """
    Plots the training and validation loss and accuracy.

    Parameters:
    - history: The history object returned from the fit method of a Keras model.
    - test_loss (optional): Test loss.
    - test_accuracy (optional): Test accuracy.
    """
    # Plotting training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    if test_loss is not None:
        plt.axhline(y=test_loss, color="r", linestyle="--", label="Test Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    if test_accuracy is not None:
        plt.axhline(y=test_accuracy, color="r", linestyle="--", label="Test Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"/ghome/group05/week4_2/output/{run_name}.jpg")
    # plt.show()
