import os
import numpy as np
import random
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import (
    SGD,
    RMSprop,
    Adam,
    AdamW,
    Adadelta,
    Adagrad,
    Adamax,
    Adafactor,
    Nadam,
    Ftrl,
)


def plot_hist(history, job_id):
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("accuracy_{}.jpg".format(job_id))
    plt.close()

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("loss_{}.jpg".format(job_id))


def perform_augmentation(dir, width, height, batch_size, **kwargs):
    print("augmentation kwargs :{}".format(kwargs))
    # Define the data generator for data augmentation and preprocessing
    augmentation_kwargs = kwargs.get("augmentation_kwargs", {})
    if augmentation_kwargs == {}:
        assert False, f"augmentation_kwargs is {augmentation_kwargs}"
        
    train_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=augmentation_kwargs.get("rotation_range", 0),
        width_shift_range=augmentation_kwargs.get("width_shift_range", 0),
        height_shift_range=augmentation_kwargs.get("height_shift_range", 0),
        shear_range=augmentation_kwargs.get("shear_range", 0),
        zoom_range=augmentation_kwargs.get("zoom_range", 0),
        horizontal_flip=augmentation_kwargs.get("horizontal_flip", False),
        vertical_flip=False,
    )

    # Load and preprocess the training dataset
    train_dataset = train_data_generator.flow_from_directory(
        directory=dir + "/train/",
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    # Load and preprocess the validation dataset
    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=dir + "/test/",
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    # Load and preprocess the test dataset
    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_dataset = test_data_generator.flow_from_directory(
        directory=dir + "/test/",
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    return train_dataset, validation_dataset, test_dataset


def get_mobilenet_model(
    NUM_CLASSES, lr, momentum, optimizer_name, NUM_FROZEN_LAYERS=None, **kwargs
):
    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
    )

    # Set all layers to non-trainable
    base_model.trainable = False

    x = base_model.output

    # Fine-tuning methods
    if kwargs.get("finetuning_method") == "method_1":
        for layer in base_model.layers:
            if "conv" in layer.name:
                x = Dropout(kwargs.get("dropout"))(x)
    elif kwargs.get("finetuning_method") == "method_2":
        layer_names = [
            "conv_pw_3_relu",
            "conv_pw_5_relu",
            "conv_pw_11_relu",
            "conv_pw_13_relu",
        ]
        for layer in base_model.layers:
            if layer.name in layer_names:
                x = Dropout(kwargs.get("dropout"))(x)

    # Adding classification layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)
    
    # Set top layers to trainable
    for layer in base_model.layers[:NUM_FROZEN_LAYERS]:
        layer.trainable = True
 

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    if optimizer_name.lower() == "sgd":
        optimizer = SGD(lr=lr, momentum=momentum)
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if optimizer_name == "SGD":
        optimizer = SGD(learning_rate=lr, momentum=momentum)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(learning_rate=lr, momentum=momentum)
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad(learning_rate=lr, ema_momentum=momentum)
    elif optimizer_name == "Adadelta":
        optimizer = Adadelta(learning_rate=lr, ema_momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = Adam(learning_rate=lr, ema_momentum=momentum)
    elif optimizer_name == "Adamax":
        optimizer = Adamax(learning_rate=lr, ema_momentum=momentum)
    elif optimizer_name == "Nadam":
        optimizer = Nadam(learning_rate=lr, ema_momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # compile  model
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_best_worst_model(models_folder_path: str):
    model_files = os.listdir(models_folder_path)
    assert len(model_files) >= 2, "YOU NEED AT LEAST 2 SAVED MODELS TO COMPARE !!"

    model_accuracies = {}
    for file_name in model_files:
        if file_name.endswith(".h5"):
            parts = file_name.split("_")
            try:
                accuracy = float(parts[-1].replace(".h5", ""))
                model_accuracies[file_name] = accuracy
            except ValueError:
                print(f"Skipping file: {file_name}")

    sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1])

    worst_model_file, worst_model_acc = sorted_models[0]
    best_model_file, best_model_acc = sorted_models[-1]

    worst_model = tf.keras.models.load_model(
        os.path.join(models_folder_path, worst_model_file)
    )
    best_model = tf.keras.models.load_model(
        os.path.join(models_folder_path, best_model_file)
    )

    # find last conv layer
    def find_last_conv_layer(model):
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D)):
                return layer.name
        return None

    last_conv_layer_name = find_last_conv_layer(best_model)

    return best_model, worst_model, last_conv_layer_name


# helper function for gradcam


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # We pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(
    img, heatmap, class_name, alpha=0.4, subplot_num=None, grid_size=(1, 8)
):
    """
    Displays the Grad CAM heatmap superimposed on the original image.

    Args:
    - img: Original image
    - heatmap: Grad CAM heatmap
    - class_name: The actual class name to be displayed as title
    - alpha: Transparency level for the heatmap overlay
    - subplot_num: The subplot number in the 1x3 grid
    """
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    ax = plt.subplot(grid_size[0], grid_size[1], subplot_num)
    plt.imshow(superimposed_img)
    plt.title(class_name)
    plt.axis("off")


def get_random_images_for_each_class(dataset, num_classes=8):
    """Returns a random image for each class in the dataset."""
    images_per_class = {cls: [] for cls in range(num_classes)}
    for images, labels in dataset.unbatch():
        cls = labels.numpy().astype("uint8")
        images_per_class[cls].append(images.numpy().astype("uint8"))

    random_images = {
        cls: random.choice(images) for cls, images in images_per_class.items()
    }
    return random_images


# helper function to plot feature map
def visualize_conv_layer_feature_maps(
    model, img_path, preprocess_func, target_size=(224, 224), max_features=64
):
    """
    Visualize feature maps for each convolutional layer in the model.

    Args:
    - model: Trained Keras model.
    - img_path: Path to the input image.
    - preprocess_func: Function to preprocess the input image.
    - target_size: Target size for image resizing.
    - max_features: Maximum number of feature maps to display for each layer.
    """

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)

    # Identify Convolutional Layers
    conv_layers = [layer.name for layer in model.layers if "conv" in layer.name]

    for layer_name in conv_layers:
        # Create a model for each conv layer
        layer_model = tf.keras.models.Model(
            inputs=model.inputs, outputs=model.get_layer(layer_name).output
        )

        # Generate feature maps
        feature_maps = layer_model.predict(img_array)

        # Number of features to display
        num_features = min(max_features, feature_maps.shape[-1])

        # Set up the grid for plotting
        size = int(np.ceil(np.sqrt(num_features)))
        fig, axes = plt.subplots(size, size, figsize=(12, 12))

        for i in range(size**2):
            ax = axes[i // size, i % size]
            ax.axis("off")
            if i < num_features:
                # Display feature map
                ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
                ax.set_title(f"Map {i+1}")
        plt.suptitle(f"Layer: {layer_name}")
        plt.show()


def visualize_one_feature_map_per_conv_layer(
    model, img_path, preprocess_func, target_size=(224, 224), n=4, save_path=None
):
    """
    Visualize one feature map for each convolutional layer in the model in a grid.

    Args:
    - model: Trained Keras model.
    - img_path: Path to the input image.
    - preprocess_func: Function to preprocess the input image.
    - target_size: Target size for image resizing.
    - n_cols: Number of columns in the grid.
    - save_path: Path to save the generated figure (e.g., 'output.png').
    """

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)

    # Identify Convolutional Layers
    conv_layers = [layer.name for layer in model.layers if "conv" in layer.name]

    # Determine the number of rows based on the number of columns and layers
    num_layers = len(conv_layers)
    n_cols = 12
    n_rows = int(np.ceil(num_layers / n_cols))

    # Plotting setup
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for i, layer_name in enumerate(conv_layers):
        # Create a model for each conv layer
        layer_model = tf.keras.models.Model(
            inputs=model.inputs, outputs=model.get_layer(layer_name).output
        )

        # Generate feature map
        feature_map = layer_model.predict(img_array)[0, :, :, 0]

        # Calculate the position in the grid
        row = i // n_cols
        col = i % n_cols

        # Display the feature map
        ax = axes[row, col]
        ax.imshow(feature_map, cmap="viridis")
        ax.set_title(f"Layer: {layer_name}")
        ax.axis("off")

    # Remove empty subplots
    for i in range(num_layers, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    # Save the figure if save_path is provided
    plt.savefig("/ghome/group05/week 3/output/featuremap.jpg", bbox_inches="tight")

    plt.show()
