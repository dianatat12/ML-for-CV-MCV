import os

# job_id = os.environ["SLURM_JOB_ID"]

# os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD

from utils import *

# import keras

import tensorflow as tf

import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback


print("_" * 100)
if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print("_" * 100)

DATASET_DIR = "/ghome/mcv/datasets/C3/MIT_small_train_1"
checkpoint_filepath = "./saved_models"

IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 8
BATCH_SIZE = 32
NUMBER_OF_EPOCHS = 20
FROZEN_LAYERS = 40

ES_MONITOR = "val_accuracy"
ES_MODE = "max"
ES_MIN_DELTA = 0.005
ES_PATIENCE = 10
ES_RESTORE_BEST = True

LR_MONITOR = "val_accuracy"
LR_MODE = "max"
LR_MIN_DELTA = 0.01
LR_PATIENCE = 3
LR_FACTOR = 0.25
LR_MIN_LR = 0.00000001


reduce_lr_kwargs = {
    "monitor": LR_MONITOR,
    "mode": LR_MODE,
    "min_delta": LR_MIN_DELTA,
    "patience": LR_PATIENCE,
    "factor": LR_FACTOR,
    "min_lr": LR_MIN_LR,
}

early_stopping_kwargs = {
    "monitor": ES_MONITOR,
    "mode": ES_MODE,
    "min_delta": ES_MIN_DELTA,
    "patience": ES_PATIENCE,
    "restore_best_weights": ES_RESTORE_BEST,
}

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=LR_MONITOR,
    mode=LR_MODE,
    min_delta=LR_MIN_DELTA,
    patience=LR_PATIENCE,
    factor=LR_FACTOR,
    min_lr=LR_MIN_LR,
    cooldown=0,
    verbose=1,
)

early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor=ES_MONITOR,
    mode=ES_MODE,
    min_delta=ES_MIN_DELTA,
    patience=ES_PATIENCE,
    restore_best_weights=ES_RESTORE_BEST,
    verbose=1,
)


available_batch_sizes = [i for i in range(8, 8 * 10, 8)]
available_epochs = [10, 50, 100]
available_optimizers = [
    "SGD",
    # "RMSprop",
    # "Adagrad",
    # "Adadelta",
    "Adam",
    # "Adamax",
    # "Nadam",
]
available_learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
available_momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

avaialble_finetuning_method = ["method_1", "method_2", "method_3"]

available_dropout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
available_num_frozen_layers = [i for i in range(5, 29, 5)]
available_num_frozen_layers.append(None)


def objective(trail):
    NUMBER_OF_EPOCHS = trail.suggest_categorical("NUMBER_OF_EPOCHS", available_epochs)
    BATCH_SIZE = trail.suggest_categorical("BATCH_SIZE", available_batch_sizes)
    NUMBER_OF_EPOCHS = 5  # hardcoding for now
    NUM_FROZEN_LAYERS = trail.suggest_categorical(
        "NUM_FROZEN_LAYERS", available_num_frozen_layers
    )

    selected_finetuning_method = trail.suggest_categorical(
        "finetuning_method", avaialble_finetuning_method
    )

    # augmenation
    rotation_range = trail.suggest_categorical(
        "rotation_range", [i for i in range(0, 60, 10)]
    )
    width_shift_range = trail.suggest_categorical(
        "width_shift_range", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    height_shift_range = trail.suggest_categorical(
        "height_shift_range", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    shear_range = trail.suggest_categorical(
        "shear_range", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    zoom_range = trail.suggest_categorical("zoom_range", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    horizontal_flip = trail.suggest_categorical("horizontal_flip", [True, False])

    lr = trail.suggest_categorical("learning_rate", available_learning_rate)
    momentum = trail.suggest_categorical("momentum", available_momentum)
    optimizer_name = trail.suggest_categorical("optimizer", available_optimizers)

    selected_dropout = trail.suggest_categorical("dropout", available_dropout)

    augmentation_kwargs = {
        "rotation_range": rotation_range,
        "width_shift_range": width_shift_range,
        "height_shift_range": height_shift_range,
        "shear_range": shear_range,
        "zoom_range": zoom_range,
        "horizontal_flip": horizontal_flip,
    }

    run = wandb.init(
        project="week3_mobilenet",
        # config={
        #     "layers": "",  # TODO: Add layer types to config
        #     "dataset": DATASET_DIR,
        #     "epochs": NUMBER_OF_EPOCHS,
        #     "batch_size": BATCH_SIZE,
        #     # "early_stopping": early_stopping_kwargs,
        #     "augmentation": augmentation_kwargs,
        #     # "reduce_lr_kwargs": reduce_lr_kwargs,
        # },
    )
    print("#" * 100)
    print(run.name)
    print("-" * 20)
    print(f"NUMBER_OF_EPOCHS: {NUMBER_OF_EPOCHS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_FROZEN_LAYERS: {NUM_FROZEN_LAYERS}")
    print(f"rotation_range: {rotation_range}")
    print(f"width_shift_range: {width_shift_range}")
    print(f"height_shift_range: {height_shift_range}")
    print(f"shear_range: {shear_range}")
    print(f"zoom_range: {zoom_range}")
    print(f"horizontal_flip: {horizontal_flip}")
    print(f"learning_rate: {lr}")
    print(f"momentum: {momentum}")
    print(f"optimizer: {optimizer_name}")
    print("Augmentation parameters:", augmentation_kwargs)

    print("#" * 100)
    train_dataset, validation_dataset, test_dataset = perform_augmentation(
        DATASET_DIR,
        IMG_WIDTH,
        IMG_HEIGHT,
        BATCH_SIZE,
        augmentation_kwargs=augmentation_kwargs,
    )
    model = get_mobilenet_model(
        NUM_CLASSES=NUM_CLASSES,
        lr=lr,
        momentum=momentum,
        optimizer_name=optimizer_name,
        NUM_FROZEN_LAYERS=NUM_FROZEN_LAYERS,
        finetuning_method=selected_finetuning_method,
        dropout=selected_dropout,
    )
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset,
        callbacks=[early_stopper, reduce_lr],
    )
    test_loss, test_accuracy = model.evaluate(test_dataset)

    wandb.log(
        {
            "val_accuracy": history.history["val_accuracy"][0],
            "train_accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "train_loss": history.history["loss"][0],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
    )

    # save model
    model_file_name = os.path.join(
        checkpoint_filepath, f"{run.name}_test_accuracy_{test_accuracy}.h5"
    )
    model.save(model_file_name)

    plot_hist(history, run.name)
    return test_accuracy


if __name__ == "__main__":
    wandb_kwargs = {"project": "week3_mobilenet"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(
        objective,
        n_trials=1000,
        timeout=600000,
        callbacks=[wandbc],  # weight and bias connection
    )
