import tensorflow as tf

import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

# local imports
from utils import *
from models import *

print("_" * 100)
if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
print("_" * 100)

# DIRECTORIES
DATASET_DIR = "/ghome/mcv/datasets/C3/MIT_small_train_1"
CHECKPOINT_DIR = "./saved_models"

# DEFAULT PARAMETERS
IMG_SIZE = 224
IMG_CHANNEL = 3
NUM_CLASSES = 8
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 100


ES_MONITOR = "val_accuracy"
ES_MODE = "max"
ES_MIN_DELTA = 0.005
ES_PATIENCE = 10
ES_RESTORE_BEST = True

LR_MONITOR = "val_accuracy"
LR_MODE = "max"
LR_MIN_DELTA = 0.01
LR_PATIENCE = 5
LR_FACTOR = 0.25
LR_MIN_LR = 0.00000001


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


# available_learning_rate = [0.0001, 0.001, 0.01, 0.1]
# available_momentum = [0.0, 0.2, 0.4, 0.6, 0.8]


rotation_range = 30
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.5
zoom_range = 0.3
horizontal_flip = True

augmentation_kwargs = {
    "rotation_range": rotation_range,
    "width_shift_range": width_shift_range,
    "height_shift_range": height_shift_range,
    "shear_range": shear_range,
    "zoom_range": zoom_range,
    "horizontal_flip": horizontal_flip,
}


available_optimizers = [
    "SGD",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adam",
    "Adamax",
    "Nadam",
]

available_models = [
    "MODEL_1",
    # "MODEL_2",
    # "MODEL_3",
    "MODEL_4",
    # "MODEL_5",
    # "MODEL_6",
    # "MODEL_7",
    "MODEL_8",
    "MODEL_9",
]


available_image_size = [32, 64, 128, 224]

TRAIN_DIR = "/ghome/group05/mcv/datasets/C3/MIT_small_train_3/train"
TEST_DIR = "/ghome/group05/mcv/datasets/C3/MIT_small_train_3/test"
VALIDATION_DIR = "/ghome/group05/mcv/datasets/C3/MIT_small_train_2/test"


def objective(trail):
    # selected_lr = trail.suggest_categorical("learning_rate", available_learning_rate)
    # slected_momentum = trail.suggest_categorical("momentum", available_momentum)
    selected_optimizer_name = trail.suggest_categorical(
        "optimizer", available_optimizers
    )
    selected_model_name = trail.suggest_categorical("cnn_model", available_models)

    run = wandb.init(
        project="week4_method1",
    )
    IMG_SIZE = trail.suggest_categorical("image_size", available_image_size)

    train_dataset, validation_dataset, test_dataset = prepare_dataset(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        validation_dir=VALIDATION_DIR,
        width=IMG_SIZE,
        height=IMG_SIZE,
        batch_size=BATCH_SIZE,
        augmentation_kwargs=augmentation_kwargs,
    )

    model, total_param = get_model(
        IMG_SIZE=IMG_SIZE,
        NUM_CLASSES=NUM_CLASSES,
        IMG_CHANNEL=IMG_CHANNEL,
        model_name=selected_model_name,
        optimizer_name=selected_optimizer_name,
        # lr=selected_lr,
        # momentum=slected_momentum,
    )
    history = model.fit(
        train_dataset,
        epochs=NUMBER_OF_EPOCHS,
        validation_data=validation_dataset,
        callbacks=[early_stopper, reduce_lr],
    )

    # save the model
    model.save_weights(f"/ghome/group05/week4_method1/models/{run.name}.h5")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    plot_training_history(history, run.name, test_loss, test_accuracy)

    performance = performance_metric(
        test_acc=test_accuracy, total_parameter=total_param
    )
    wandb.log(
        {
            "val_accuracy": history.history["val_accuracy"][0],
            "train_accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "train_loss": history.history["loss"][0],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "performance_metric": performance,
            "total_param": total_param,
        }
    )

    return test_accuracy


if __name__ == "__main__":
    wandb_kwargs = {"project": "week4_method1"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(
        objective,
        n_trials=20,
        timeout=600000,
        callbacks=[wandbc],  # weight and bias connection
    )
