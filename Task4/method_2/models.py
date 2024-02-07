import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    ReLU,
    Dense,
    GlobalAveragePooling2D,
    Add,
    SeparableConv2D,
)
from tensorflow.keras.models import Model
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1


# model 1
import tensorflow as tf
from tensorflow.keras import layers


class Model_1(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_1, self).__init__()
        # Convolutional Block 1
        self.conv1 = Conv2D(
            16, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
        )
        self.pool1 = MaxPooling2D((2, 2))
        self.drop1 = Dropout(0.25)

        # Convolutional Block 2
        self.conv2 = Conv2D(32, (3, 3), activation="relu")
        self.pool2 = MaxPooling2D((2, 2))
        self.drop2 = Dropout(0.25)

        # Convolutional Block 3
        self.conv3 = Conv2D(64, (3, 3), activation="relu")
        self.pool3 = MaxPooling2D((2, 2))
        self.drop3 = Dropout(0.25)

        # Flatten layer
        self.flatten = Flatten()

        # Dense Layer
        self.dense1 = Dense(128, activation="relu")
        self.drop4 = Dropout(0.5)

        # Output Layer
        self.dense2 = Dense(NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop4(x)
        return self.dense2(x)


# model 4
class Model_4(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_4, self).__init__()
        self.conv1 = SeparableConv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
        )
        self.pool1 = MaxPooling2D((2, 2))
        self.drop1 = Dropout(0.25)

        self.conv2 = SeparableConv2D(64, (3, 3), activation="relu")
        self.pool2 = MaxPooling2D((2, 2))
        self.drop2 = Dropout(0.25)

        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.drop3 = Dropout(0.5)
        self.dense2 = Dense(NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop3(x)
        return self.dense2(x)


# model 5 - Diana

# model 8


class Model_8(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_8, self).__init__()

        # Convolutional Layer 1
        self.conv1 = Conv2D(
            filters=128,
            kernel_size=(5, 5),
            padding="valid",
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
            kernel_regularizer=l2(0.0001),
        )
        self.norm1 = BatchNormalization()
        self.act1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.3)  # Adjusted dropout rate

        # Convolutional Layer 2
        self.conv2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="valid",
            kernel_regularizer=l2(0.0001),
        )
        self.norm2 = BatchNormalization()
        self.act2 = Activation("relu")
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = Dropout(0.4)  # Adjusted dropout rate

        # Convolutional Layer 3
        self.conv3 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="valid",
            kernel_regularizer=l2(0.0001),
        )
        self.norm3 = BatchNormalization()
        self.act3 = Activation("relu")
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.dropout3 = Dropout(0.4)  # Adjusted dropout rate

        # Dense Layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=256, activation="relu")
        self.dropout4 = Dropout(0.5)
        self.dense2 = Dense(units=NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x)
        x = self.dense2(x)

        return x


# model 9


class Model_9(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_9, self).__init__()
        self.model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
                    kernel_regularizer=l2(0.0001),
                ),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.3),
                Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.0001)),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.4),
                Flatten(),
                Dense(128, activation=tf.nn.relu),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation=tf.nn.softmax),
            ]
        )

    def call(self, x):
        return self.model(x)


# model 10


class Model_10(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_10, self).__init__()

        self.model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.2),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.3),
                Conv2D(128, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.4),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

    def call(self, x):
        return self.model(x)


# model 11
class Model_11(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_11, self).__init__()
        self.model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

    def call(self, x):
        return self.model(x)


# model 12


def get_model(
    IMG_SIZE: int,
    IMG_CHANNEL: int,
    NUM_CLASSES: int,
    model_name: str,
    optimizer_name: str,
    lr: float = None,
    momentum: float = None,
):
    if optimizer_name == "SGD":
        optimizer = SGD()
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop()
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad()
    elif optimizer_name == "Adadelta":
        optimizer = Adadelta()
    elif optimizer_name == "Adam":
        optimizer = Adam()
    elif optimizer_name == "Adamax":
        optimizer = Adamax()
    elif optimizer_name == "Nadam":
        optimizer = Nadam()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # compile  model
    if model_name == "MODEL_1":
        model = Model_1(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )
    # elif model_name == "MODEL_2":
    #     model = Model_2(
    #         IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
    #     )
    # elif model_name == "MODEL_3":
    #     model = Model_3(
    #         IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
    #     )
    elif model_name == "MODEL_4":
        model = Model_4(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )
    # elif model_name == "MODEL_5":
    #     model = Model_5(
    #         IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
    #     )
    # elif model_name == "MODEL_6":
    #     model = Model_5(
    #         IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
    #     )
    # elif model_name == "MODEL_7":
    #     model = Model_5(
    #         IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
    #     )
    elif model_name == "MODEL_8":
        model = Model_8(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )
    elif model_name == "MODEL_9":
        model = Model_9(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )
    elif model_name == "MODEL_10":
        model = Model_10(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )
    elif model_name == "MODEL_11":
        model = Model_11(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )

    model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # total number of parameters
    total_params = model.count_params()
    # print(model.summary())
    return model, total_params
