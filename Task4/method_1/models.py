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
from tensorflow.keras.regularizers import l2


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


class Model_2(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_2, self).__init__()
        # Convolutional Block 1
        self.conv1 = Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
        )
        self.pool1 = MaxPooling2D((2, 2))
        self.drop1 = Dropout(0.25)

        # Convolutional Block 2
        self.conv2 = Conv2D(64, (3, 3), activation="relu")
        self.pool2 = MaxPooling2D((2, 2))
        self.drop2 = Dropout(0.3)

        # Convolutional Block 3
        self.conv3 = Conv2D(128, (3, 3), activation="relu")
        self.pool3 = MaxPooling2D((2, 2))
        self.drop3 = Dropout(0.3)

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


# model 3
# class ResidualBlock(Layer):
#     def __init__(self, num_filters, kernel_size):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = Conv2D(
#             num_filters, kernel_size, padding="same", activation="relu"
#         )
#         self.conv2 = Conv2D(num_filters, kernel_size, padding="same")

#     def call(self, input_tensor):
#         x = self.conv1(input_tensor)
#         x = self.conv2(x)
#         x += input_tensor
#         return tf.nn.relu(x)


# class Model_3(tf.keras.Model):
#     def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
#         super(Model_3, self).__init__()
#         self.conv1 = Conv2D(
#             32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
#         )
#         self.pool1 = MaxPooling2D((2, 2))
#         self.res1 = ResidualBlock(32, (3, 3))

#         self.conv2 = Conv2D(64, (3, 3), activation="relu")
#         self.pool2 = MaxPooling2D((2, 2))
#         self.res2 = ResidualBlock(64, (3, 3))

#         self.flatten = Flatten()
#         self.dense1 = Dense(128, activation="relu")
#         self.drop = Dropout(0.5)
#         self.dense2 = Dense(NUM_CLASSES, activation="softmax")

#     def call(self, x):
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.res1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.res2(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.drop(x)
#         return self.dense2(x)


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


class Model_5(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_5, self).__init__()

        # Convolutional Blocks
        self.conv1 = Conv2D(
            16, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
        )
        self.pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch1 = BatchNormalization()

        self.conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")
        self.pool2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch2 = BatchNormalization()

        self.conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool3 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch3 = BatchNormalization()

        self.conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")
        self.pool4 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch4 = BatchNormalization()

        self.conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")
        self.pool5 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch5 = BatchNormalization()

        self.conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")
        self.pool6 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch6 = BatchNormalization()

        # Global Average Pooling and Dense Layer
        self.global_pool = GlobalAveragePooling2D()
        self.fc = Dense(NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.batch4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = self.batch5(x)

        x = self.conv6(x)
        x = self.pool6(x)
        x = self.batch6(x)

        x = self.global_pool(x)
        return self.fc(x)


# model 6 - Diana


class Model_6(tf.keras.Model):
    def __init__(
        self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES, dropout_rate=0.25, l2_reg=0.001
    ):
        super(Model_6, self).__init__()

        # Convolutional Blocks
        self.conv1 = Conv2D(
            16,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
            kernel_regularizer=l2(l2_reg),
        )
        self.pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch1 = BatchNormalization()
        self.drop1 = Dropout(dropout_rate)

        self.conv2 = Conv2D(
            32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(l2_reg)
        )
        self.pool2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch2 = BatchNormalization()
        self.drop2 = Dropout(dropout_rate)

        self.conv3 = Conv2D(
            64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(l2_reg)
        )
        self.pool3 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch3 = BatchNormalization()
        self.drop3 = Dropout(dropout_rate)

        self.conv4 = Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=l2(l2_reg),
        )
        self.pool4 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")
        self.batch4 = BatchNormalization()
        self.drop4 = Dropout(dropout_rate)

        # Global Average Pooling and Dense Layer
        self.global_pool = GlobalAveragePooling2D()
        self.fc = Dense(
            NUM_CLASSES, activation="softmax", kernel_regularizer=l2(l2_reg)
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.batch4(x)
        x = self.drop4(x)

        x = self.global_pool(x)
        return self.fc(x)


# model 7 - Diana


class Model_7(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_7, self).__init__()

        # Convolutional Blocks with additional pooling and batch normalization
        self.conv1 = Conv2D(
            16, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
        )
        self.pool1 = MaxPooling2D((2, 2))
        self.batch1 = BatchNormalization()

        self.conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")
        self.pool2 = MaxPooling2D((2, 2))
        self.batch2 = BatchNormalization()

        self.conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool3 = MaxPooling2D((2, 2))
        self.batch3 = BatchNormalization()

        self.conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")
        self.pool4 = MaxPooling2D((2, 2))
        self.batch4 = BatchNormalization()

        self.conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")
        self.pool5 = MaxPooling2D((2, 2))
        self.batch5 = BatchNormalization()

        self.conv6 = Conv2D(512, (3, 3), activation="relu", padding="same")
        self.pool6 = MaxPooling2D((2, 2))
        self.batch6 = BatchNormalization()

        # Global Average Pooling and Dense Layer
        self.global_pool = GlobalAveragePooling2D()
        self.fc = Dense(NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.batch4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = self.batch5(x)

        x = self.conv6(x)
        x = self.pool6(x)
        x = self.batch6(x)

        x = self.global_pool(x)
        return self.fc(x)


# model 8


class Model_8(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_8, self).__init__()

        self.conv1 = Conv2D(
            filters=128,
            kernel_size=(5, 5),
            padding="valid",
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
        )
        self.act1 = Activation("relu")
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.norm1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="valid",
            kernel_regularizer=l2(0.00005),
        )
        self.act2 = Activation("relu")
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.norm2 = BatchNormalization()

        self.conv3 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="valid",
            kernel_regularizer=l2(0.00005),
        )
        self.act3 = Activation("relu")
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.norm3 = BatchNormalization()

        self.flatten = Flatten()
        self.dense1 = Dense(units=256, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(units=NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.norm3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
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
                ),
                MaxPooling2D(2, 2),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation=tf.nn.relu),
                Dense(NUM_CLASSES, activation=tf.nn.softmax),
            ]
        )

    def call(self, x):
        return self.model(x)


# model 10
# model 11
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
    elif model_name == "MODEL_2":
        model = Model_2(
            IMG_SIZE=IMG_SIZE, IMG_CHANNEL=IMG_CHANNEL, NUM_CLASSES=NUM_CLASSES
        )
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
