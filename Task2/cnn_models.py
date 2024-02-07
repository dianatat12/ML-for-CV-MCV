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
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import (
    MobileNetV2,
    VGG16,
    ResNet50,
    EfficientNetB0,
    DenseNet121,
)
from tensorflow.keras.regularizers import l2

from vit_keras import vit, utils
import warnings

warnings.filterwarnings("ignore")


# pretrained CNN models
def preatrined_mobilenet_V2(
    IMG_SIZE: int, IMG_CHANNEL: int, NUM_CLASSES: int, ONLY_FEATURE: bool
):
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if ONLY_FEATURE:
        # if only_feature is True, return the model up to the global average pooling layer
        model = Model(inputs=base_model.input, outputs=x)
    else:
        # if ONLY_FEATURE is False, add the classification layers
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    return model


def pretrained_vgg16(
    IMG_SIZE: int, IMG_CHANNEL: int, NUM_CLASSES: int, ONLY_FEATURE: bool
):
    # load the pre-trained VGG16 model from Keras applications with ImageNet weights
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
    )

    # set the pre-trained layers to be non-trainable
    base_model.trainable = False

    # obtain the output tensor from the last layer of the base model
    x = base_model.output

    # apply global average pooling to reduce spatial dimensions and obtain a fixed-size feature vector
    x = GlobalAveragePooling2D()(x)

    if ONLY_FEATURE:
        # return the model up to the global average pooling layer
        model = Model(inputs=base_model.input, outputs=x)
    else:
        # add additional classification layers
        x = Dense(1024, activation="relu")(x)  # add a dense layer with ReLU activation
        predictions = Dense(NUM_CLASSES, activation="softmax")(
            x
        )  # add the final dense layer for classification
        model = Model(inputs=base_model.input, outputs=predictions)

    # Return the constructed VGG16-based model
    return model


def pretrained_resnet50(
    IMG_SIZE: int, IMG_CHANNEL: int, NUM_CLASSES: int, ONLY_FEATURE: bool
):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if ONLY_FEATURE:
        model = Model(inputs=base_model.input, outputs=x)

    else:
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    return model


def pretrained_efficientnet(
    IMG_SIZE: int, IMG_CHANNEL: int, NUM_CLASSES: int, ONLY_FEATURE: bool
):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if ONLY_FEATURE:
        model = Model(inputs=base_model.input, outputs=x)
    else:
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    return model


def pretrained_vit(
    IMG_SIZE: int, IMG_CHANNEL: int, NUM_CLASSES: int, ONLY_FEATURE: bool
):
    model = vit.vit_b16(
        image_size=IMG_SIZE,
        patch_size=16,
        num_classes=NUM_CLASSES,
        channels=IMG_CHANNEL,
        classifier_activation="softmax",
    )

    model.trainable = False

    if ONLY_FEATURE:
        model = Model(inputs=model.input, outputs=model.layers[-2].output)

    else:
        x = Dense(1024, activation="relu")(model.output)
        predictions = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=model.input, outputs=predictions)

    return model


def pretrained_densenet(
    IMG_SIZE: int, IMG_CHANNEL: int, NUM_CLASSES: int, ONLY_FEATURE: bool
):
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if ONLY_FEATURE:
        model = Model(inputs=base_model.input, outputs=x)

    else:
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    return model


# CNN from scratch
# Test accuracy: 0.8525402545928955, Test loss: 0.4695713520050049
class CNN_Patch_Model(tf.keras.Model):
    def __init__(
        self,
        IMG_SIZE: int,
        IMG_CHANNEL: int,
        NUM_CLASSES: int,
        ONLY_FEATURE: bool,
        PATCH_SIZE: int,
    ):
        super(CNN_Patch_Model, self).__init__()
        self.ONLY_FEATURE = ONLY_FEATURE
        self.conv1 = Conv2D(
            filters=64,
            kernel_size=PATCH_SIZE,
            strides=PATCH_SIZE,
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
        )
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.25)

        self.conv2 = Conv2D(128, (3, 3), activation="relu")
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = Dropout(0.25)

        self.conv3 = Conv2D(256, (3, 3), activation="relu")
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2))
        self.dropout3 = Dropout(0.25)

        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dropout4 = Dropout(0.2)
        self.dense2 = Dense(NUM_CLASSES, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        if self.ONLY_FEATURE:
            return x
        x = self.dense1(x)
        x = self.dropout4(x)
        return self.dense2(x)


# TODO: Model with residual connections


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, increase_filter=False):
        super(ResidualBlock, self).__init__()
        self.increase_filter = increase_filter
        self.conv1 = Conv2D(
            num_filters,
            kernel_size,
            padding="same",
            strides=(2 if increase_filter else 1),
            kernel_regularizer=l2(1e-4),
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            num_filters, kernel_size, padding="same", kernel_regularizer=l2(1e-4)
        )
        self.bn2 = BatchNormalization()

        if self.increase_filter:
            self.residual_conv = Conv2D(
                num_filters,
                (1, 1),
                strides=2,
                padding="same",
                kernel_regularizer=l2(1e-4),
            )

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = Activation("relu")(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.increase_filter:
            input_tensor = self.residual_conv(input_tensor)

        x += input_tensor
        x = Activation("relu")(x)
        return x


class CNN_Residual_Model(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES, ONLY_FEATURE):
        super(CNN_Residual_Model, self).__init__()
        self.conv1 = Conv2D(
            32,
            (3, 3),
            padding="same",
            input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL),
            kernel_regularizer=l2(1e-4),
        )
        self.bn1 = BatchNormalization()
        # self.res_block1 = ResidualBlock(32, (3, 3))
        self.res_block2 = ResidualBlock(64, (3, 3), increase_filter=True)
        self.res_block3 = ResidualBlock(128, (3, 3), increase_filter=True)
        # self.res_block4 = ResidualBlock(256, (3, 3), increase_filter=True)
        # self.res_block5 = ResidualBlock(512, (3, 3), increase_filter=True)
        self.pool1 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(NUM_CLASSES, activation="softmax")

        self.only_feature = ONLY_FEATURE

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = Activation("relu")(x)

        # x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.res_block3(x, training=training)
        # x = self.res_block4(x, training=training)
        # x = self.res_block5(x, training=training)
        x = self.pool1(x)
        x = self.flatten(x)

        if not self.only_feature:
            x = self.dense1(x)
            x = self.dropout(x, training=training)
            x = self.dense2(x)

        return x


def get_CNN_model(
    MODEL_NAME: str,
    IMG_SIZE: int,
    IMG_CHANNEL: int,
    NUM_CLASSES: int,
    ONLY_FEATURE: bool = False,
    NUM_PATCH: int = None,
):
    if MODEL_NAME == "pretrained_mobilenet_V2":
        return preatrined_mobilenet_V2(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )

    elif MODEL_NAME == "pretrained_vgg16":
        return pretrained_vgg16(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )

    elif MODEL_NAME == "pretrained_resnet50":
        return pretrained_resnet50(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )

    elif MODEL_NAME == "pretrained_efficientnet":
        return pretrained_efficientnet(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )

    elif MODEL_NAME == "pretrained_vit":
        return pretrained_efficientnet(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )

    elif MODEL_NAME == "pretrained_densenet":
        return pretrained_efficientnet(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )
    elif MODEL_NAME == "scratch_CNN_Patch_Model":
        return CNN_Patch_Model(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
            NUM_PATCH=NUM_PATCH,
        )
    elif MODEL_NAME == "scratch_CNN_Residual_Model":
        return CNN_Residual_Model(
            IMG_SIZE=IMG_SIZE,
            IMG_CHANNEL=IMG_CHANNEL,
            NUM_CLASSES=NUM_CLASSES,
            ONLY_FEATURE=ONLY_FEATURE,
        )


if __name__ == "__main__":
    import numpy as np

    print("_" * 100)
    print("TEST")
    IMG_SIZE = 256
    IMG_CHANNEL = 3
    NUM_CLASSES = 8
    ONLY_FEATURE = True
    model = CNN_Residual_Model(
        IMG_SIZE=IMG_SIZE,
        IMG_CHANNEL=IMG_CHANNEL,
        NUM_CLASSES=NUM_CLASSES,
        ONLY_FEATURE=ONLY_FEATURE,
    )
    input_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
    model.build((None, *input_shape))

    batch_size = 1
    random_input = np.random.random(
        (batch_size, IMG_SIZE, IMG_SIZE, IMG_CHANNEL)
    ).astype("float32")

    output = model(random_input)

    print("Model's output shape:", output.shape)
