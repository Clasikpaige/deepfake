import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Model

DATA_DIR = "datasets/aligned_faces/"
MODEL_DIR = "models/deepfake_model/"

def build_model():
    input_img = Input(shape=(256, 256, 3))

    # Encoder
    x = Conv2D(64, (5, 5), padding="same")(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (5, 5), padding="same", strides=(2, 2))(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Bottleneck
    bottleneck = Flatten()(x)
    dense = Dense(1024, activation="relu")(bottleneck)

    # Decoder
    x = Dense(128 * 128 * 128, activation="relu")(dense)
    x = tf.reshape(x, (-1, 128, 128, 128))
    x = UpSampling2D((2, 2))(x)
    output_img = Conv2D(3, (5, 5), padding="same", activation="sigmoid")(x)

    return Model(input_img, output_img)

def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = build_model()

    # Compile Model
    model.compile(optimizer="adam", loss="mse")

    # Load training images
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=(256, 256), batch_size=32
    )

    # Train the model
    model.fit(train_dataset, epochs=50)
    model.save(os.path.join(MODEL_DIR, "deepfake_model.h5"))
    print("Model training complete and saved!")

if __name__ == "__main__":
    train_model()
