import os
from tensorflow.keras.layers import (Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D)
from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.optimizers import Adam


# Convolutional AutoEncoder
class CAE:
    def __init__(self, input_dim=784, latent_dim=64, intermediate_dim=512, name="cae"):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.model, self.encoder, self.decoder = self.build_model()
        self.name = name

    def build_model(self):
        input_img = Input(shape=(self.input_dim,))
        img = Reshape((28, 28, 1))(input_img)

        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(self.intermediate_dim, activation='relu')(x)
        encoded = Dense(self.latent_dim, activation='relu')(x)

        decoder = Sequential([
            Dense(self.intermediate_dim, activation='relu'),
            Dense(7 * 7 * 64, activation='relu'),
            # Upscale to match the shape before the final MaxPooling in the encoder
            Reshape((7, 7, 64)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(1, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(28 * 28, activation='sigmoid'),
        ])

        decoded = decoder(encoded)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        encoder = Model(input_img, encoded)

        return autoencoder, encoder, decoder

    def classifier(self, lr=1e-3):
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        enc = self.encoder(x)
        x = Dense(128, activation='relu')(enc)
        classifier_output = Dense(10, activation='softmax')(x)

        classifier = Model(input_img, classifier_output)
        classifier.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier

    def image_encoder(self):
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        y = self.encoder(x)
        encoder = Model(input_img, y)
        return encoder

    def clone_encoder(self):
        cloned_encoder = clone_model(self.encoder)
        cloned_encoder.set_weights(self.encoder.get_weights())
        cloned_encoder.trainable = False
        return cloned_encoder

    def save(self, path="trained_models"):
        self.encoder.save(os.path.join(path, f"{self.name}_encoder.ckpt.h5"))
        self.decoder.save(os.path.join(path, f"{self.name}_decoder.ckpt.h5"))
        self.model.save(os.path.join(path, f"{self.name}_model.ckpt.h5"))

    @classmethod
    def load(cls, path, model_name="cae"):
        model = cls()
        model.encoder = load_model(os.path.join(path, f"{model_name}_encoder.ckpt.h5"))
        model.decoder = load_model(os.path.join(path, f"{model_name}_decoder.ckpt.h5"))
        model.model = load_model(os.path.join(path, f"{model_name}_model.ckpt.h5"))
        return model
