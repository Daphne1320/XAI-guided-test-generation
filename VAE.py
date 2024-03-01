from tensorflow.python.keras.layers import (Input, Dense, Lambda, Layer, Multiply, Add, Flatten)
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow as tf
import tensorflow.python.keras.backend as K


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

# Variational Autoencoder
class VAE:
    def __init__(self, input_dim=784, latent_dim=2, intermediate_dim=512):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.model, self.encoder, self.decoder = self.build_model()

    def build_model(self):
        x = Input(shape=(self.input_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(x)

        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=1.0,
                                           shape=(K.shape(x)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        decoder = Sequential([
            Dense(self.intermediate_dim, input_dim=self.latent_dim, activation='relu'),
            Dense(self.input_dim, activation='sigmoid')
        ], name="vae_decoder")

        x_pred = decoder(z)

        # end-to-end autoencoder
        vae = Model(inputs=[x, eps], outputs=x_pred)
        vae.compile(optimizer='rmsprop', loss=nll)

        # Compile with binary cross entropy loss (the actual loss is computed in AELossLayer)
        encoder = Model(x, z_mu)

        return vae, encoder, decoder

    def classifier(self, lr=1e-3):
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        enc = self.encoder(x)
        x = Dense(128, activation='relu')(enc)
        classifier_output = Dense(10, activation='softmax')(x)

        classifier = Model(input_img, classifier_output)
        classifier.compile(optimizer=adam_v2.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier

    def image_encoder(self):
        return self.encoder
