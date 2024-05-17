import numpy as np
import tensorflow as tf
from tqdm import tqdm
import datetime
import os
from tensorflow.keras.models import load_model


# Conditional Variational Autoencoder
class CVAE(tf.keras.Model):

    def __init__(self, latent_dim=64, inter_dim=512):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.inter_dim = inter_dim

        input_y = tf.keras.layers.Input(shape=(10,))
        img = tf.keras.layers.Input((28, 28, 1))

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(img)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        concat_layer = tf.keras.layers.concatenate([x, input_y], axis=1)
        x = tf.keras.layers.Dense(self.inter_dim, activation='relu')(concat_layer)
        encoded = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(x)
        self.encoder = tf.keras.Model(inputs=[img, input_y], outputs=[encoded])  # mu and log_var

        embedd = tf.keras.layers.Input((self.latent_dim,))
        merged_input = tf.keras.layers.Concatenate()([embedd, input_y])

        x = tf.keras.layers.Dense(self.inter_dim)(merged_input)
        x = tf.keras.layers.Dense(7 * 7 * 32, activation='relu')(x)
        x = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu')(x)

        x = tf.keras.layers.Conv2DTranspose(filters=32,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same')(x)

        self.decoder = tf.keras.Model(inputs=[embedd, input_y], outputs=[x])

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def fit(self, train_gen, test_gen, epochs=12):
        for epoch in tqdm(range(epochs)):
            for x_batch, y_batch in train_gen:
                self.train_step(x_batch, y_batch)
                # with self.train_summary_writer.as_default():
                #    tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

            for x_batch, y_batch in test_gen:
                self.test_step(x_batch, y_batch)
                # with self.test_summary_writer.as_default():
                #    tf.summary.scalar('loss', self.test_loss.result(), step=epoch)

            template = 'Epoch {}, Loss: {}, Test Loss: {}'
            print(template.format(epoch + 1, self.train_loss.result(), self.test_loss.result()))
            # Reset metrics every epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            # generate_and_save_images(model, epoch, test_sample_x, test_sample_y)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)

    @tf.function
    def test_step(self, x_test, y_test):
        self.test_loss(self.compute_loss(x_test, y_test))

    def compute_loss(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, y)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    # Sample z ~ Q(z|X,y)
    @tf.function
    def sample(self, eps, y):
        return self.decode(eps, y, apply_sigmoid=True)

    def encode(self, x, y):
        mean, logvar = tf.split(self.encoder([x, y]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, y, apply_sigmoid=False):
        logits = self.decoder([z, y])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def save(self, path):
        self.encoder.save(os.path.join(path, "CVAE_encoder.ckpt.h5"))
        self.decoder.save(os.path.join(path, "CVAE_decoder.ckpt.h5"))

    def load(self, path):
        self.encoder = load_model(os.path.join(path, "CVAE_encoder.ckpt.h5"))
        self.decoder = load_model(os.path.join(path, "CVAE_decoder.ckpt.h5"))
