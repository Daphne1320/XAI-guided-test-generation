import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dense
from tensorflow.keras.activations import softmax


class Lava():
    def __init__(self, classifier, encoder, decoder, adversarial_method="gradient", verbose=False):
        self.classifier = classifier
        self.encoder = encoder
        self.decoder = decoder
        self.adversarial_method = adversarial_method
        self.verbose = verbose
        self.is_cvae = isinstance(self.encoder.input_shape, list) and len(self.encoder.input_shape) > 1
        self.xai = self.stack_to_xai()

    def stack_to_xai(self):
        if self.is_cvae:
            x_input = Input(shape=self.decoder.input_shape[0])
            x_input_label = Input(shape=self.decoder.input_shape[1])
            x_dec = self.decoder([x_input, x_input_label])
        else:
            x_input = Input(shape=self.decoder.input_shape)
            x_dec = self.decoder(x_input)
        x_img = Reshape((28, 28, 1))(x_dec)
        y = self.classifier(x_img)
        return Model(inputs=[x_input, x_input_label] if self.is_cvae else x_input, outputs=y)

    def generate_adversarial_image(self, image, image_label, step=0.5):
        x = np.reshape([image], (1, 784,))
        if self.is_cvae:
            x_label = np.array([image_label])
            h = self.encoder.predict([x, x_label])[0][:self.decoder.input_shape[0][-1]]
        else:
            h = self.encoder.predict(x)[0]
        return self.generate_image_via_h(h, image_label, step=step)

    def generate_image_via_h(self, h, image_label, step=0.5):
        x = np.array([h])
        y = np.array([image_label])
        h_delta = self.adversarial(x, y, self.xai)
        h_lava = self.get_h_lava_via_one_step(h, h_delta, step=step)
        h_lava_input = np.array([h_lava])
        if self.is_cvae:
            image_adv = tf.sigmoid(self.decoder.predict([h_lava_input, y])[0]).numpy().reshape((28, 28, 1))
            if np.any(np.isnan(image_adv)):
                raise ValueError(h_lava_input, y)
        else:
            image_adv = self.decoder.predict(h_lava_input)[0].reshape((28, 28, 1))
        return image_adv, h_lava

    @staticmethod
    def get_h_lava_via_one_step(h, delta_h, step=0.5):
        assert h.shape == delta_h.shape
        h_lava = np.copy(h)
        h_lava += step * delta_h
        return np.reshape(h_lava, h.shape)

    @staticmethod
    def get_h_lava_via_one_grid_step(h, delta_h, step=0.5):
        max_grad_index = np.argmax(np.abs(delta_h))
        max_direction = np.sign(delta_h[max_grad_index])

        delta_h = np.zeros(delta_h.shape)
        delta_h[max_grad_index] = max_direction
        return Lava.get_h_lava_via_one_step(h, delta_h, step=step)

    def adversarial(self, x, y, model, before_softmax=True):
        if before_softmax:
            model = self.remove_last_softmax_activation(model)

        input_data = tf.convert_to_tensor(x, dtype=tf.float32)
        true_labels = tf.convert_to_tensor(y, dtype=tf.float32)

        if self.adversarial_method == "forward_gradient":
            # print(input_data.shape)
            delta = self.forward_gradient_of_input(model, input_data, true_labels)
        elif self.adversarial_method == "bp_gradient":
            delta = self.gradient_of_input(model, input_data, true_labels)
        else:  # salience
            delta = self.saliency_of_input(model, input_data, true_labels)
        return np.squeeze(delta)

    @staticmethod
    def gradient_of_input(model, input_data, output_data):
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            if isinstance(model.input_shape, list):
                predictions = model([input_data, output_data], training=False)
            else:
                predictions = model(input_data, training=False)
            loss = tf.keras.losses.categorical_crossentropy(output_data, predictions)
        return tape.gradient(loss, input_data).numpy()

    @staticmethod
    def saliency_of_input(model, input_data, output_data):
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            if isinstance(model.input_shape, list):
                predictions = model([input_data, output_data], training=False)
            else:
                predictions = model(input_data, training=False)
            target = tf.reduce_max(predictions, axis=1)
        return -tape.gradient(target, input_data).numpy()

    @staticmethod
    def forward_gradient_of_input(model, input_data, output_data):
        delta = 1e-5
        g = np.zeros(input_data.shape[-1])
        for i in range(len(g)):
            x_delta = np.zeros(input_data.shape[-1])
            x_delta[i] = 1  # [0, 0, 1, 0, 0, 0,0]
            input_data_2 = input_data + x_delta * delta
            if isinstance(model.input_shape, list):
                predictions = model([input_data, output_data], training=False)
                predictions_2 = model([input_data_2, output_data], training=False)
            else:
                predictions = model(input_data, training=False)
                predictions_2 = model(input_data_2, training=False)

            loss = tf.keras.losses.categorical_crossentropy(output_data, predictions)
            loss_2 = tf.keras.losses.categorical_crossentropy(output_data, predictions_2)
            loss_delta = loss_2 - loss
            g[i] = loss_delta / delta
        return g

    @staticmethod
    def remove_last_softmax_activation(model):
        if isinstance(model.layers[-1], Dense) and getattr(model.layers[-1], 'activation') == softmax:
            # Modify the last layer to have a linear activation
            model_clone = tf.keras.models.clone_model(model)
            model_clone.set_weights(model.get_weights())
            model_clone.layers[-1].activation = tf.keras.activations.linear
            model = Model(inputs=model_clone.inputs, outputs=model_clone.layers[-1].output)
        return model


class LavaMultiSteps(Lava):
    def generate_adversarial_image(self, image, image_label, alpha=0.01, num_iterations=100):
        x = np.reshape([image], (1, 784,))
        if self.is_cvae:
            x_label = np.array([image_label])
            h = self.encoder.predict([x, x_label])[0][:self.decoder.input_shape[0][-1]]
        else:
            h = self.encoder.predict(x)[0]

        label_true = np.argmax(image_label)
        image_adv = image
        h_lava = h

        for iteration in range(num_iterations):
            # print(f"{self.encoder}, h outside: {h.shape}")
            image_adv, h_lava = self.generate_image_via_h(h, image_label, step=alpha)

            if self.predict_label(self.classifier, image_adv) != label_true:
                image_adv_org = image_adv
                h_lava_org = h_lava

                inner_alpha = alpha / num_iterations
                image_adv, h_lava, inner_iteration = self.iterative_gradient_descent(h, image_label, label_true,
                                                                                     num_iterations, alpha=inner_alpha)
                if h_lava is None:
                    image_adv = image_adv_org
                    h_lava = h_lava_org
                else:
                    if self.verbose:
                        print(f"Misclassification achieved at iteration {iteration}, inner iteration {inner_iteration}")
                    return image_adv, h_lava
            h = h_lava

        return image_adv, h_lava

    def iterative_gradient_descent(self, h, image_label, label_true, num_iterations, alpha):
        for iteration in range(num_iterations):
            image_adv, h_lava = self.generate_image_via_h(h, image_label, step=alpha)
            if self.predict_label(self.classifier, image_adv) != label_true:
                return image_adv, h_lava, iteration
            h = h_lava
        return None, None, None

    @staticmethod
    def predict_label(cnn, image):
        return np.argmax(cnn.predict(np.array([image]))[0])
