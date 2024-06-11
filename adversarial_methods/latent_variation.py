import numpy as np
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model

from xai import gradient_of_x


class Lava():

    def __init__(self, model, encoder, decoder):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.xai = self.stack_to_xai()

    def generate_adversarial_image(self, image, image_label, step=0.5):
        """

        Args:
            image: np.array
            image_label: onehot vector
            step:

        Returns:

        """

        x = np.array([image])
        h = self.encoder.predict(x)[0]
        return self.generate_image_via_h(h, image_label, step=step)

    def generate_adversarial_image_multi_steps(self, image, image_label, alpha=0.01, num_iterations=100):
        x = np.array([image])
        h = self.encoder.predict(x)[0]
        label_true = np.argmax(image_label)
        image_adv = image
        h_lava = h

        for iteration in range(num_iterations):
            h_lava, image_adv = self.generate_image_via_h(h, image_label, step=alpha)

            if self.predict_label(self.model, image_adv) != label_true:
                inner_alpha = alpha / num_iterations
                h_lava, image_adv, inner_iteration = self.iterative_gradient_descent(h, image_label, label_true,
                                                                                     num_iterations, alpha=inner_alpha)
                if h_lava is not None:
                    print(f"Misclassification achieved at iteration {iteration}, inner iteration {inner_iteration}")
                    return h_lava, image_adv

            h = h_lava

        return image_adv, h_lava

    def stack_to_xai(self):
        x_input = Input(shape=self.decoder.input_shape)
        x_dec = self.decoder(x_input)
        x_img = Reshape((28, 28, 1))(x_dec)
        y = self.model(x_img)
        return Model(inputs=x_input, outputs=y)

    def generate_image_via_h(self, h, image_label, step=0.5):
        x = np.array([h])
        y = np.array([image_label])
        g = gradient_of_x(x, y, self.xai)
        g_npy = np.squeeze(g.numpy())
        h_lava = self.get_h_lava_via_one_step(x, g_npy, step=step)
        image_adv = self.decoder.predict(h_lava)[0].reshape((28, 28, 1))
        return image_adv, h_lava

    def iterative_gradient_descent(self, h, image_label, label_true, num_iterations, alpha):
        for iteration in range(num_iterations):
            h_lava, image_adv = self.generate_image_via_h(h, image_label, step=alpha)
            if self.predict_label(self.model, image_adv) != label_true:
                return h_lava, image_adv, iteration
            h = h_lava
        return None, None, None

    @staticmethod
    def get_h_lava_via_one_grid_step(h, gradient, step=0.5):
        max_grad_index = np.argmax(np.abs(gradient))
        h_lava = np.copy(h)  # _lava means latent variants
        h_lava[0, max_grad_index] += step * np.sign(gradient[max_grad_index])
        return h_lava

    @staticmethod
    def get_h_lava_via_one_step(h, gradient, step=0.5):
        h_lava = np.copy(h)  # _lava means latent variants
        h_lava += step * gradient
        return h_lava

    @staticmethod
    def predict_label(cnn, image):
        return np.argmax(cnn.predict(np.array([image]))[0])