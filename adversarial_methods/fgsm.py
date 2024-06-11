import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model

from xai import gradient_of_x


class FGSM():

    def __init__(self, model):
        self.model = model

    def generate_adversarial_image(self, image, image_label, step=0.5):
        """

        Args:
            image: np.array, of input shape of the self.model
            image_label: onehot vector
            step:

        Returns:

        """
        image = np.reshape(image, self.model.input_shape[1:])
        x = np.array([image])
        y = np.array([image_label])
        g = gradient_of_x(x, y, self.model)
        g_npy = np.squeeze(g.numpy())
        g_sign = np.reshape(tf.sign(g_npy), self.model.input_shape[1:])

        image_adv = np.copy(image)  # _lava means latent variants
        image_adv += step * g_sign
        return image_adv