import numpy as np
from absl import app, flags

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

FLAGS = flags.FLAGS
flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
flags.DEFINE_bool(
    "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
)


class CleverHans:
    FAST_GRADIENT = "fast_gradient"
    PROJ_GRADIENT = "projected_gradient"

    def __init__(self, model):
        self.model = model

    def generate_adversarial_image(self, image, method=None):
        method = self.FAST_GRADIENT if method is None else method
        if method == self.FAST_GRADIENT:
            return fast_gradient_method(self.model, [image], FLAGS.eps, np.inf)[0]
        elif method == self.PROJ_GRADIENT:
            return projected_gradient_descent(self.model, [image], FLAGS.eps, 0.01, 40, np.inf)[0]