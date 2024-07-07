import tensorflow as tf

from validity_check.validity_check_xmutant import compute_valid


class VAEValidation():

    def __init__(self, vae_threshold=0.26608911681183206):
        VAE = "mnist_vae_all_classes"
        # VAE = "mnist_vae_stocco_all_classes"
        self.decoder = tf.keras.models.load_model("./validity_check/trained/" + VAE + "/decoder", compile=False)
        self.encoder = tf.keras.models.load_model("./validity_check/trained/" + VAE + "/encoder", compile=False)
        self.vae_threshold = vae_threshold

    def validate(self, image):
        image = image.reshape((28, 28, 1))
        distr, loss = compute_valid(image, self.encoder, self.decoder, self.vae_threshold)
        return distr
