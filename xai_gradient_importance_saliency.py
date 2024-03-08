import tensorflow as tf
from train_vae import *
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore

from xai_gradient_importance import *

def saliency_map_of_x(x, y, model):
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)
    number = y.shape[-1]
    score = CategoricalScore(number)
    return saliency(score, x)


if __name__ == '__main__':
    # 1. data
    # load data
    train_images, train_labels, test_images, test_labels = mnist_data()
    samples, sample_labels = sample_and_categorize(train_images, train_labels, number=3000)
    print(samples.shape)
    print(sample_labels.shape)

    # reshape data
    x_train = np.reshape(samples, (-1, 784))
    # x_train = np.reshape(train_images, (-1, 784))
    print(x_train.shape)

    # 2. model: train a VAE
    model_pre = VAE(latent_dim=12)

    # Fit the model. Note that the 'eps' input is ignored because it is an Input tensor.
    dummy_eps_input = np.zeros((len(x_train), model_pre.latent_dim))
    history = model_pre.model.fit([x_train, dummy_eps_input], x_train, shuffle=True, epochs=50, batch_size=100)
    # plot_learning_curve(history)
    model_pre.save()

    exit()

    # visualize latent space in 2 dimension
    # tsne = TSNE(n_components=2, random_state=42)
    # encodings_samples_2d = tsne.fit_transform(encodings_samples)
    # plot_encodings2d_with_labels(encodings_samples_2d, sample_labels)

    # XAI framework
    vae = VAE.load("trained_models")
    cnn = load_model("trained_models/CNN/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))
    xai.summary()

    x_samples = np.reshape(samples, (-1, 784))
    encodings_samples = vae.encoder.predict(x_samples)
    sample_labels_onehot = tf.one_hot(tf.constant(sample_labels), depth=10).numpy()

    count = 10
    for i in range(len(x_samples)):
        x = np.array([encodings_samples[i]])
        y = np.array([sample_labels_onehot[i]])
        g = saliency_map_of_x(x, y, xai)

        g_abs = np.abs(np.squeeze(g.numpy()))

        plot_image_and_gradient(np.reshape(x_samples[i], (28, 28)), g_abs)

        # Identify the maximum gradient entry
        max_grad_index = np.argmax(g_abs)
        # latent_space_display(x, vae.decoder, highlight_dim=max_grad_index)
        latent_space_display_mark(x, sample_labels[i], vae.decoder, xai, highlight_dim=max_grad_index)

        if count <= 0:
            break
        count -= 1
