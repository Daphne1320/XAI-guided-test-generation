from xai import *

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

    # XAI framework
    vae = VAE.load("trained_models")
    cnn = load_model("trained_models/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))
    xai.summary()

    x_samples = np.reshape(samples, (-1, 784))
    encodings_samples = vae.encoder.predict(x_samples)
    sample_labels_onehot = tf.one_hot(tf.constant(sample_labels), depth=10).numpy()

    # Specify the label you're interested in
    interested_label = 2

    count = 10
    for i in range(len(x_samples)):
        if sample_labels[i] != interested_label:
            continue

        x = np.array([encodings_samples[i]])
        y = np.array([sample_labels_onehot[i]])
        g = gradient_of_x(x, y, xai)

        g_abs = np.abs(np.squeeze(g.numpy()))

        plot_image_and_gradient(np.reshape(x_samples[i], (28, 28)), g_abs)

        # Identify the maximum gradient entry
        max_grad_index = np.argmax(g_abs)
        # latent_space_display(x, vae.decoder, highlight_dim=max_grad_index)
        latent_space_display_mark(x, sample_labels[i], vae.decoder, xai, highlight_dim=max_grad_index)

        if count <= 0:
            break
        count -= 1
