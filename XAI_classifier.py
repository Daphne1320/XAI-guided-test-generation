from tensorflow.python.keras.layers import Input, Reshape
from tensorflow.python.keras.models import Model


def xai_model_cvae(decoder, nn, input_shape=(2,), name="xai"):
    x_input = Input(shape=input_shape)
    x_dec = decoder(x_input)
    x_img = Reshape((28, 28, 1))(x_dec)
    y = nn(x_img)
    return Model(inputs=x_input, outputs=y, name=name)


def xai_model_vae(decoder, nn, input_shape=(2,)):
    x_input = Input(shape=input_shape)
    x_dec = decoder(x_input)
    x_img = Reshape((28, 28, 1))(x_dec)
    y = nn(x_img)
    return Model(inputs=x_input, outputs=y)
