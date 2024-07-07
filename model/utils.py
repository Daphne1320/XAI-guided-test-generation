from tensorflow.keras.models import clone_model

def clone_encoder(encoder):
    cloned_encoder = clone_model(encoder)
    cloned_encoder.set_weights(encoder.get_weights())
    cloned_encoder.trainable = False
    return cloned_encoder