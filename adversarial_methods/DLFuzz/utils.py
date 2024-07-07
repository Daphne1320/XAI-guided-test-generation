# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
from datetime import datetime

from tensorflow.keras.preprocessing import image


def clear_up_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for i in os.listdir(directory):
            path_file = os.path.join(directory, i)
            if os.path.isfile(path_file):
                os.remove(path_file)


def load_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def get_signature():
    now = datetime.now()
    past = datetime(2015, 6, 6, 0, 0, 0, 0)
    timespan = now - past
    time_sig = int(timespan.total_seconds() * 1000)

    return str(time_sig)
