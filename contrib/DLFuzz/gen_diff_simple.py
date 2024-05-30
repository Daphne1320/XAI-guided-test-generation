# -*- coding: utf-8 -*-

from __future__ import print_function

from PIL import Image
import os
from tqdm import tqdm

from tensorflow.keras.layers import Input, Activation
from tensorflow.keras import backend as K

from dlfuzz import DLFuzz
from gen_diff import dlfuzz_load_model, clear_up_dir, load_image, deprocess_image, get_signature

if __name__ == "__main__":

    model = dlfuzz_load_model("model1", Input(shape=(28, 28, 1)))

    # input images
    img_dir = './MNIST/seeds_50'
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)

    # output images
    save_dir = './MNIST/generated_inputs/0510'
    clear_up_dir(save_dir)

    # prepare
    K.set_learning_phase(0)
    dlfuzz = DLFuzz(model)

    # start
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        # load image
        img_path = os.path.join(img_dir, img_path)
        # print(img_path)
        tmp_img = load_image(img_path)

        # calculate fuzz image
        gen_img = dlfuzz.generate_fuzzy_image(tmp_img)

        # save image
        # mannual_label = int(img_name.split('_')[1])
        gen_img_deprocessed = deprocess_image(gen_img.numpy())
        img_name = img_paths[i].split('.')[0]
        save_img = save_dir + img_name + '_' + str(get_signature()) + '.png'
        Image.fromarray(gen_img_deprocessed).save(save_img)
