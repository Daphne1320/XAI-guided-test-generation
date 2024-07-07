# -*- coding: utf-8 -*-

from __future__ import print_function

from PIL import Image
from tqdm import tqdm
import argparse

from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

from dlfuzz import DLFuzz
from Model1 import Model1
from utils import *


def dlfuzz_load_model(model_name, input_tensor):
    if model_name == 'model1':
        model = Model1(input_tensor=input_tensor)
    # elif model_name == 'model2':
    #    model = Model2(input_tensor=input_tensor)
    # elif model_name == 'model3':
    #    model = Model3(input_tensor=input_tensor)
    else:
        print('please specify correct model name')
        os._exit(0)
    print(model.name)
    return model


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some parameters for neuron selection strategy.")

    # Add arguments with default values
    parser.add_argument(
        'neuron_select_strategy',
        type=int,
        default=[2],
        nargs='+',  # This will capture all following inputs as part of the list
        help='The strategy to select neurons (default: [2])'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='The threshold value (default: 0.5)'
    )
    parser.add_argument(
        '--neuron_to_cover_num',
        type=int,
        default=5,
        help='The number of neurons to cover (default: 5)'
    )
    parser.add_argument(
        '--subdir',
        type=str,
        default='0602',
        help='The subdirectory to work in (default: "0602")'
    )
    parser.add_argument(
        '--iteration_times',
        type=int,
        default=5,
        help='The number of iteration times (default: 5)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='model1',
        help='The name of the model (default: "model1")'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    neuron_select_strategy = args.neuron_select_strategy
    threshold = args.threshold
    neuron_to_cover_num = args.neuron_to_cover_num
    subdir = args.subdir
    iteration_times = args.iteration_times
    model_name = args.model_name

    # neuron_select_strategy, threshold, neuron_to_cover_num, subdir, iteration_times, model_name = [2], 0.5, 5, "0602", 5, "model1"

    neuron_to_cover_weight = 0.5
    predict_weight = 0.5
    learning_step = 0.02

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)
    model = dlfuzz_load_model(model_name, input_tensor)

    img_dir = 'MNIST/seeds_50'
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)

    save_dir = './generated_inputs/' + subdir + '/'
    clear_up_dir(save_dir)

    K.set_learning_phase(0)
    dlfuzz = DLFuzz(model, neuron_select_strategy, threshold, neuron_to_cover_num, iteration_times,
                    neuron_to_cover_weight, predict_weight, learning_step)

    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        # load image
        img_path = os.path.join(img_dir, img_path)
        # print(img_path)
        tmp_img = load_image(img_path)

        # calculate fuzz image
        gen_img = dlfuzz.generate_fuzzy_image(tmp_img)

        # save image
        # mannual_label = int(img_name.split('_')[1])
        gen_img_deprocessed = deprocess_image(gen_img)
        img_name = img_paths[i].split('.')[0]
        save_img = save_dir + img_name + '_' + str(get_signature()) + '.png'
        Image.fromarray(gen_img_deprocessed).save(save_img)

"""

python gen_diff.py 1 2 3 --threshold 0.7 --neuron_to_cover_num 10 --subdir ./subdir --iteration_times 10 --model_name model2

"""
