# -*- coding: utf-8 -*-

from __future__ import print_function

import random
import numpy as np

from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

model_layer_weights_top_k = []


class DLFuzz:
    """
    model_layer_times is a dictionary: key:(layer_name, index), value:times

    """

    def __init__(self, model,  neuron_select_strategy=None, threshold=0.5, neuron_to_cover_num=5, iteration_times=5,
                    neuron_to_cover_weight=.05, predict_weight=0.5, learning_step=0.02):
        self.model = model
        self.neuron_select_strategy = ["2"] if neuron_select_strategy is None else neuron_select_strategy
        self.threshold = threshold
        self.neuron_to_cover_num = neuron_to_cover_num
        self.iteration_times = iteration_times
        self.neuron_to_cover_weight = neuron_to_cover_weight
        self.predict_weight = predict_weight
        self.learning_step = learning_step

        # extreme value means the activation value for a neuron can be as high as possible ...
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            self.neuron_to_cover_weight = 2

        self.model_layer_times1 = self.init_coverage_times(model)  # times of each neuron covered
        self.model_layer_times2 = self.init_coverage_times(model)  # update when new image and adversarial images found
        self.model_layer_value1 = self.init_coverage_value(model)

    @staticmethod
    def init_times(model, model_layer_times):
        """init times with zeros"""
        for layer in model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = 0

    @staticmethod
    def init_coverage_times(model):
        model_layer_times = defaultdict(int)
        DLFuzz.init_times(model, model_layer_times)
        return model_layer_times

    @staticmethod
    def init_coverage_value(model):
        """??? not the same as init_coverage_times ???"""
        model_layer_value = defaultdict(float)
        DLFuzz.init_times(model, model_layer_value)
        return model_layer_value

    @staticmethod
    def neuron_to_cover(not_covered, model_layer_dict):
        if not_covered:
            layer_name, index = random.choice(not_covered)
            not_covered.remove((layer_name, index))
        else:
            layer_name, index = random.choice(model_layer_dict.keys())
        return layer_name, index

    @staticmethod
    def scale(intermediate_layer_output, rmax=1, rmin=0):
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
                intermediate_layer_output.max() - intermediate_layer_output.min())
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    @staticmethod
    def random_strategy(model, model_layer_times, neuron_to_cover_num):
        loss_neuron = []
        not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_times.items() if v == 0]
        for _ in range(neuron_to_cover_num):
            layer_name, index = DLFuzz.neuron_to_cover(not_covered, model_layer_times)
            loss00_neuron = K.mean(model.get_layer(layer_name).output[..., index])
            # if loss_neuron == 0:
            #     loss_neuron = loss00_neuron
            # else:
            #     loss_neuron += loss00_neuron
            # loss_neuron += loss1_neuron
            loss_neuron.append(loss00_neuron)
        return loss_neuron

    @staticmethod
    def update_coverage(input_data, model, model_layer_times, threshold=0.0):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and
                       'input' not in layer.name and
                       'reshape' not in layer.name and
                       'dropout' not in layer.name]
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            intermediate_layer_output = intermediate_layer_output.numpy()
            scaled = DLFuzz.scale(intermediate_layer_output[0])
            # xrange(scaled.shape[-1])
            for num_neuron in range(scaled.shape[-1]):
                if np.mean(scaled[
                               ..., num_neuron]) > threshold:  # and model_layer_dict[(layer_names[i], num_neuron)] == 0:
                    model_layer_times[(layer_names[i], num_neuron)] += 1

        return intermediate_layer_outputs

    @staticmethod
    def update_coverage_value(input_data, model, model_layer_value):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]

        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs = intermediate_layer_model(input_data)

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            intermediate_layer_output = intermediate_layer_output.numpy()
            scaled = DLFuzz.scale(intermediate_layer_output[0])
            # xrange(scaled.shape[-1])
            for num_neuron in range(scaled.shape[-1]):
                model_layer_value[(layer_names[i], num_neuron)] = np.mean(scaled[..., num_neuron])

        return intermediate_layer_outputs

    @staticmethod
    def neuron_select_high_weight(model, layer_names, top_k):
        global model_layer_weights_top_k
        model_layer_weights_dict = {}
        for layer_name in layer_names:
            weights = model.get_layer(layer_name).get_weights()
            if len(weights) <= 0:
                continue
            w = np.asarray(weights[0])  # 0 is weights, 1 is biases
            w = w.reshape(w.shape)
            for index in range(model.get_layer(layer_name).output_shape[-1]):
                index_w = np.mean(w[..., index])
                if index_w <= 0:
                    continue
                model_layer_weights_dict[(layer_name, index)] = index_w
        # notice!
        model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)

        k = 0
        for (layer_name, index), weight in model_layer_weights_list:
            if k >= top_k:
                break
            model_layer_weights_top_k.append([layer_name, index])
            k += 1

    @staticmethod
    def neuron_selection(model, input_tensor, model_layer_times, model_layer_value, neuron_select_strategy,
                         neuron_to_cover_num,
                         threshold):
        if neuron_select_strategy == 'None':
            return DLFuzz.random_strategy(model, model_layer_times, neuron_to_cover_num)

        num_strategy = len([x for x in neuron_select_strategy if x in ['0', '1', '2', '3']])
        neuron_to_cover_num_each = neuron_to_cover_num // num_strategy

        loss_neuron = []
        # initialization for strategies
        if ('0' in list(neuron_select_strategy)) or ('1' in list(neuron_select_strategy)):
            i = 0
            neurons_covered_times = []
            neurons_key_pos = {}
            for (layer_name, index), time in model_layer_times.items():
                neurons_covered_times.append(time)
                neurons_key_pos[i] = (layer_name, index)
                i += 1
            neurons_covered_times = np.asarray(neurons_covered_times)
            times_total = sum(neurons_covered_times)

        # select neurons covered often
        if '0' in list(neuron_select_strategy):
            if times_total == 0:
                return DLFuzz.random_strategy(model, model_layer_times, 1)  # The beginning of no neurons covered
            neurons_covered_percentage = neurons_covered_times / float(times_total)
            # num_neuron0 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage)
            num_neuron0 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False,
                                           p=neurons_covered_percentage)
            for num in num_neuron0:
                layer_name0, index0 = neurons_key_pos[num]
                # loss0_neuron = tf.reduce_mean(model.get_layer(layer_name0).output[..., index0])
                loss0_neuron = tf.reduce_mean(
                    Model(inputs=model.inputs, outputs=model.get_layer().output)(input_tensor)[[..., index0]])
                loss_neuron.append(loss0_neuron)

        # select neurons covered rarely
        if '1' in list(neuron_select_strategy):
            if times_total == 0:
                return DLFuzz.random_strategy(model, model_layer_times, 1)
            neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)
            neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(
                sum(neurons_covered_times_inverse))
            # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
            num_neuron1 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False,
                                           p=neurons_covered_percentage_inverse)
            for num in num_neuron1:
                layer_name1, index1 = neurons_key_pos[num]
                # loss1_neuron = tf.reduce_mean(model.get_layer(layer_name1).output[..., index1])
                loss1_neuron = tf.reduce_mean(
                    Model(inputs=model.inputs, outputs=model.get_layer(layer_name1).output)(input_tensor)[..., index1])

                loss_neuron.append(loss1_neuron)

        # select neurons with largest weights (feature maps with largest filter weights)
        if '2' in list(neuron_select_strategy):
            layer_names = [layer.name for layer in model.layers if
                           'flatten' not in layer.name and 'input' not in layer.name]
            k = 0.1
            top_k = k * len(model_layer_times)  # number of neurons to be selected within
            global model_layer_weights_top_k
            if len(model_layer_weights_top_k) == 0:
                DLFuzz.neuron_select_high_weight(model, layer_names, top_k)  # Set the value

            num_neuron2 = np.random.choice(range(len(model_layer_weights_top_k)), neuron_to_cover_num_each,
                                           replace=False)
            for i in num_neuron2:
                # i = np.random.choice(range(len(model_layer_weights_top_k)))
                layer_name2 = model_layer_weights_top_k[i][0]
                index2 = model_layer_weights_top_k[i][1]
                # loss2_neuron = tf.reduce_mean(model.get_layer(layer_name2).output[..., index2])
                loss2_neuron = tf.reduce_mean(
                    Model(inputs=model.inputs, outputs=model.get_layer(layer_name2).output)(input_tensor)[..., index2])

                loss_neuron.append(loss2_neuron)

        if '3' in list(neuron_select_strategy):
            above_threshold = []
            below_threshold = []
            above_num = neuron_to_cover_num_each / 2
            below_num = neuron_to_cover_num_each - above_num
            above_i = 0
            below_i = 0
            for (layer_name, index), value in model_layer_value.items():
                if threshold + 0.25 > value > threshold and layer_name != 'fc1' and layer_name != 'fc2' and \
                        layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                        and above_i < above_num:
                    above_threshold.append([layer_name, index])
                    above_i += 1
                    # print(layer_name,index,value)
                    # above_threshold_dict[(layer_name, index)]=value
                elif threshold > value > threshold - 0.2 and layer_name != 'fc1' and layer_name != 'fc2' and \
                        layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                        and below_i < below_num:
                    below_threshold.append([layer_name, index])
                    below_i += 1
            #
            # loss3_neuron_above = 0
            # loss3_neuron_below = 0
            loss_neuron = []
            if len(above_threshold) > 0:
                for above_item in range(len(above_threshold)):
                    loss3_neuron_above = tf.reduce_mean(
                        Model(inputs=model.inputs, outputs=model.get_layer(above_threshold[above_item][0]).output)(
                            input_tensor)[..., above_threshold[above_item][1]])
                    loss_neuron.append(loss3_neuron_above)

            if len(below_threshold) > 0:
                for below_item in range(len(below_threshold)):
                    loss3_neuron_below = -tf.reduce_mean(
                        Model(inputs=model.inputs, outputs=model.get_layer(below_threshold[below_item][0]).output)(
                            input_tensor)[..., below_threshold[below_item][1]])
                    loss_neuron.append(loss3_neuron_below)

            # loss_neuron += loss3_neuron_below - loss3_neuron_above

            # for (layer_name, index), value in model_layer_value.items():
            #     if 0.5 > value > 0.25:
            #         above_threshold.append([layer_name, index])
            #     elif 0.25 > value > 0.2:
            #         below_threshold.append([layer_name, index])
            # loss3_neuron_above = 0
            # loss3_neuron_below = 0
            # if len(above_threshold)>0:
            #     above_i = np.random.choice(range(len(above_threshold)))
            #     loss3_neuron_above = tf.reduce_mean(model.get_layer(above_threshold[above_i][0]).output[..., above_threshold[above_i][1]])
            # if len(below_threshold)>0:
            #     below_i = np.random.choice(range(len(below_threshold)))
            #     loss3_neuron_below = tf.reduce_mean(model.get_layer(below_threshold[below_i][0]).output[..., below_threshold[below_i][1]])
            # loss_neuron += loss3_neuron_below - loss3_neuron_above
            if loss_neuron == 0:
                return DLFuzz.random_strategy(model, model_layer_times, 1)  # The beginning of no neurons covered

        return loss_neuron

    @staticmethod
    def neuron_covered(model_layer_times):
        covered_neurons = len([v for v in model_layer_times.values() if v > 0])
        total_neurons = len(model_layer_times)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    @staticmethod
    def before_softmax_model(model):
        if isinstance(model.layers[-1], Activation):
            return Model(inputs=model.inputs, outputs=model.get_layer('before_softmax').output)
        else:
            # Get the configuration of the original last layer
            last_layer = model.layers[-1]
            last_layer_config = last_layer.get_config()
            last_layer_config["name"] += "_clone"
            last_layer_config['activation'] = None  # Set activation to None

            # Create a new Dense layer with the modified configuration
            new_last_layer = Dense(**last_layer_config)

            # Apply the new layer to the output of the penultimate layer
            output_before_softmax = new_last_layer(model.layers[-2].output)

            # Create the new model
            new_model = Model(inputs=model.inputs, outputs=output_before_softmax)

            # Set weights of the new last layer to match those of the original last layer
            new_last_layer.set_weights(last_layer.get_weights())

            return new_model

    def loss_function(self, image_data):
        input_tensor = tf.Variable(initial_value=image_data)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            pred1 = self.model(input_tensor)
            # orig_pred = pred1

            DLFuzz.update_coverage_value(image_data, self.model, self.model_layer_value1)
            DLFuzz.update_coverage(image_data, self.model, self.model_layer_times1, self.threshold)

            label1 = np.argmax(pred1[0])
            orig_label = label1
            label_top5 = np.argsort(pred1[0])[-5:]

            # model_label = Model(inputs=self.model.inputs, outputs=self.model.get_layer('before_softmax').output)
            # output1 = model_label(input_tensor)
            # loss_1 = tf.reduce_mean(output1[..., orig_label])
            loss_1 = tf.reduce_mean(self.before_softmax_model(self.model)(input_tensor)[..., orig_label])
            loss_2 = tf.reduce_mean(self.before_softmax_model(self.model)(input_tensor)[..., label_top5[-2]])
            loss_3 = tf.reduce_mean(self.before_softmax_model(self.model)(input_tensor)[..., label_top5[-3]])
            loss_4 = tf.reduce_mean(self.before_softmax_model(self.model)(input_tensor)[..., label_top5[-4]])
            loss_5 = tf.reduce_mean(self.before_softmax_model(self.model)(input_tensor)[..., label_top5[-5]])

            layer_output = (self.predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)
            # layer_output = loss_1

            # neuron coverage loss
            loss_neuron = DLFuzz.neuron_selection(self.model, input_tensor, self.model_layer_times1,
                                                  self.model_layer_value1,
                                                  self.neuron_select_strategy,
                                                  self.neuron_to_cover_num, self.threshold)

            layer_output += self.neuron_to_cover_weight * tf.reduce_sum(loss_neuron)

            # for adversarial image generation
            final_loss = tf.reduce_mean(layer_output)

        gradients = tape.gradient(final_loss, input_tensor)

        # we compute the gradient of the input picture wrt this loss
        grads = self.normalize(gradients[0])

        grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        return grads_tensor_list, orig_label

    def iterate_condition(self, gen_img, orig_img, iter):
        # further process condition 1
        # previous accumulated neuron coverage
        previous_coverage = self.neuron_covered(self.model_layer_times1)[2]
        self.update_coverage(gen_img, self.model, self.model_layer_times1, self.threshold)  # for seed selection
        current_coverage = self.neuron_covered(self.model_layer_times1)[2]

        # further process condition 2
        diff_img = gen_img - orig_img
        L2_norm = np.linalg.norm(diff_img)
        orig_L2_norm = np.linalg.norm(orig_img)
        perturb_adversial = L2_norm / orig_L2_norm
        return current_coverage - previous_coverage > 0.01 / (iter + 1) and perturb_adversial < 0.02

    def generate_adversarial_image(self, tmp_img):
        tmp_img = np.array([tmp_img], dtype="float32")
        orig_img = tmp_img.copy()

        # start_time = time.clock()
        self.update_coverage(tmp_img, self.model, self.model_layer_times2, self.threshold)

        process_queue_for_the_img = []
        process_queue_for_the_img.append(tmp_img)
        while len(process_queue_for_the_img) > 0:
            gen_img = process_queue_for_the_img.pop(0)
            self.loss_function(gen_img)

            # we run gradient ascent for 3 steps
            for iter in range(self.iteration_times):

                loss_neuron_list, orig_label = self.loss_function(gen_img)

                perturb = loss_neuron_list[-1] * self.learning_step
                gen_img += perturb

                if self.iterate_condition(gen_img, orig_img, iter):
                    process_queue_for_the_img.append(gen_img)
                    # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)

                pred1 = self.model.predict(gen_img)
                label1 = np.argmax(pred1[0])
                if label1 != orig_label:
                    self.update_coverage(gen_img, self.model, self.model_layer_times2, self.threshold)

        # end_time = time.clock()

        # verbose
        # print('covered neurons percentage %d neurons %.3f'
        #      % (len(model_layer_times2), DLFuzz.neuron_covered(model_layer_times2)[2]))
        # duration = end_time - start_time
        # print('used time : ' + str(duration))

        # total_time += duration
        # total_norm += L2_norm
        # total_perturb_adversial += perturb_adversial
        # print('L2 norm : ' + str(L2_norm))
        # print('ratio perturb = ', perturb_adversial)

        # adversial_num += 1
        return gen_img.numpy()[0]

    @staticmethod
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
