import csv
import glob
import math
import ntpath
import os

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import pandas as pd

image_size = 28
image_chn = 1
input_shape = (image_size, image_size, image_chn)

VAE_THRESHOLDS = {"0.68": 0.18253154154176687,
                  "0.9": 0.23811348920483438,
                  "0.95": 0.26608911681183206,
                  "0.99": 0.32404093551779506,
                  "0.999": 0.39775952595272196,
                  "0.9999": 0.4656065673983675,
                  "0.99999": 0.5298851235622417}


# Logic for calculating reconstruction probability
def reconstruction_probability(dec, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """
    sampled_zs = sampling([z_mean, z_log_var])
    mu_hat = dec(sampled_zs)

    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(X, mu_hat), axis=(-1)
        )
    )

    return reconstruction_loss


# Calculates and returns probability density of test input
def calculate_density(x_target_orig, enc, dec):
    x_target_orig = np.clip(x_target_orig, 0, 1)
    x_target = np.reshape(x_target_orig, (-1, 28 * 28))
    z_mean, z_log_var, _ = enc(x_target)
    reconstructed_prob_x_target = reconstruction_probability(dec, z_mean, z_log_var, x_target)
    return reconstructed_prob_x_target


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def compute_valid(sample, encoder, decoder, tshd):
    # fp = []
    # tn = []
    # for batch in anomaly_test:
    rec_loss = calculate_density(sample, encoder, decoder)
    if rec_loss > tshd or math.isnan(rec_loss):
        distr = 'ood'
        # tn.append(rec_loss)
    else:
        distr = 'id'
    return distr, rec_loss.numpy()
    # fp.append(rec_loss)

    # print("id: " + str(len(fp)))
    # print("ood: " + str(len(tn)))


def main():
    csv_file = r"losses/ood_analysis_xmutant_all_classes.csv"
    # VAE density threshold for classifying invalid inputs
    vae_threshold = 0.26608911681183206
    VAE = "mnist_vae_all_classes"

    decoder = tf.keras.models.load_model("trained/" + VAE + "/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/" + VAE + "/encoder", compile=False)

    RESULTS_PATH = r"../result/digits/S_R_sm/"

    with open(csv_file, 'e', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TOOL', 'SAMPLE', 'ID/OOD', 'loss'])

        print("== XMutant ==")
        DJ_FOLDER = RESULTS_PATH + "*.npy"
        filelist = [f for f in glob.glob(DJ_FOLDER)]

        count_ids = 0
        if len(filelist) != 0:
            for sample in filelist:
                # TODO: correct the input images to drop the white frame, a reshape only is not a fix
                s = np.load(sample)  # .reshape(28, 28, 1)
                # plt.figure()
                # plt.imshow(s, cmap='gray')
                # plt.axis('off')
                # plt.show()
                distr, loss = compute_valid(s, encoder, decoder, vae_threshold)
                if distr == 'id':
                    count_ids += 1
                sample_name = ntpath.split(sample)[-1]
                writer.writerow(['XMutant', sample_name, distr, loss])

        print("Found samples: " + str(len(filelist)))
        print("Valid (in-distribution): %d (%d%%)" % (count_ids, count_ids / len(filelist) * 100))


def main_xmutant(record_name, vae_threshold=0.26608911681183206):
    # VAE density threshold for classifying invalid inputs

    VAE = "mnist_vae_all_classes"
    # VAE = "mnist_vae_stocco_all_classes"
    decoder = tf.keras.models.load_model("trained/" + VAE + "/decoder", compile=False)
    encoder = tf.keras.models.load_model("trained/" + VAE + "/encoder", compile=False)

    record_df = pd.read_csv(record_name)
    # record_df["ID/OOD"] = None
    record_df["loss"] = None

    for index, row in record_df.iterrows():
        np_array = np.load(row['image_path'] + '.npy')  # .reshape(28, 28, 1)
        distr, loss = compute_valid(np_array, encoder, decoder, vae_threshold)

        # record_df.at[index, "ID/OOD"] = distr
        record_df.at[index, "loss"] = loss

    record_df.to_csv(record_name, index=False)


def new_threshold(record_name: str):
    record_df = pd.read_csv(record_name)

    for confidence, threshold in VAE_THRESHOLDS.items():
        record_df["ID_" + confidence] = [1 if loss < threshold else 0 for loss in record_df["loss"]]

    record_df.to_csv(record_name, index=False)


def check_validity_rate_over_iteration(record_name, digit=None):
    RESULTS_PATH = r"../result/digits/"
    df_record = pd.read_csv(record_name)

    if digit is not None:
        df_record = df_record[df_record["expected_label"] == digit]

    # ----------------------------------------------------------------
    df_record = df_record[df_record['mutation_number'] != 1]
    df_record['mutation_number'] = df_record['mutation_number'] - 1
    # df_record['ID'] = df_record['ID/OOD'].apply(lambda x:1 if x.lower()== "id" else 0)
    df_record['one'] = 1

    df_record['mutation_number'] = df_record['mutation_number'].astype(int)

    df_cumulative_validity = pd.DataFrame()
    df_cumulative_validity["idx"] = df_record.groupby('mutation_number')['ID_0.9'].sum().to_frame().index

    for confidence, threshold in VAE_THRESHOLDS.items():
        df_cumulative_validity["pop_num"] = df_record.groupby('mutation_number')['one'].sum().to_list()
        df_cumulative_validity["pop_cum_num"] = df_record.groupby('mutation_number')['one'].sum().cumsum().to_list()

        df_cumulative_validity["ID_num" + "_" + confidence] = (
            df_record.groupby('mutation_number')['ID' + "_" + confidence].sum().to_list())
        df_cumulative_validity["ID_cum_num" + "_" + confidence] = (
            df_record.groupby('mutation_number')['ID' + "_" + confidence].sum().cumsum().to_list())

        df_cumulative_validity["ID_cum_rate" + "_" + confidence] = (
                df_cumulative_validity["ID_cum_num" + "_" + confidence] / df_cumulative_validity["pop_cum_num"]
        ).round(decimals=3)

        df_cumulative_validity["ID_rate" + "_" + confidence] = (
                df_cumulative_validity["ID_num" + "_" + confidence] / df_cumulative_validity["pop_num"]
        ).round(decimals=3)

    df_cumulative_validity.to_csv(
        os.path.join(RESULTS_PATH, "cumulative_clear_validity_rate_" + record_name.split("/")[-1]))


if __name__ == "__main__":
    '''
    Usage: place the MNIST digits for validation in the folder 'mnist/selforacle/generated_images/mnist_inputs/mnist_xm',
     either in npy or png format (other formats are currently not supported). 
    Then run the main function to calculate the reconstruction probability of the images and classify them as ID or OOD.
    The results are stored in a csv file 'ood_analysis_xmutant_all_classes.csv' within the 'mnist/selforacle/losses' folder.
    A different threshold value can be selected looking at the selforacle_thresholds_all_classes.json file.
    '''

    # tf.keras.utils.set_random_seed(0)
    # main()

    # xai_types_final = ['C_C_IG', 'C_R_IG', 'R_R']
    # vae_threshold = 0.26608911681183206
    # record_name = "record_R_R.csv"
    # main_xc(record_name) # ONLY DO IT ONCE FOR EACH CSV
    # for i in range(10):

    # new_threshold(record_name="record_C_C.csv")
    print(os.getcwd())
    # os.chdir("./validity_check")
    print(os.getcwd())
    RESULTS_PATH = r"../result/digits"
    folders = glob.glob(os.path.join(RESULTS_PATH, "record*.csv"))
    print(folders)
    for csv_file in folders:
        # main_xc(csv_file) #get loss
        # new_threshold(record_name=csv_file) # compare all thresholds
        check_validity_rate_over_iteration(record_name=csv_file, digit=None)  # get cumulative validity rate
