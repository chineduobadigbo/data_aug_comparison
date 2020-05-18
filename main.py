from CCVAE import CCVAE
from CCGAN import CCGAN
from Classifier import Classifier
from load_mnist import load_mist, load_classifier_data, load_generated_imgs, onehot_encode, SEED
import tensorflow as tf
from gan_functions import train_gan
from vae_functions import train_vae
from classifier_functions import train_classifier
from generate_samples import VAE_Generate_Mult, GAN_Generate_Mult
from save_run import save_model, save_cycle, save_run, save_directory, read_run_dict, read_run_num, increase_run_num
from load_logger import load_logger_conf
import argparse
import os
import json
import traceback
import logging
import sys
import numpy as np
import time
from PIL import Image
import gc
import random

def reset_seeds():
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    print("RANDOM SEEDS RESET")

def create_img_label_dirs(img_path):
    imgs_paths = []
    for i in range(10):
        new_path = img_path+'label_{}/'.format(i)
        save_directory(new_path)
        imgs_paths.append(new_path)
    return imgs_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script runs through the whole training process.')
    parser.add_argument('--run', type=int, help='this is the run num')
    parser.add_argument('--cycle', type=int, help='this is the cycle num')
    args = parser.parse_args()
    # set up logger
    load_logger_conf()
    try:
        logging.info('Run: {}'.format(args.run))
        logging.info('Cycle: {}'.format(args.cycle))
        # num_new_samples is per class
        experiment_cycles = read_run_dict()
        label_num = 10

        current_run_num = args.run
        # create run number dir
        new_run_path = './runs/run_{}/'.format(current_run_num)
        save_directory(new_run_path)

        cycle = str(args.cycle)
        cycle_dict = experiment_cycles[cycle]

        #tf.config.experimental_run_functions_eagerly(True)
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()
        print('Starting new cycle with number: {}'.format(cycle))
        batch_size = cycle_dict['batch_size']
        num_epochs = cycle_dict['num_epochs']
        num_epochs_cl = cycle_dict['num_epochs_cl']
        # create cycle number dir
        new_cycle_path = new_run_path+'cycles/cycle_{}/'.format(cycle)
        save_directory(new_cycle_path)
        # create new model dir 
        new_model_path = new_cycle_path+'models/'
        save_directory(new_model_path)

        print('Loading data.')
        # load training set
        data = load_mist(cycle_dict['samples'], fashion=cycle_dict['fashion'])

        # training VAE
        # pre-batch vae dataset
        train_dataset_vae = tf.data.Dataset.from_tensor_slices((data['train_data_vae'][0], data['train_data_vae'][1])).batch(batch_size)
        test_dataset_vae = tf.data.Dataset.from_tensor_slices((data['test_data_vae'][0], data['test_data_vae'][1])).batch(batch_size)
        vae = CCVAE(cycle_dict['latent_dim'], data['train_data_vae'][0][0].shape, data['train_data_vae'][1].shape[1], label_num)
        print('Starting VAE training.')
        vae = train_vae(vae, train_dataset_vae, test_dataset_vae, epochs=num_epochs)
        # save vae to file
        vae.train_info['file_path'] = save_model(vae.decoder, new_model_path, vae.model_name)
        # create cycle dir for generated vae images
        new_cycle_vae_imgs_path = new_cycle_path+'vae_imgs/'
        save_directory(new_cycle_vae_imgs_path)
        vae_imgs_paths = create_img_label_dirs(new_cycle_vae_imgs_path)
        # generate samples
        vae_gen_time_start = time.time()
        VAE_Generate_Mult(vae.train_info['file_path'], cycle_dict['latent_dim'], cycle_dict['num_new_samples'], vae_imgs_paths)
        vae_gen_time_end = time.time()
        vae_gen_time = vae_gen_time_end - vae_gen_time_start
        vae.train_info['vae_gen_time_mins'] = vae_gen_time/60
        vae.train_info['vae_gen_time_sec'] = vae_gen_time
        # reset everything to avoid clutter
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()

        # train classifier with vae data
        cl_train_data_vae = load_classifier_data(data['train_data_cl'], data['test_data_cl'], new_cycle_vae_imgs_path)
        vae_classifier = Classifier(cl_train_data_vae['cl_train_data'][0][0].shape, label_num, 'Classifier_VAE')
        vae_classifier = train_classifier(vae_classifier, cl_train_data_vae['cl_train_data'], cl_train_data_vae['cl_test_data'], epochs=num_epochs_cl)
        vae_classifier.train_info['file_path'] = save_model(vae_classifier.model, new_model_path, vae_classifier.model_name)
        # reset everything to avoid clutter
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()

        # training GAN
        gan = CCGAN(cycle_dict['latent_dim'], data['train_data_gan'][0][0].shape, data['train_data_gan'][1].shape[1], label_num)
        print('Starting GAN training.')
        gan = train_gan(gan, data['train_data_gan'], cycle_dict['latent_dim'], n_epochs=num_epochs, n_batch=batch_size)
        # save gan to file
        gan.train_info['file_path'] = save_model(gan.generator, new_model_path, gan.model_name)
        # create cycle dir for generated gan images
        new_cycle_gan_imgs_path = new_cycle_path+'gan_imgs/'
        save_directory(new_cycle_gan_imgs_path)
        gan_imgs_paths = create_img_label_dirs(new_cycle_gan_imgs_path)
        # generate samples
        gan_gen_time_start = time.time()
        GAN_Generate_Mult(gan.train_info['file_path'], cycle_dict['latent_dim'], cycle_dict['num_new_samples'], gan_imgs_paths)
        gan_gen_time_end = time.time()
        gan_gen_time = gan_gen_time_end - gan_gen_time_start
        gan.train_info['gan_gen_time_mins'] = gan_gen_time/60
        gan.train_info['gan_gen_time_sec'] = gan_gen_time
        # reset everything to avoid clutter
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()

        # train classifier with gan data
        cl_train_data_gan = load_classifier_data(data['train_data_cl'], data['test_data_cl'], new_cycle_gan_imgs_path)
        gan_classifier = Classifier(cl_train_data_gan['cl_train_data'][0][0].shape, label_num, 'Classifier_GAN')
        gan_classifier = train_classifier(gan_classifier, cl_train_data_gan['cl_train_data'], cl_train_data_gan['cl_test_data'], epochs=num_epochs_cl)
        gan_classifier.train_info['file_path'] = save_model(gan_classifier.model, new_model_path, gan_classifier.model_name)
        # reset everything to avoid clutter
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()

        # train classifier with aug data
        new_cycle_aug_imgs_path = new_cycle_path+'aug_imgs/'
        save_directory(new_cycle_aug_imgs_path)
        aug_imgs_paths = create_img_label_dirs(new_cycle_aug_imgs_path)
        # take num_new_samples because num_new_samples is num per label
        aug_train_time, aug_gen_time, cl_train_data_aug = load_classifier_data(data['train_data_cl'], data['test_data_cl'], number_aug=cycle_dict['num_new_samples'], imgs_paths=aug_imgs_paths)
        aug_classifier = Classifier(cl_train_data_aug['cl_train_data'][0][0].shape, label_num, 'Classifier_AUG')
        aug_classifier.train_info['aug_train_time_mins'] = aug_train_time/60
        aug_classifier.train_info['aug_train_time_sec'] = aug_train_time
        aug_classifier.train_info['aug_gen_time_mins'] = aug_gen_time/60
        aug_classifier.train_info['aug_gen_time_sec'] = aug_gen_time
        aug_classifier = train_classifier(aug_classifier, cl_train_data_aug['cl_train_data'], cl_train_data_aug['cl_test_data'], epochs=num_epochs_cl)
        aug_classifier.train_info['file_path'] = save_model(aug_classifier.model, new_model_path, aug_classifier.model_name)
        # reset everything to avoid clutter
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()

        # train classifier with only train data
        cl_train_data = load_classifier_data(data['train_data_cl'], data['test_data_cl'], gen=False)
        classifier = Classifier(cl_train_data['cl_train_data'][0][0].shape, label_num, 'Classifier')
        classifier = train_classifier(classifier, cl_train_data['cl_train_data'], cl_train_data['cl_test_data'], epochs=num_epochs_cl)
        classifier.train_info['file_path'] = save_model(classifier.model, new_model_path, classifier.model_name)
        # reset everything to avoid clutter
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        reset_seeds()

        # save cycle information to json
        models = [vae, gan, vae_classifier, gan_classifier, classifier, aug_classifier]
        save_cycle(models, new_cycle_path, cycle_dict, cycle)
        # clear tf graph
        print('Reseting graph.')
        del models
        del vae
        del gan
        del vae_classifier
        del gan_classifier
        del classifier
        del aug_classifier
        gc.collect()
        logging.info('Completed cycle {}'.format(cycle))
    except Exception as e:
        trace = traceback.format_exc()
        logging.error('An error occurred in main.py. Error {}. Trace: {}'.format(e, trace))


