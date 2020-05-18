import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from decimal import Decimal, getcontext
import logging
from load_logger import load_logger_conf
from PIL import Image
import glob
import os
import sys
import time
from generate_samples import save_plot
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 12345

#loads n number of samples form mnist with 80/20 split
def load_mist(n, fashion=False):
    if fashion:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate((train_images, test_images))
    y = np.concatenate((train_labels, test_labels))
    x = x.reshape((x.shape[0], 28, 28, 1)).astype('float32')
    y = y.reshape(y.shape[0], 1).astype('float32')

    gen_train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')
    gen_train_labels = train_labels.reshape(train_labels.shape[0], 1).astype('float32')
    gen_test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32')
    gen_test_labels = test_labels.reshape(test_labels.shape[0], 1).astype('float32')
    # gan uses only training set vae also uses test set
    # train_size = 0.8
    # #use decimal since 1-0.8 is 0.199999999 due to float inprecision
    # getcontext().prec = 3
    # test_size = float(Decimal(1)-Decimal(train_size))
    if n < gen_train_images.shape[0]:
        gen_train_images, throwaway_images, gen_train_labels, throwaway_labels = train_test_split(gen_train_images, gen_train_labels, train_size=n, random_state=SEED, shuffle=True, stratify=gen_train_labels)
        # fetch only n elements so that len(train_images)+len(test_images) = n
        # train_n = int((n/100)*(train_size*100))
        # test_n = int((n/100.)*(test_size*100))
        gen_train_images = gen_train_images[:n]
        gen_train_labels = gen_train_labels[:n]

    # gen_test_images = gen_test_images[:test_n]
    # gen_test_labels = gen_test_labels[:test_n]
    # test_labels_onehot = np.zeros((test_labels.size, test_labels.max()+1))
    # test_labels_onehot[np.arange(test_labels.size), test_labels] = 1
    logging.info('Created train test split of {} train elements and {} test elements'.format(gen_train_images.shape[0], gen_test_images.shape[0]))

    # Normalization
    # for GAN
    gen_train_images_gan = (gen_train_images - 127.5) / 127.5
    gen_test_images_gan = (gen_test_images - 127.5) / 127.5

    # for VAE
    gen_train_images_vae = gen_train_images / 255.
    gen_test_images_vae = gen_test_images / 255.
    # for Classifier
    train_images_cl = gen_train_images / 255.
    test_images_cl = gen_test_images / 255.

    dataset = {}
    dataset['train_data_gan'] = [gen_train_images_gan, gen_train_labels]
    dataset['test_data_gan'] = [gen_test_images_gan, gen_test_labels]

    dataset['train_data_vae'] = [gen_train_images_vae, gen_train_labels]
    dataset['test_data_vae'] = [gen_test_images_vae, gen_test_labels]

    dataset['train_data_cl'] = [train_images_cl, onehot_encode(gen_train_labels)]
    dataset['test_data_cl'] = [test_images_cl, onehot_encode(gen_test_labels)]
    return dataset

def load_generated_imgs(img_path):
    image_list = []
    y = []
    for i in range(10):
        for filename in glob.glob(img_path+'label_{}/*.png'.format(i)):
            im=Image.open(filename)
            image_list.append(im.copy())
            im.close()
            y.append(i)
    x = np.array(list(map(np.asarray, image_list)))
    y = np.array(y)
    x = x.reshape((x.shape[0], 28, 28, 1)).astype('float32')
    y = y.reshape(y.shape[0], 1).astype('float32')
    # x, y = shuffle(x, y, random_state=SEED)
    # Normalization 
    x /= 255.
    dataset = {}
    dataset['cl_train_data'] = [x, onehot_encode(y)]
    return dataset

def onehot_encode(labels, num_classes=10):
    labels = labels.astype(np.uint8)
    targets = np.array(labels).reshape(-1)
    labels_onehot = np.eye(num_classes)[targets]
    return labels_onehot

def load_classifier_data(train_dataset, test_dataset, img_path=None, gen=True, number_aug=None, imgs_paths=None):
    dataset = {}
    if gen:
        if img_path == None:
            # augmented data
            train_time, gen_time, augmented_data = mnist_augmented(train_dataset[0], train_dataset[1], number_aug, imgs_paths)
            generated_data = augmented_data
        else:
            # generated data
            generated_data = load_generated_imgs(img_path)
        x_train = np.concatenate((generated_data['cl_train_data'][0], train_dataset[0]))
        y_train = np.concatenate((generated_data['cl_train_data'][1], train_dataset[1]))
        x_test = test_dataset[0]
        y_test = test_dataset[1]
        dataset['cl_train_data'] = shuffle(x_train, y_train, random_state=SEED)
        dataset['cl_test_data'] = [x_test, y_test]
    else:
        x_train = train_dataset[0]
        y_train = train_dataset[1]
        x_test = test_dataset[0]
        y_test = test_dataset[1]
        dataset['cl_train_data'] = shuffle(x_train, y_train, random_state=SEED)
        dataset['cl_test_data'] = [x_test, y_test]
    if img_path == None and gen == True:
        return train_time, gen_time, dataset
    return dataset

def mnist_augmented(x_train, y_train, augment_size, imgs_paths):
    image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False)
    # go through labels and generate for each label individually to preserve balance
    result_x = np.empty((0, 28, 28, 1))
    result_y = np.empty((0, 10))
    train_time = 0
    gen_time = 0
    for i in range(10):
        l = float(i)
        # print(l)
        lab = y_train
        # turn onehot encoded labels into numeric
        lab_numeric = np.array([np.where(r==1.)[0][0] for r in lab])
        ind = np.where(lab_numeric==l)
        print(len(ind[0]))
        sliced_x = np.copy(x_train[ind, ])
        sliced_y = np.copy(y_train[ind, ])
        sliced_x = sliced_x.reshape((sliced_x.shape[1], 28, 28, 1)).astype('float32')
        sliced_y = sliced_y.reshape((sliced_y.shape[1], 10)).astype('float32')
        # print('Sliced_x shape: {}'.format(sliced_x.shape))
        # print('Sliced_y shape: {}'.format(sliced_y.shape))
        # fit data for zca whitening
        train_time_start = time.time()
        image_generator.fit(sliced_x, augment=True, seed=SEED)
        train_time_end = time.time()
        train_time += (train_time_end-train_time_start)
        # get transformed images
        gen_time_start = time.time()
        randidx = np.random.randint(sliced_x.shape[0], size=augment_size)
        x_augmented = sliced_x[randidx].copy()
        y_augmented = sliced_y[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
        imgs_path = imgs_paths[i]
        save_plot(x_augmented, 'AUG', imgs_path, i)
        gen_time_end = time.time()
        gen_time += (gen_time_end-gen_time_start)
        # print('x_augment shape: {}'.format(x_augmented.shape))
        # print('y_augment shape: {}'.format(y_augmented.shape))
        #display_image_from_array(x_augmented[0])
        result_x = np.concatenate((result_x, x_augmented))
        result_y = np.concatenate((result_y, y_augmented))
        # print('Result_x array shape: {}'.format(result_x.shape))
        # print('Result_y array shape: {}'.format(result_y.shape))
    dataset = {}
    dataset['cl_train_data'] = [result_x, result_y]
    return train_time, gen_time, dataset

def display_image_from_array(arr):
    img = Image.fromarray((arr*255).reshape((28,28)).astype(np.uint8))
    img.show()

if __name__ == "__main__":
    # img_path = './runs/run_1/cycles/cycle_1/vae_imgs/'
    # load_generated_imgs(img_path)
    #mnist_augmented()
    load_mist(25600, fashion=False)
    