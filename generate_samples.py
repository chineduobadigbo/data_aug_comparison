import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

#region Mixed
def generate_labels(n_samples, cl=0.):
    label = [cl]
    labels = np.zeros((n_samples,len(label)))
    for i in range(labels.shape[0]):
        labels[i] = np.array(label)
    return labels

# create and save a plot of generated images
def save_plot(examples, model_type, imgs_path, cl):
    for i, ex in enumerate(examples):
        ex_arr = ex * 255
        ex_arr = ex_arr.reshape(ex_arr.shape[0], ex_arr.shape[1]).astype(np.uint8)
        im = Image.fromarray(ex_arr)
        path = imgs_path+'{}.png'.format(i)
        im.save(path)
#endregion

#region GAN
# generate points in latent space as input for the generator
def generate_latent_points_gan(latent_dim, n_samples, cl=0., n_classes=10):
	# generate points in the latent space
    x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
    labels = generate_labels(n_samples, cl)
	#labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

def GAN_Generate(model_file_name, latent_dim, num_examples_to_generate, imgs_path, cl=0.):
    batch_size = 100
    # load model
    model = load_model(model_file_name, compile=False)
    # generate input
    latent_points, labels = generate_latent_points_gan(latent_dim, num_examples_to_generate, cl=cl)
    # split into batches
    points = tf.data.Dataset.from_tensor_slices((latent_points, labels)).batch(batch_size)
    # generate images
    X = []
    for point, ls in points:
        X.append(model.predict([point, ls]))
    X = tf.convert_to_tensor(X)
    X = np.array(X)
    X = X.reshape((num_examples_to_generate, X.shape[2], X.shape[3], X.shape[4]))
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot the result
    save_plot(X, 'GAN', imgs_path, int(cl))

# generates a number of samples for each label
def GAN_Generate_Mult(model_file_name, latent_dim, num_examples, imgs_paths):
    for i in range(10):
        GAN_Generate(model_file_name, latent_dim, num_examples, imgs_paths[i], float(i))
#endregion

#region VAE
def generate_latent_points_vae(latent_dim, n_samples, cl=0., n_classes=10):
    eps = tf.random.normal(shape=(n_samples, latent_dim))
    labels = generate_labels(n_samples, cl)
    return [eps, labels]

def decode(model, z, cond, apply_sigmoid=False):
    logits = model([z, cond])
    if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
    return logits
    
def VAE_Generate(model_file_name, latent_dim, num_examples_to_generate, imgs_path, cl=0.):
    batch_size = 100
    # load model
    model = load_model(model_file_name, compile=False)
    # generate input
    latent_points, labels = generate_latent_points_vae(latent_dim, num_examples_to_generate, cl=cl)
    # split into batches
    points = tf.data.Dataset.from_tensor_slices((latent_points, labels)).batch(batch_size)
    # generate images
    X = []
    for point, ls in points:
        X.append(decode(model, point, ls, apply_sigmoid=True))
    X = tf.convert_to_tensor(X)
    X = np.array(X)
    X = X.reshape((num_examples_to_generate, X.shape[2], X.shape[3], X.shape[4]))
    # plot the result
    save_plot(X, 'VAE', imgs_path, int(cl))

# generates a number of samples for each label
def VAE_Generate_Mult(model_file_name, latent_dim, num_examples, imgs_paths):
    for i in range(10):
        VAE_Generate(model_file_name, latent_dim, num_examples, imgs_paths[i], float(i))
#endregion


if __name__ == "__main__":
    model_path = './first_models/ccgan_generator.h5'
    GAN_Generate(model_path, 100, 10, './first_models/gan_samples/', cl=2.)
