import tensorflow as tf
import numpy as np
from numpy.random import randn
from numpy.random import randint

# inspired by: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

class CCGAN(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, input_cond_dim, label_num, embedding_size=50):
        super(CCGAN, self).__init__()
        # this stores the path to the generator network file
        # this stores a list of minimized losses reached during training fror each network 
        # (discriminator after real batch, discriminator after generated batch, generator)
        # this stores the training time in seconds and minutes
        self.train_info = {}
        self.model_name = 'CCGAN'
        self.train_info['latent_dim'] = latent_dim
        # define discriminator network
        discr_input = tf.keras.layers.Input(shape=input_dim)
        discr_cond_input = tf.keras.layers.Input(shape=input_cond_dim)
        # embedding for categorical input
        discr_cond_emb = tf.keras.layers.Embedding(label_num, embedding_size)(discr_cond_input)
        # scale embedding up to image dimensions with linear activation
        n_nodes = input_dim[0] * input_dim[1]
        discr_cond_emb = tf.keras.layers.Dense(n_nodes)(discr_cond_emb)
        # reshape to add additional channel
        discr_cond_emb = tf.keras.layers.Reshape((input_dim[0], input_dim[1], 1))(discr_cond_emb)
        # concat conditional as a channel
        discr_merge = tf.keras.layers.Concatenate()([discr_input, discr_cond_emb])
        # downsample
        discr_conv = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(discr_merge)
        discr_conv = tf.keras.layers.LeakyReLU(alpha=0.2)(discr_conv)
        # downsample again
        discr_conv = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(discr_conv)
        discr_conv = tf.keras.layers.LeakyReLU(alpha=0.2)(discr_conv)
        # flatten feature maps
        discr_flat = tf.keras.layers.Flatten()(discr_conv)
        # dropout
        discr_dropout = tf.keras.layers.Dropout(0.4)(discr_flat)
        # output
        discr_out = tf.keras.layers.Dense(1, activation='sigmoid')(discr_dropout)
        # define model
        self.discriminator = tf.keras.models.Model(inputs=[discr_input, discr_cond_input], outputs=discr_out, name="Discriminator")
        # compile model
        opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # define generator network
        # image generator input
        gen_input = tf.keras.layers.Input(shape=(latent_dim,))
        gen_cond_input = tf.keras.layers.Input(shape=input_cond_dim)
        # embedding for categorical input
        gen_cond_emb = tf.keras.layers.Embedding(10, 50)(gen_cond_input)
        # linear multiplication
        n_nodes = 7 * 7
        gen_cond_emb = tf.keras.layers.Dense(n_nodes)(gen_cond_emb)
        # reshape to additional channel
        gen_cond_emb = tf.keras.layers.Reshape((7, 7, 1))(gen_cond_emb)
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        gen = tf.keras.layers.Dense(n_nodes)(gen_input)
        gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
        gen = tf.keras.layers.Reshape((7, 7, 128))(gen)
        # merge image gen and label input
        gen_merge = tf.keras.layers.Concatenate()([gen, gen_cond_emb])
        gen_dense = tf.keras.layers.Dense(7*7*128)(gen_merge)
        # upsample to 14x14
        gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen_dense)
        gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
        # upsample to 28x28
        gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
        gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
        # output
        gen_out = tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        # define model
        self.generator = tf.keras.models.Model([gen_input, gen_cond_input], gen_out, name="Generator")
        # define GAN (combined model)
        # make weights in the discriminator not trainable
        self.discriminator.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = self.generator.input
        # get image output from the generator model
        gen_output = self.generator.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = self.discriminator([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        self.GAN = tf.keras.models.Model([gen_noise, gen_label], gan_output, name='GAN')
        # compile model
        opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
        self.GAN.compile(loss='binary_crossentropy', optimizer=opt)

    def delete_models(self):
        self.discriminator.reset_states()
        self.GAN.reset_states()
        self.generator.reset_states()
        del self.discriminator
        del self.GAN
        del self.generator
    
    def __del__(self):
        self.delete_models()
        print("Deleted GAN.")

    # samples from the real dataset, hier y are the labels for the discriminator
    # i.e. 1 for real images 0 for generated images
    def sample_real(self, dataset, n_samples):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        ix = randint(0, images.shape[0], n_samples)
        # select images and labels
        labels = labels[ix]
        X = images[ix]
        # generate class labels
        y = np.ones((n_samples, 1))
        return [X, labels], y

    # generate points in latent space as input for the generator
    def sample_latent_points(self, latent_dim, n_samples, n_classes=10):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        # generate random labels
        labels = randint(0, n_classes, n_samples)
        return [z_input, labels]
    
    # use the generator to generate n fake examples, with class labels
    def sample_fake(self, latent_dim, n_samples):
        # generate points in latent space
        z_input, labels_input = self.sample_latent_points(latent_dim, n_samples)
        # predict outputs
        images = self.generator.predict([z_input, labels_input])
        # create class labels
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y

