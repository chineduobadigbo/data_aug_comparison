import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# inspired by: https://www.tensorflow.org/tutorials/generative/cvae

class CCVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, input_cond_dim, label_num, embedding_size=50):
        super(CCVAE, self).__init__()
        # this stores the path to the decoder network file
        # this stores the maximized elbo reached during training
        # this stores the training time in seconds and minutes
        self.train_info = {}
        self.model_name = 'CCVAE'
        #size of the latent vector
        self.latent_dim = latent_dim
        self.train_info['latent_dim'] = self.latent_dim
        #define encoder network
        enc_input = tf.keras.layers.Input(shape=input_dim)
        enc_cond_input = tf.keras.layers.Input(shape=input_cond_dim)
        enc_emb = tf.keras.layers.Embedding(label_num, embedding_size)(enc_cond_input)
        enc_emb = tf.keras.layers.Dense(input_dim[0]*input_dim[1])(enc_emb)
        enc_emb = tf.keras.layers.Reshape((input_dim[0],input_dim[1],1))(enc_emb)
        enc_merge = tf.keras.layers.concatenate([enc_input, enc_emb])
        #enc_emb = tf.keras.layers.Flatten()(enc_emb)
        enc_conv_one = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu')(enc_merge)
        enc_conv_two = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu')(enc_conv_one)
        enc_flat = tf.keras.layers.Flatten()(enc_conv_two)
        #maybe change to leaky relu
        enc_out = tf.keras.layers.Dense(latent_dim+latent_dim)(enc_flat)

        self.encoder = tf.keras.models.Model(inputs=[enc_input, enc_cond_input], outputs=enc_out, name="Encoder")

        #define decoder network (reversed encoder)
        dec_input = tf.keras.layers.Input(shape=(latent_dim,))
        dec_cond_input = tf.keras.layers.Input(shape=input_cond_dim)
        dec_emb = tf.keras.layers.Embedding(label_num, embedding_size)(dec_cond_input)
        dec_emb = tf.keras.layers.Dense(7*7)(dec_emb)
        dec_emb = tf.keras.layers.Reshape((7,7,1))(dec_emb)
        dec = tf.keras.layers.Dense(32*7*7)(dec_input)
        dec = tf.keras.layers.LeakyReLU(alpha=0.2)(dec)
        dec = tf.keras.layers.Reshape((7, 7, 32))(dec)
        dec_merge = tf.keras.layers.concatenate([dec, dec_emb])
        dec_dense = tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu)(dec_merge)
        #dec_reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(dec_dense)
        dec_conv = tf.keras.layers.Conv2DTranspose(filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME")(dec_dense)
        dec_conv = tf.keras.layers.LeakyReLU(alpha=0.2)(dec_conv)
        dec_conv = tf.keras.layers.Conv2DTranspose(filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME")(dec_conv)
        dec_conv = tf.keras.layers.LeakyReLU(alpha=0.2)(dec_conv)
        dec_out = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")(dec_conv)
        #dec_out = tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')(dec_conv)
        self.decoder = tf.keras.models.Model(inputs=[dec_input, dec_cond_input], outputs=dec_out, name="Decoder")

        
    def delete_models(self):
        self.decoder.reset_states()
        self.encoder.reset_states()
        del self.decoder
        del self.encoder
    
    def __del__(self):
        self.delete_models()
        print("Deleted VAE.")

    def sample(self, cond, num, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(num, self.latent_dim))
            print(eps)
        return self.decode(eps, cond, apply_sigmoid=True)

    def encode(self, x, cond):
        mean, logvar = tf.split(self.encoder([x, cond]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, cond, apply_sigmoid=False):
        logits = self.decoder([z, cond])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
