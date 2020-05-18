from CCVAE import CCVAE
from CCGAN import CCGAN
from Classifier import Classifier
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script runs through the whole training process.')

    latent_dim = 50

    vae = CCVAE(latent_dim, (28,28,1), (1,), 10)

    gan = CCGAN(latent_dim, (28,28,1), (1,), 10)

    cl = Classifier((28,28,1), 10, 'Classifier')

    print(vae.encoder.summary())

    print(gan.discriminator.summary())

    print(vae.decoder.summary())

    print(gan.generator.summary())

    print(gan.GAN.summary())

    print(cl.model.summary())
    