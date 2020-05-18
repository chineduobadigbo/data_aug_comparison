import numpy as np
import time

# train the generator and discriminator
def train_gan(gan_model, dataset, latent_dim, n_epochs=200, n_batch=100):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    start_train_time = time.time()
    d_loss1_hist = []
    d_loss2_hist = []
    g_loss_hist = []
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = gan_model.sample_real(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = gan_model.discriminator.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = gan_model.sample_fake(latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = gan_model.discriminator.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = gan_model.sample_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.GAN.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            d_loss1_hist.append({i+1: ('{}/{}'.format(j+1, bat_per_epo),float(d_loss1))})
            d_loss2_hist.append({i+1: ('{}/{}'.format(j+1, bat_per_epo),float(d_loss2))})
            g_loss_hist.append({i+1: ('{}/{}'.format(j+1, bat_per_epo),float(g_loss))})
    end_train_time = time.time()
    print('Total training time: {} mins or {}s'.format((end_train_time-start_train_time)/60, end_train_time-start_train_time))
    gan_model.train_info['d_loss1'] = float(d_loss1)
    gan_model.train_info['d_loss2'] = float(d_loss2)
    gan_model.train_info['g_loss'] = float(g_loss)
    gan_model.train_info['d_loss1_hist'] = d_loss1_hist
    gan_model.train_info['d_loss2_hist'] = d_loss2_hist
    gan_model.train_info['g_loss_hist'] = g_loss_hist
    gan_model.train_info['train_time_sec'] =  (end_train_time-start_train_time)
    gan_model.train_info['train_time_mins'] =  ((end_train_time-start_train_time)/60)
    return gan_model