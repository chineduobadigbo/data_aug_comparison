import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from save_run import save_directory, read_run_dict

# create and save a plot of generated images
def save_plot(examples, n, path):
	# plot images
    for i in range(n * n):
		# define subplot
        plt.subplot(n, n, 1 + i)
		# turn off axis
        plt.axis('off')
		# plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray')
    plt.savefig(path)
    plt.close()
    print('Saved {}'.format(path))



def make_collages(res, plot_dir, second_plot_dir):
    # first save vae_img collage
    vae_imgs_path = os.path.join(res, 'vae_imgs')
    vae_img_dirs = os.listdir(vae_imgs_path)
    vae_image_list = []
    for dir in vae_img_dirs:
        vae_img_dir = os.path.join(vae_imgs_path, dir)
        #print(vae_img_dir)
        vae_img_label_dirs = os.listdir(vae_img_dir)
        vae_img_label_dirs.sort()
        for i in range(10):
            img_path = os.path.join(vae_img_dir, vae_img_label_dirs[i])
            im=Image.open(img_path)
            vae_image_list.append(im.copy())
            im.close()
    vae_imgs = np.array(list(map(np.asarray, vae_image_list)))
    vae_imgs = vae_imgs.reshape((vae_imgs.shape[0], 28, 28, 1)).astype('float32')
    vae_collage_path = os.path.join(plot_dir, 'vae_collage_{}.png'.format(res.split('\\')[-1]))
    save_plot(vae_imgs, 10, vae_collage_path)
    vae_collage_path = os.path.join(second_plot_dir, 'vae_collage_{}.png'.format(res.split('\\')[-1]))
    save_plot(vae_imgs, 10, vae_collage_path)

    # then save gan_img collage
    gan_imgs_path = os.path.join(res, 'gan_imgs')
    gan_img_dirs = os.listdir(gan_imgs_path)
    gan_image_list = []
    for dir in gan_img_dirs:
        gan_img_dir = os.path.join(gan_imgs_path, dir)
        #print(gan_img_dir)
        gan_img_label_dirs = os.listdir(gan_img_dir)
        gan_img_label_dirs.sort()
        for i in range(10):
            img_path = os.path.join(gan_img_dir, gan_img_label_dirs[i])
            im=Image.open(img_path)
            gan_image_list.append(im.copy())
            im.close()
    gan_imgs = np.array(list(map(np.asarray, gan_image_list)))
    gan_imgs = gan_imgs.reshape((gan_imgs.shape[0], 28, 28, 1)).astype('float32')
    gan_collage_path = os.path.join(plot_dir, 'gan_collage_{}.png'.format(res.split('\\')[-1]))
    save_plot(gan_imgs, 10, gan_collage_path)
    gan_collage_path = os.path.join(second_plot_dir, 'gan_collage_{}.png'.format(res.split('\\')[-1]))
    save_plot(gan_imgs, 10, gan_collage_path)

    # then save aug_img collage
    aug_imgs_path = os.path.join(res, 'aug_imgs')
    aug_img_dirs = os.listdir(aug_imgs_path)
    aug_image_list = []
    for dir in aug_img_dirs:
        aug_img_dir = os.path.join(aug_imgs_path, dir)
        #print(aug_img_dir)
        aug_img_label_dirs = os.listdir(aug_img_dir)
        aug_img_label_dirs.sort()
        for i in range(10):
            img_path = os.path.join(aug_img_dir, aug_img_label_dirs[i])
            im=Image.open(img_path)
            aug_image_list.append(im.copy())
            im.close()
    aug_imgs = np.array(list(map(np.asarray, aug_image_list)))
    aug_imgs = aug_imgs.reshape((aug_imgs.shape[0], 28, 28, 1)).astype('float32')
    aug_collage_path = os.path.join(plot_dir, 'aug_collage_{}.png'.format(res.split('\\')[-1]))
    save_plot(aug_imgs, 10, aug_collage_path)
    aug_collage_path = os.path.join(second_plot_dir, 'aug_collage_{}.png'.format(res.split('\\')[-1]))
    save_plot(aug_imgs, 10, aug_collage_path)


def make_dim_comp_plots(data_dict, plot_dir, second_plot_dir):
    mnist = data_dict['mnist']
    fashion = data_dict['fashion']
    mnist_small = data_dict['mnist_small']
    fashion_small = data_dict['fashion_small']
    dim_comp_mnist_dict = {}
    dim_comp_mnist_dict[50] = data_dict['mnist']['CCVAE']['train_elbo_hist'][1]
    for i in mnist_small['CCVAE']['train_elbo_hist']:
        dim_comp_mnist_dict[i[1]] = i[0]
    first_cycle = []
    second_cycle = []
    third_cycle = []
    fourth_cycle = []
    keys = []
    dims = list(dim_comp_mnist_dict.keys())
    for cycle1, cycle2, cycle3, cycle4 in zip(dim_comp_mnist_dict[dims[0]], dim_comp_mnist_dict[dims[1]], dim_comp_mnist_dict[dims[2]], dim_comp_mnist_dict[dims[3]]):
        for key, val in cycle1.items():
            keys.append(key)
            first_cycle.append(val)
        for key, val in cycle2.items():
            second_cycle.append(val)
        for key, val in cycle3.items():
            third_cycle.append(val)
        for key, val in cycle4.items():
            fourth_cycle.append(val)
    fig, ax = plt.subplots(figsize=(15,5))
    x = keys
    ax.plot(x, first_cycle, c='0.15')
    ax.plot(x, second_cycle, c='0.35')
    ax.plot(x, third_cycle, c='0.55')
    ax.plot(x, fourth_cycle, c='0.85')
    ax.legend(['Latent Dim: {}'.format(dims[0]), 'Latent Dim: {}'.format(dims[1]), 'Latent Dim: {}'.format(dims[2]), 'Latent Dim: {}'.format(dims[3])])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ELBO')
    ax.set_title('ELBO history in relation to latent dimension. (MNIST 400 samples)')
    n = 10  # Keeps every 10th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    vae_dim_comp_mnist = os.path.join(plot_dir, 'vae_dim_comp_mnist.png')
    plt.tight_layout()
    fig.savefig(vae_dim_comp_mnist)
    vae_dim_comp_mnist = os.path.join(second_plot_dir, 'vae_dim_comp_mnist.png')
    fig.savefig(vae_dim_comp_mnist)
    plt.close('all')


    dim_comp_fashion_dict = {}
    dim_comp_fashion_dict[50] = data_dict['fashion']['CCVAE']['train_elbo_hist'][1]
    for i in fashion_small['CCVAE']['train_elbo_hist']:
        dim_comp_fashion_dict[i[1]] = i[0]
    first_cycle = []
    second_cycle = []
    third_cycle = []
    fourth_cycle = []
    keys = []
    dims = list(dim_comp_fashion_dict.keys())
    for cycle1, cycle2, cycle3, cycle4 in zip(dim_comp_fashion_dict[dims[0]], dim_comp_fashion_dict[dims[1]], dim_comp_fashion_dict[dims[2]], dim_comp_fashion_dict[dims[3]]):
        for key, val in cycle1.items():
            keys.append(key)
            first_cycle.append(val)
        for key, val in cycle2.items():
            second_cycle.append(val)
        for key, val in cycle3.items():
            third_cycle.append(val)
        for key, val in cycle4.items():
            fourth_cycle.append(val)
    fig, ax = plt.subplots(figsize=(15,5))
    x = keys
    ax.plot(x, first_cycle, c='0.15')
    ax.plot(x, second_cycle, c='0.35')
    ax.plot(x, third_cycle, c='0.55')
    ax.plot(x, fourth_cycle, c='0.85')
    ax.legend(['Latent Dim: {}'.format(dims[0]), 'Latent Dim: {}'.format(dims[1]), 'Latent Dim: {}'.format(dims[2]), 'Latent Dim: {}'.format(dims[3])])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ELBO')
    ax.set_title('ELBO history in relation to latent dimension. (Fashion-MNIST 400 samples)')
    n = 10  # Keeps every 10th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    vae_dim_comp_fashion = os.path.join(plot_dir, 'vae_dim_comp_fashion.png')
    plt.tight_layout()
    fig.savefig(vae_dim_comp_fashion)
    vae_dim_comp_fashion = os.path.join(second_plot_dir, 'vae_dim_comp_fashion.png')
    fig.savefig(vae_dim_comp_fashion)
    plt.close('all')

    # GAN
    dim_comp_mnist_dict = {}
    dim_comp_mnist_dict[50] = data_dict['mnist']['CCGAN']['g_loss_hist'][1]
    for i in mnist_small['CCGAN']['g_loss_hist']:
        dim_comp_mnist_dict[i[1]] = i[0]
    first_cycle_g = {}
    second_cycle_g = {}
    third_cycle_g = {}
    fourth_cycle_g = {}
    batch_num = list(mnist['CCGAN']['g_loss_hist'][1][1].values())[0][0]
    samples = int(batch_num.split('/')[-1])*100
    dims = list(dim_comp_mnist_dict.keys())
    for g1, g2, g3, g4 in zip(dim_comp_mnist_dict[dims[0]], dim_comp_mnist_dict[dims[1]], dim_comp_mnist_dict[dims[2]], dim_comp_mnist_dict[dims[3]]):
        for key, val in g1.items():
            first_cycle_g[key] = val[1]
        for key, val in g2.items():
            second_cycle_g[key] = val[1]
        for key, val in g3.items():
            third_cycle_g[key] = val[1]
        for key, val in g4.items():
            fourth_cycle_g[key] = val[1]

    fig, ax = plt.subplots(figsize=(15,5))
    x = list(first_cycle_g.keys())
    ax.plot(x, list(first_cycle_g.values()), c='0.15')
    ax.plot(x, list(second_cycle_g.values()), c='0.35')
    ax.plot(x, list(third_cycle_g.values()), c='0.55')
    ax.plot(x, list(fourth_cycle_g.values()), c='0.85')
    ax.legend(['Latent Dim: {}'.format(dims[0]), 'Latent Dim: {}'.format(dims[1]), 'Latent Dim: {}'.format(dims[2]), 'Latent Dim: {}'.format(dims[3])])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss values')
    ax.set_title('Generator Loss history in relation to latent dimension. (MNIST {} samples)'.format(samples))
    n = 10  # Keeps every 10th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    gan_dim_comp_mnist = os.path.join(plot_dir, 'gan_dim_comp_mnist.png')
    print(gan_dim_comp_mnist)
    plt.tight_layout()
    fig.savefig(gan_dim_comp_mnist)
    gan_dim_comp_mnist = os.path.join(second_plot_dir, 'gan_dim_comp_mnist.png')
    print(gan_dim_comp_mnist)
    fig.savefig(gan_dim_comp_mnist)
    plt.close('all')

    dim_comp_fashion_dict = {}
    dim_comp_fashion_dict[50] = data_dict['fashion']['CCGAN']['g_loss_hist'][1]
    for i in fashion_small['CCGAN']['g_loss_hist']:
        dim_comp_fashion_dict[i[1]] = i[0]
    first_cycle_g = {}
    second_cycle_g = {}
    third_cycle_g = {}
    fourth_cycle_g = {}
    batch_num = list(fashion['CCGAN']['g_loss_hist'][1][1].values())[0][0]
    samples = int(batch_num.split('/')[-1])*100
    dims = list(dim_comp_fashion_dict.keys())
    for g1, g2, g3, g4 in zip(dim_comp_fashion_dict[dims[0]], dim_comp_fashion_dict[dims[1]], dim_comp_fashion_dict[dims[2]], dim_comp_fashion_dict[dims[3]]):
        for key, val in g1.items():
            first_cycle_g[key] = val[1]
        for key, val in g2.items():
            second_cycle_g[key] = val[1]
        for key, val in g3.items():
            third_cycle_g[key] = val[1]
        for key, val in g4.items():
            fourth_cycle_g[key] = val[1]

    fig, ax = plt.subplots(figsize=(15,5))
    x = list(first_cycle_g.keys())
    ax.plot(x, list(first_cycle_g.values()), c='0.15')
    ax.plot(x, list(second_cycle_g.values()), c='0.35')
    ax.plot(x, list(third_cycle_g.values()), c='0.55')
    ax.plot(x, list(fourth_cycle_g.values()), c='0.85')
    ax.legend(['Latent Dim: {}'.format(dims[0]), 'Latent Dim: {}'.format(dims[1]), 'Latent Dim: {}'.format(dims[2]), 'Latent Dim: {}'.format(dims[3])])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss values')
    ax.set_title('Generator Loss history in relation to latent dimension. (Fashion-MNIST {} samples)'.format(samples))
    n = 10  # Keeps every 10th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    gan_dim_comp_fashion = os.path.join(plot_dir, 'gan_dim_comp_fashion.png')
    print(gan_dim_comp_fashion)
    plt.tight_layout()
    fig.savefig(gan_dim_comp_fashion)
    gan_dim_comp_fashion = os.path.join(second_plot_dir, 'gan_dim_comp_fashion.png')
    print(gan_dim_comp_fashion)
    fig.savefig(gan_dim_comp_fashion)
    plt.close('all')


def make_run_plots(data_dict, plot_dir, second_plot_dir):
    mnist = data_dict['mnist']
    fashion = data_dict['fashion']

    make_dim_comp_plots(data_dict, plot_dir, second_plot_dir)
    #sys.exit(1)
    # mnist plots
    x = mnist['Cycle']['samples']
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, mnist['Classifier_VAE']['test_accuracy'], c='0.15')
    ax.plot(x, mnist['Classifier_GAN']['test_accuracy'], c='0.35')
    ax.plot(x, mnist['Classifier_AUG']['test_accuracy'], c='0.55')
    ax.plot(x, mnist['Classifier']['test_accuracy'], c='0.85')
    ax.legend(['Classifier trained on VAE data', 'Classifier trained on GAN data', 'Classifier trained on  Traditional AUG data', 'Classifier trained only on train data'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Accuracy on Test-Set')
    ax.set_title('Accuracy on Test-Set in relation with training data size. (MNIST)')
    test_acc_mnist_path = os.path.join(plot_dir, 'test_acc_mnist.png')
    plt.tight_layout()
    fig.savefig(test_acc_mnist_path)
    test_acc_mnist_path = os.path.join(second_plot_dir, 'test_acc_mnist.png')
    fig.savefig(test_acc_mnist_path)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, mnist['CCVAE']['train_time_sec'], c='0.15')
    ax.plot(x, mnist['CCGAN']['train_time_sec'], c='0.55')
    ax.plot(x, mnist['Classifier_AUG']['aug_train_time_sec'], c='0.85')
    ax.legend(['VAE', 'GAN', 'Traditional AUG'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Time in sec')
    ax.set_title('Train times in relation with training data size. (MNIST)')
    train_time_sec_mnist_path = os.path.join(plot_dir, 'train_time_sec_mnist.png')
    plt.tight_layout()
    fig.savefig(train_time_sec_mnist_path)
    train_time_sec_mnist_path = os.path.join(second_plot_dir, 'train_time_sec_mnist.png')
    fig.savefig(train_time_sec_mnist_path)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, mnist['Classifier_VAE']['test_loss'], c='0.15')
    ax.plot(x, mnist['Classifier_GAN']['test_loss'], c='0.35')
    ax.plot(x, mnist['Classifier_AUG']['test_loss'], c='0.55')
    ax.plot(x, mnist['Classifier']['test_loss'], c='0.85')
    ax.legend(['Classifier trained on VAE data', 'Classifier trained on GAN data', 'Classifier trained on  Traditional AUG data', 'Classifier trained only on train data'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Loss on Test-Set')
    ax.set_title('Loss on Test-Set in relation with training data size. (MNIST)')
    test_loss_mnist_path = os.path.join(plot_dir, 'test_loss_mnist.png')
    plt.tight_layout()
    fig.savefig(test_loss_mnist_path)
    test_loss_mnist_path = os.path.join(second_plot_dir, 'test_loss_mnist.png')
    fig.savefig(test_loss_mnist_path)
    plt.close('all')

    # fashion-mnist plots

    x = fashion['Cycle']['samples']
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, fashion['Classifier_VAE']['test_accuracy'], c='0.15')
    ax.plot(x, fashion['Classifier_GAN']['test_accuracy'], c='0.35')
    ax.plot(x, fashion['Classifier_AUG']['test_accuracy'], c='0.55')
    ax.plot(x, fashion['Classifier']['test_accuracy'], c='0.85')
    ax.legend(['Classifier trained on VAE data', 'Classifier trained on GAN data', 'Classifier trained on  Traditional AUG data', 'Classifier trained only on train data'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Accuracy on Test-Set')
    ax.set_title('Accuracy on Test-Set in relation with training data size. (Fashion-MNIST)')
    test_acc_fashion_path = os.path.join(plot_dir, 'test_acc_fashion.png')
    plt.tight_layout()
    fig.savefig(test_acc_fashion_path)
    test_acc_fashion_path = os.path.join(second_plot_dir, 'test_acc_fashion.png')
    fig.savefig(test_acc_fashion_path)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, fashion['CCVAE']['train_time_sec'], c='0.15')
    ax.plot(x, fashion['CCGAN']['train_time_sec'], c='0.55')
    ax.plot(x, fashion['Classifier_AUG']['aug_train_time_sec'], c='0.85')
    ax.legend(['VAE', 'GAN', 'Traditional AUG'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Time in sec')
    ax.set_title('Train times in relation with training data size. (Fashion-MNIST)')
    train_time_sec_fashion_path = os.path.join(plot_dir, 'train_time_sec_fashion.png')
    plt.tight_layout()
    fig.savefig(train_time_sec_fashion_path)
    train_time_sec_fashion_path = os.path.join(second_plot_dir, 'train_time_sec_fashion.png')
    fig.savefig(train_time_sec_fashion_path)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, fashion['Classifier_VAE']['test_loss'], c='0.15')
    ax.plot(x, fashion['Classifier_GAN']['test_loss'], c='0.35')
    ax.plot(x, fashion['Classifier_AUG']['test_loss'], c='0.55')
    ax.plot(x, fashion['Classifier']['test_loss'], c='0.85')
    ax.legend(['Classifier trained on VAE data', 'Classifier trained on GAN data', 'Classifier trained on  Traditional AUG data', 'Classifier trained only on train data'])
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Loss on Test-Set')
    ax.set_title('Loss on Test-Set in relation with training data size. (Fashion-MNIST)')
    test_loss_fashion_path = os.path.join(plot_dir, 'test_loss_fashion.png')
    plt.tight_layout()
    fig.savefig(test_loss_fashion_path)
    test_loss_fashion_path = os.path.join(second_plot_dir, 'test_loss_fashion.png')
    fig.savefig(test_loss_fashion_path)
    plt.close('all')

    make_gan_loss_plots(0, data_dict, plot_dir, second_plot_dir)
    make_gan_loss_plots(4, data_dict, plot_dir, second_plot_dir)
    make_gan_loss_plots(5, data_dict, plot_dir, second_plot_dir)
    make_gan_loss_plots(6, data_dict, plot_dir, second_plot_dir)
    make_gan_loss_plots(8, data_dict, plot_dir, second_plot_dir)

    make_vae_elbo_plots([0, 4, 8], data_dict, plot_dir, second_plot_dir)




def last_chars(x):
    return(x[-2:])

def make_vae_elbo_plots(index, data_dict, plot_dir, second_plot_dir):
    mnist = data_dict['mnist']
    fashion = data_dict['fashion']
    
    # mnist
    first_cycle = []
    second_cycle = []
    third_cycle = []
    keys = []
    for cycle1, cycle2, cycle3 in zip(mnist['CCVAE']['train_elbo_hist'][index[0]], mnist['CCVAE']['train_elbo_hist'][index[1]], mnist['CCVAE']['train_elbo_hist'][index[2]]):
        for key, val in cycle1.items():
            keys.append(key)
            first_cycle.append(val)
        for key, val in cycle2.items():
            second_cycle.append(val)
        for key, val in cycle3.items():
            third_cycle.append(val)
    fig, ax = plt.subplots(figsize=(15,5))
    x = keys
    samples_list = mnist['Cycle']['samples']
    samples1 = samples_list[index[0]]
    samples2 = samples_list[index[1]]
    samples3 = samples_list[index[2]]
    ax.plot(x, first_cycle, c='0.15')
    ax.plot(x, second_cycle, c='0.45')
    ax.plot(x, third_cycle, c='0.85')
    ax.legend(['{} samples'.format(samples1), '{} samples'.format(samples2), '{} samples'.format(samples3)])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss values')
    ax.set_title('VAE ELBO history in relation with training data size. (MNIST)')
    n = 10  # Keeps every 7th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    vae_elbo_hist_mnist_path = os.path.join(plot_dir, 'vae_elbo_hist_mnist_{}_{}_{}.png'.format(index[0], index[1], index[2]))
    print(vae_elbo_hist_mnist_path)
    plt.tight_layout()
    fig.savefig(vae_elbo_hist_mnist_path)
    vae_elbo_hist_mnist_path = os.path.join(second_plot_dir, 'vae_elbo_hist_mnist_{}_{}_{}.png'.format(index[0], index[1], index[2]))
    print(vae_elbo_hist_mnist_path)
    fig.savefig(vae_elbo_hist_mnist_path)
    plt.close('all')

    # fashion
    first_cycle = []
    second_cycle = []
    third_cycle = []
    keys = []
    for cycle1, cycle2, cycle3 in zip(fashion['CCVAE']['train_elbo_hist'][index[0]], fashion['CCVAE']['train_elbo_hist'][index[1]], fashion['CCVAE']['train_elbo_hist'][index[2]]):
        for key, val in cycle1.items():
            keys.append(key)
            first_cycle.append(val)
        for key, val in cycle2.items():
            second_cycle.append(val)
        for key, val in cycle3.items():
            third_cycle.append(val)
    fig, ax = plt.subplots(figsize=(15,5))
    x = keys
    samples_list = fashion['Cycle']['samples']
    samples1 = samples_list[index[0]]
    samples2 = samples_list[index[1]]
    samples3 = samples_list[index[2]]
    ax.plot(x, first_cycle, c='0.15')
    ax.plot(x, second_cycle, c='0.45')
    ax.plot(x, third_cycle, c='0.85')
    ax.legend(['{} samples'.format(samples1), '{} samples'.format(samples2), '{} samples'.format(samples3)])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss values')
    ax.set_title('VAE ELBO history in relation with training data size. (Fashion-MNIST)')
    n = 10  # Keeps every 10th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    vae_elbo_hist_fashion_path = os.path.join(plot_dir, 'vae_elbo_hist_fashion_{}_{}_{}.png'.format(index[0], index[1], index[2]))
    print(vae_elbo_hist_fashion_path)
    plt.tight_layout()
    fig.savefig(vae_elbo_hist_fashion_path)
    vae_elbo_hist_fashion_path = os.path.join(second_plot_dir, 'vae_elbo_hist_fashion_{}_{}_{}.png'.format(index[0], index[1], index[2]))
    print(vae_elbo_hist_fashion_path)
    fig.savefig(vae_elbo_hist_fashion_path)
    plt.close('all')

def make_gan_loss_plots(index, data_dict, plot_dir, second_plot_dir):
    mnist = data_dict['mnist']
    fashion = data_dict['fashion']
    # mnist
    first_cycle_d1 = {}
    first_cycle_d2 = {}
    first_cycle_g = {}
    batch_num = list(mnist['CCGAN']['d_loss1_hist'][index][0].values())[0][0]
    samples = int(batch_num.split('/')[-1])*100
    for d1, d2, g in zip(mnist['CCGAN']['d_loss1_hist'][index], mnist['CCGAN']['d_loss2_hist'][index], mnist['CCGAN']['g_loss_hist'][index]):
        for key, val in d1.items():
            first_cycle_d1[key] = val[1]
        for key, val in d2.items():
            first_cycle_d2[key] = val[1]
        for key, val in g.items():
            first_cycle_g[key] = val[1]
    fig, ax = plt.subplots(figsize=(15,5))
    x = list(first_cycle_d1.keys())
    ax.plot(x, list(first_cycle_d1.values()), c='0.15')
    ax.plot(x, list(first_cycle_d2.values()), c='0.45')
    ax.plot(x, list(first_cycle_g.values()), c='0.85')
    ax.legend(['Discriminator loss on real samples', 'Discriminator loss on fake samples', 'Generator loss'])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss values')
    ax.set_title('GAN Losses during training with {} samples. (MNIST)'.format(samples))
    n = 10  # Keeps every 10th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    gan_loss_mnist_path = os.path.join(plot_dir, 'gan_losses_mnist_{}_samples.png'.format(samples))
    print(gan_loss_mnist_path)
    plt.tight_layout()
    fig.savefig(gan_loss_mnist_path)
    gan_loss_mnist_path = os.path.join(second_plot_dir, 'gan_losses_mnist_{}_samples.png'.format(samples))
    print(gan_loss_mnist_path)
    fig.savefig(gan_loss_mnist_path)
    plt.close('all')



    # fashion
    first_cycle_d1 = {}
    first_cycle_d2 = {}
    first_cycle_g = {}
    batch_num = list(fashion['CCGAN']['d_loss1_hist'][index][0].values())[0][0]
    samples = int(batch_num.split('/')[-1])*100
    for d1, d2, g in zip(fashion['CCGAN']['d_loss1_hist'][index], fashion['CCGAN']['d_loss2_hist'][index], fashion['CCGAN']['g_loss_hist'][index]):
        for key, val in d1.items():
            first_cycle_d1[key] = val[1]
        for key, val in d2.items():
            first_cycle_d2[key] = val[1]
        for key, val in g.items():
            first_cycle_g[key] = val[1]
    fig, ax = plt.subplots(figsize=(15,5))
    x = list(first_cycle_d1.keys())
    ax.plot(x, list(first_cycle_d1.values()), c='0.15')
    ax.plot(x, list(first_cycle_d2.values()), c='0.45')
    ax.plot(x, list(first_cycle_g.values()), c='0.85')
    ax.legend(['Discriminator loss on real samples', 'Discriminator loss on fake samples', 'Generator loss'])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss values')
    ax.set_title('GAN Losses during training with {} samples. (Fashion-MNIST)'.format(samples))
    n = 10  # Keeps every 7th label
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i+1) % n != 0]
    ax.set_xticklabels(x, rotation=45, ha="center", fontsize=12)
    gan_loss_fashion_path = os.path.join(plot_dir, 'gan_losses_fashion_{}_samples.png'.format(samples))
    print(gan_loss_fashion_path)
    plt.tight_layout()
    fig.savefig(gan_loss_fashion_path)
    gan_loss_fashion_path = os.path.join(second_plot_dir, 'gan_losses_fashion_{}_samples.png'.format(samples))
    print(gan_loss_fashion_path)
    fig.savefig(gan_loss_fashion_path)
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script creates all the necessary plots for each cycle.')
    args = parser.parse_args()
    run_dir = './runs/run_5/'
    second_plot_dir = 'C:/Users/Gavin/Documents/FH/BA_Backup/plots/'
    save_directory(second_plot_dir)
    cycles_dir = os.path.join(run_dir, 'cycles')

    filenames = os.listdir(cycles_dir) # get all files' and folders' names in the current directory

    result = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(cycles_dir), filename)): # check whether the current object is a folder or not
            result.append(os.path.join(os.path.abspath(cycles_dir), filename))
    result = sorted(result, key=last_chars)
    cycle_paths = result[-9:]
    cycle_paths.extend(result[:len(result)-9])
    for res in cycle_paths:
        #print(res)
        cycle_plot_dir = os.path.join(res, 'plots')
        save_directory(cycle_plot_dir)
        #make_collages(res, cycle_plot_dir, second_plot_dir)

    cycle_paths_full_latent_dim = cycle_paths[:9]
    cycle_paths_full_latent_dim.extend(cycle_paths[12:21])
    cycle_paths_full_latent_dim_mnist = cycle_paths_full_latent_dim[:9]
    data_dict_mnist = {}
    data_dict_mnist['Cycle'] = {'samples': []}
    data_dict_mnist['Classifier_VAE'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_mnist['Classifier_GAN'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_mnist['Classifier'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_mnist['Classifier_AUG'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}

    data_dict_mnist['CCVAE'] = {'train_time_sec': [], 'train_time_mins': [], 'vae_gen_time_mins': [], 'vae_gen_time_sec': [], 'train_elbo_hist': []}
    data_dict_mnist['CCGAN'] = {'train_time_sec': [], 'train_time_mins': [], 'gan_gen_time_mins': [], 'gan_gen_time_sec': [], 'd_loss1_hist': [], 'd_loss2_hist': [], 'g_loss_hist': []}
    data_dict_mnist['Classifier_AUG'].update({'aug_train_time_sec': [], 'aug_train_time_mins': [], 'aug_gen_time_mins': [], 'aug_gen_time_sec': []})
    # plots for cycles where laten_dim is 50 and mnist
    for res in cycle_paths_full_latent_dim_mnist:
        plot_dir = os.path.join(res, 'plots')
        save_directory(plot_dir)

        cycle_info_path = os.path.join(res, 'cycle_info.json')
        cycle_info_dict = read_run_dict(cycle_info_path)
        cycle = list(cycle_info_dict.keys())[0]

        # store classifier accuracies accross all cycles
        for cl, cl_dict in data_dict_mnist.items():
            for metr, val in cl_dict.items():
                cl_dict[metr].append(cycle_info_dict[cycle][cl][metr])

    cycle_paths_full_latent_dim_fashion = cycle_paths_full_latent_dim[-9:]
    data_dict_fashion = {}
    data_dict_fashion['Cycle'] = {'samples': []}
    data_dict_fashion['Classifier_VAE'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_fashion['Classifier_GAN'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_fashion['Classifier'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_fashion['Classifier_AUG'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}

    data_dict_fashion['CCVAE'] = {'train_time_sec': [], 'train_time_mins': [], 'vae_gen_time_mins': [], 'vae_gen_time_sec': [], 'train_elbo_hist': []}
    data_dict_fashion['CCGAN'] = {'train_time_sec': [], 'train_time_mins': [], 'gan_gen_time_mins': [], 'gan_gen_time_sec': [], 'd_loss1_hist': [], 'd_loss2_hist': [], 'g_loss_hist': []}
    data_dict_fashion['Classifier_AUG'].update({'aug_train_time_sec': [], 'aug_train_time_mins': [], 'aug_gen_time_mins': [], 'aug_gen_time_sec': []})
    # plots for cycles where laten_dim is 50 and fashion
    for res in cycle_paths_full_latent_dim_fashion:
        plot_dir = os.path.join(res, 'plots')
        save_directory(plot_dir)

        cycle_info_path = os.path.join(res, 'cycle_info.json')
        cycle_info_dict = read_run_dict(cycle_info_path)
        cycle = list(cycle_info_dict.keys())[0]

        # store classifier accuracies accross all cycles
        for cl, cl_dict in data_dict_fashion.items():
            for metr, val in cl_dict.items():
                cl_dict[metr].append(cycle_info_dict[cycle][cl][metr])


    cycle_paths_small_latent_dim = cycle_paths[9:12]
    cycle_paths_small_latent_dim.extend(cycle_paths[21:])
    cycle_paths_small_latent_dim_mnist = cycle_paths_small_latent_dim[:3]
    data_dict_small_mnist = {}
    data_dict_small_mnist['Cycle'] = {'samples': []}
    data_dict_small_mnist['Classifier_VAE'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_mnist['Classifier_GAN'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_mnist['Classifier'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_mnist['Classifier_AUG'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_mnist['CCVAE'] = {'train_time_sec': [], 'train_time_mins': [], 'vae_gen_time_mins': [], 'vae_gen_time_sec': [], 'train_elbo_hist': []}
    data_dict_small_mnist['CCGAN'] = {'train_time_sec': [], 'train_time_mins': [], 'gan_gen_time_mins': [], 'gan_gen_time_sec': [], 'd_loss1_hist': [], 'd_loss2_hist': [], 'g_loss_hist': []}
    data_dict_small_mnist['Classifier_AUG'].update({'aug_train_time_sec': [], 'aug_train_time_mins': [], 'aug_gen_time_mins': [], 'aug_gen_time_sec': []})
    # plots for cycles where laten_dim is < 50 and mnist
    for res in cycle_paths_small_latent_dim_mnist:
        plot_dir = os.path.join(res, 'plots')
        save_directory(plot_dir)

        cycle_info_path = os.path.join(res, 'cycle_info.json')
        cycle_info_dict = read_run_dict(cycle_info_path)
        cycle = list(cycle_info_dict.keys())[0]

        # store classifier accuracies accross all cycles
        for cl, cl_dict in data_dict_small_mnist.items():
            for metr, val in cl_dict.items():
                cl_dict[metr].append((cycle_info_dict[cycle][cl][metr], cycle_info_dict[cycle]['Cycle']['latent_dim']))

    cycle_paths_small_latent_dim_fashion = cycle_paths_small_latent_dim[-3:]
    data_dict_small_fashion = {}
    data_dict_small_fashion['Cycle'] = {'samples': []}
    data_dict_small_fashion['Classifier_VAE'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_fashion['Classifier_GAN'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_fashion['Classifier'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_fashion['Classifier_AUG'] = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': []}
    data_dict_small_fashion['CCVAE'] = {'train_time_sec': [], 'train_time_mins': [], 'vae_gen_time_mins': [], 'vae_gen_time_sec': [], 'train_elbo_hist': []}
    data_dict_small_fashion['CCGAN'] = {'train_time_sec': [], 'train_time_mins': [], 'gan_gen_time_mins': [], 'gan_gen_time_sec': [], 'd_loss1_hist': [], 'd_loss2_hist': [], 'g_loss_hist': []}
    data_dict_small_fashion['Classifier_AUG'].update({'aug_train_time_sec': [], 'aug_train_time_mins': [], 'aug_gen_time_mins': [], 'aug_gen_time_sec': []})
    # plots for cycles where laten_dim is < 50 and fashion
    for res in cycle_paths_small_latent_dim_fashion:
        plot_dir = os.path.join(res, 'plots')
        save_directory(plot_dir)

        cycle_info_path = os.path.join(res, 'cycle_info.json')
        cycle_info_dict = read_run_dict(cycle_info_path)
        cycle = list(cycle_info_dict.keys())[0]

        # store classifier accuracies accross all cycles
        for cl, cl_dict in data_dict_small_fashion.items():
            for metr, val in cl_dict.items():
                cl_dict[metr].append((cycle_info_dict[cycle][cl][metr], cycle_info_dict[cycle]['Cycle']['latent_dim']))

    run_plot_dir = os.path.join(run_dir, 'plots')
    save_directory(run_plot_dir)
    data_dict = {}
    data_dict['mnist'] = data_dict_mnist
    data_dict['fashion'] = data_dict_fashion
    data_dict['mnist_small'] = data_dict_small_mnist
    data_dict['fashion_small'] = data_dict_small_fashion
    make_run_plots(data_dict, run_plot_dir, second_plot_dir)



