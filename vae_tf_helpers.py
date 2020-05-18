import tensorflow as tf
import numpy as np


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

# maximize ELBO  on the marginal log-likelyhood
def compute_loss(model, x, cond):
    mean, logvar = model.encode(x, cond)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z, cond)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    # optimize single sample Monte Carlo estimate of expectation
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_apply_gradients(model, x, cond, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, cond)
        elbo = -loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return elbo
optimizer = tf.keras.optimizers.Adam(1e-4)