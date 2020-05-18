import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from vae_tf_helpers import compute_apply_gradients, compute_loss, optimizer

def train_vae(model, train_dataset, test_dataset, epochs=200):    
    start_train_time = time.time()
    train_elbos = []
    test_elbos = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x, train_y in train_dataset:
            train_elbo = compute_apply_gradients(model, train_x, train_y, optimizer)
        end_time = time.time()
        train_elbos.append({epoch: float(train_elbo)})
        print('Epoch: {}, Train set ELBO: {}, time elapse for current epoch {}s'.format(epoch, train_elbo, end_time - start_time))
        if epoch % 100 == 0:
            start_test_time = time.time()
            test_loss = tf.keras.metrics.Mean()
            for test_x, test_y in test_dataset:
                test_loss(compute_loss(model, test_x, test_y))
            test_elbo = -test_loss.result()
            test_elbos.append({epoch: float(test_elbo)})
            end_test_time = time.time()
            print('Epoch: {}, Test set ELBO: {}, time elapse for testing {}s'.format(epoch, test_elbo, end_test_time - start_test_time))
    end_train_time = end_time
    print('Total training time: {} mins or {}s'.format((end_train_time-start_train_time)/60, end_train_time-start_train_time))
    model.train_info['test_elbo'] = float(test_elbo)
    model.train_info['train_elbo'] = float(train_elbo)
    model.train_info['test_elbo_hist'] = test_elbos
    model.train_info['train_elbo_hist'] = train_elbos
    model.train_info['train_time_sec'] =  (end_train_time-start_train_time)
    model.train_info['train_time_mins'] =  ((end_train_time-start_train_time)/60)
    model.train_info['test_time_sec'] =  (end_test_time-start_test_time)
    model.train_info['test_time_mins'] =  ((end_test_time-start_test_time)/60)
    return model