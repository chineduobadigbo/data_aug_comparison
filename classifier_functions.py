import tensorflow as tf
import time
import logging

def train_classifier(model, train_data, test_data, epochs):
    start_train_time = time.time()
    model.model.fit(train_data[0], train_data[1], epochs=epochs)
    train_loss, train_acc = model.model.evaluate(train_data[0],  train_data[1])
    test_loss, test_acc = model.model.evaluate(test_data[0],  test_data[1])
    end_train_time = time.time()
    model.train_info['test_loss'] = float(test_loss)
    model.train_info['test_accuracy'] = float(test_acc)
    model.train_info['train_loss'] = float(train_loss)
    model.train_info['train_accuracy'] = float(train_acc)
    model.train_info['train_time_sec'] =  (end_train_time-start_train_time)
    model.train_info['train_time_mins'] =  ((end_train_time-start_train_time)/60)
    logging.info('Classifier done with train_loss: {} and train_accuracy: {} | test_loss: {} and test_accuracy: {}'.format(train_loss, train_acc, test_loss, test_acc))
    return model