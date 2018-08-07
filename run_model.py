import logging
import numpy as np
import scipy.stats as stats
import tensorflow as tf

from data.mnist_loader import MNISTLoader
from data.utils import train_val_test_split
from models.mnist_classifier import MNISTClassifier

def batch_evaluation(sess, model, ops, x, y=None, stream=False):
    fd = {model.image: x}
    if not y is None:
        fd[model.label] = y

    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='stream_metrics')
    sess.run(tf.variables_initializer(var_list=running_vars))
    sess.run(model.eval_dataset_init_op, fd)

    result = []
    while True:
        try:
            result.append(sess.run(ops, {model.is_train: False}))
        except tf.errors.OutOfRangeError:
            break
    
    if stream:
        return result[-1]
    return result

def train(sess, model, x_train, y_train, x_valid, y_valid, params):
    valid_stats = {'acc': 0.0, 'epoch': 1}

    for epoch in range(1, params.epochs + 1):
        sess.run(model.train_dataset_init_op, {model.image: x_train, model.label: y_train})
        train_loss = []
        while True:
            try:
                _, loss = sess.run(model.optimize, {model.is_train: True})
                train_loss.append(loss)
            except tf.errors.OutOfRangeError:
                break

        metrics = batch_evaluation(sess, model, model.metrics,
            x_valid, y=y_valid, stream=True)
        
        print('valid [{} / {}] => train_loss: {:5f}, accuracy: {:5f}'.format(epoch, params.epochs, np.mean(train_loss), metrics['acc']))
        logging.info('valid [{} / {}] => train_loss: {:5f}, accuracy: {:5f}'.format(epoch, params.epochs, np.mean(train_loss), metrics['acc']))

        if metrics['acc'] > valid_stats['acc']:
            valid_stats['acc'] = metrics['acc']
            valid_stats['epoch'] = epoch
            model.save_model(sess, epoch)

        if epoch - valid_stats['epoch'] >= params.patience:
            print('Early stopping epoch: {}'.format(epoch), '\n')
            logging.info('Early stopping epoch: {}'.format(epoch) + '\n')
            break
        
    return valid_stats

def evaluate(sess, model, x_test, y_test, params):
    test_stats = {'acc': 0.0}

    model.restore_model(sess)

    metrics = batch_evaluation(sess, model, model.metrics, x_test, y=y_test, stream=True)
    test_stats['acc'] = metrics['acc']

    print('test => accuracy: {:5f}'.format(test_stats['acc']))
    logging.info('test => accuracy: {:5f}'.format(test_stats['acc']))

    return test_stats

def run(params):
    logging.info(vars(params))

    mnist = MNISTLoader(params.dataset)
    logging.debug('mnist data loaded')

    x_train, y_train = mnist.train_images, mnist.train_labels
    x_train, y_train, x_valid, y_valid, _, _ = train_val_test_split(x_train, y_train, params.val, 0.0, rand_seed=params.random_seed)
    x_test, y_test = mnist.test_images, mnist.test_labels

    image = tf.placeholder(tf.float32, [None, mnist.rows * mnist.cols])
    label = tf.placeholder(tf.int32, [None, 1])
    model = MNISTClassifier(image, label, mnist.num_classes, vars(params))
    logging.debug('mnist model created')
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model.create_saver()
        sess.run(init)

        logging.debug('running train/val/test')

        valid_results = train(sess, model, x_train, y_train, x_valid, y_valid, params)
        logging.info(valid_results)
        test_results = evaluate(sess, model, x_test, y_test, params)
        logging.info(test_results)
