from functools import partial
import os
import sys
import tensorflow as tf

from .model import Model
from .utils import _conv, define_scope, _fully_connected, _relu, _softmax

class MNISTClassifier(Model):

    def __init__(self, image, label, num_classes, config):
        self.image = image
        self.label = label
        self.num_classes = num_classes
        self.is_train = tf.placeholder(tf.bool)
        self.config = self.get_config(config)
        self.saver = None
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.1, epsilon=1e-5, fused=True, center=True, scale=False)
        self.input_pipeline
        self.prediction
        self.optimize
        self.metrics

    def create_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def save_model(self, sess, step):
        self.saver.save(sess, os.path.join(self.config.save_dir, 'model.ckpt'), global_step=step)

    def restore_model(self, sess):
        checkpoint = tf.train.latest_checkpoint(self.config.save_dir)
        if checkpoint is None:
            sys.exit('Cannot restore model that does not exist')
        self.saver.restore(sess, checkpoint)

    @define_scope
    def input_pipeline(self):
        with tf.device('/cpu:0'):
            train_dataset = tf.data.Dataset.from_tensor_slices((self.image, self.label)) \
                            .shuffle(tf.shape(self.image, out_type=tf.int64)[0]) \
                            .batch(self.config.batch_size) \
                            .prefetch(1)

            eval_dataset = tf.data.Dataset.from_tensor_slices((self.image, self.label)) \
                           .batch(self.config.batch_size) \
                           .prefetch(1)

            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            self.x, self.y = iterator.get_next()
            self.train_dataset_init_op = iterator.make_initializer(train_dataset)
            self.eval_dataset_init_op = iterator.make_initializer(eval_dataset)

        return self.x, self.y

    @define_scope
    def prediction(self):
        x = tf.placeholder_with_default(self.x, self.image.get_shape().as_list(), name='input_data')
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 2), training=self.is_train))
        x = _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 16, 2), training=self.is_train))
        x = tf.contrib.layers.flatten(x)
        x = _fully_connected('fc1', x, self.num_classes)
        return x

    @define_scope
    def optimize(self):
        self.input_pipeline
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, 
            labels=tf.one_hot(self.y, self.num_classes)))
        weight_decay_loss = tf.get_collection('weight_decay')
        if len(weight_decay_loss) > 0:
            loss += weight_decay_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        return train_op, loss

    @define_scope(scope='stream_metrics')
    def metrics(self):
        self.input_pipeline
        predictions = _softmax(self.prediction)
        acc, update_acc = tf.metrics.accuracy(self.y, tf.argmax(predictions, axis=1))
        return {
            'acc': update_acc
        }
        