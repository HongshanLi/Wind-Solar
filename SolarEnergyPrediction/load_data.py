from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

def read_data_from_csv(filename_queue):
    num_features = 15
    label_size = 1
    data_size = num_features + label_size
    
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[0] for i in xrange(data_size)]
    record = tf.decode_csv(value, record_defaults=record_defaults)
    
    features = tf.slice(record, [0], [num_features])
    label = tf.slice(record, [num_features], [label_size])
    return features, label

def make_batch(features, label, min_queue_examples, batch_size):
    num_threads = 16
    input_batch, label_batch = tf.train.shuffle_batch(
        [features, label], batch_size=batch_size, num_threads=num_threads,
        capacity=min_queue_examples + 2*batch_size,
        min_after_dequeue = min_queue_examples)
    return input_batch, label_batch


    
