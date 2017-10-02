from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import load_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("batch_size", 100,
    """Number of inputs per batch""")
tf.app.flags.DEFINE_string("data_dir", "/homes/li108/Dataset/train",
    """Full path to the data directory""")

def _variable_on_cpu(name, shape, initializer):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, 
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weigt_losses")
        tf.add_to_collection("losses", weight_decay)
    return var

def 

def neural_net(input_batch):
    # Layer 1
    with tf.variable_scope("layer1") as scope:
        weights = _variable_with_weight_decay("weights", shape=[15, 15],
            stddev=5e-2, wd=0.004)
        biases = _variable_with_weight_decay("biases", shape=[15],
            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.matmul(input_batch, weights) + biases
        layer1 = tf.sigmoid(pre_activation, name=scope.name)
        _activation_summary(layer1)

    # Layer 2
    with tf.variable_scope("layer2") as scope:
        weights = _variable_with_weight_decay("weights", shape=[15, 15],
            stddev=5e-2, wd=0.004)
        biases = _variable_with_weight_decay("biases", shape=[15],
            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.matmul(layer1, weights) + biases
        layer1 = tf.sigmoid(pre_activation, name=scope.name)
        _activation_summary(layer2)

    # Layer 3 
    with tf.variable_scope("layer2") as scope:
        weights = _variable_with_weight_decay("weights", shape=[15, 15],
            stddev=5e-2, wd=0.004)
        biases = _variable_with_weight_decay("biases", shape=[15],
            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.matmul(layer2, weights) + biases
        layer1 = tf.sigmoid(pre_activation, name=scope.name)
        _activation_summary(layer3)

    # Output layer
    with tf.variable_scope("output") as scope:
        weights = _variable_with_weight_decay("weights", shape=[15, 1],
            stddev=5e-2, wd=None)
        biases = _variable_with_weight_decay("biases", shape=[1],
            initializer=tf.constant_initializer(0.0))
        output = tf.add(tf.matmul(layer3, weights), biases, name=scope.name)
        _activation_summary(output)
        
    return output

def loss(predictions, labels):
    msl = tf.losses.mean_squared_error(labels=labels, predictions=predictions,
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
    tf.add_to_collection("losses", msl)
    total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    return total_loss


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="ema")
    losses = tf.get_collection("losses")
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = TRAIN_SET_SIZE / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCH_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_step,
        LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar("learning_rate", lr)

    # Generate moving averages of all loesses and associated summaries
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # add histogram for gradients
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    
    # tracking the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
        global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name="train")
    return train_op

