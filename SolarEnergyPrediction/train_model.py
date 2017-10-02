from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import main_functions
import load_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 
                           '/home/hongshan/MachineLearning/Wind-Solar/SolarEnergyPrediction/train_event_log',
                          '''Directory where to write event logs and checkpoint''')
tf.app.flags.DEFINE_integer('max_steps', 5000, '''Number of batches to run''')
tf.app.flags.DEFINE_boolean('log_device_placement', False, '''Whether to log device placement''')

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        # Define the filename_queue here
        filenames = []
        filenames_queue = tf.train.string_input_producer(filenames)
        
         
        #Get images and labels for training
        features, label = load_data.read_data_from_csv(filename_queue)
       
        input_batch, label_batch = load_data.make_batch(features,
            label, min_queue_examples=1000, batch_size=100)
        
        
        # Build a graph that computes predicted energy
        predicted_energy = main_functions.neural_net(input_batch)

        
        # Calculate loss
        loss = main_functions.loss(input_batch, label_batch)
        
        #Build a graph that trains the model with one batch of data
        # and updates the model parameters
        train_op = main_functions.train(loss, global_step)
        
        class _LoggerHook(tf.train.SessionRunHook):
            """This class logs loss and runtime"""
            def begin(self):
                self._step = -1
                
            def before_run(self, run_context): #Asks for loss value before each run
                self._step +=1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)
            #runs to fetch loss
            
            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = float(duration)
                    
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))
                    
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir, 
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), 
                    tf.train.NanTensorHook(loss), _LoggerHook()],
            config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
            

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
    
    
                    
if __name__=='__main__':
    tf.app.run()
    
