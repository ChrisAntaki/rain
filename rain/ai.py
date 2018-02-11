"""Artificial intelligence that plays games."""
from __future__ import print_function
from glob import glob
import json
# import random
import sys
import subprocess
import tensorflow as tf


# Set a constant random seed.
# random.seed(100)

size_input = 5
size_hidden = 200
size_output = 3

a = tf.placeholder(tf.float32)
t = tf.placeholder(tf.float32, [None, size_output])
x = tf.placeholder(tf.float32, [None, size_input])
keep_prob = tf.placeholder(tf.float32)

class AI(object):
  '''AI for the rain game's player.'''
  def __init__(self, restore=0):
    # Define variables.
    weights_1 = tf.Variable(tf.truncated_normal([size_input, size_hidden]))
    weights_2 = tf.Variable(tf.truncated_normal([size_hidden, size_output]))
    biases_1 = tf.Variable(tf.zeros([size_hidden]))
    biases_2 = tf.Variable(tf.zeros([size_output]))

    # Define model.
    model = tf.matmul(x, weights_1) + biases_1
    model = tf.nn.relu(model)
    model = tf.nn.dropout(model, keep_prob)
    model = tf.matmul(model, weights_2) + biases_2
    self.prediction = tf.argmax(model, 1)
    self.loss = a * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=t, logits=model))
    self.average_score = 0.0
    self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
    self.sess = tf.Session()
    self.saver = tf.train.Saver()
    self.checkpoint_path = '/tmp/rain.ckpt'
    # My attempt to add randomness/chaos
    min_prob = tf.reduce_min(model, 1)
    boosted_y = min_prob + model
    shuffled_y = boosted_y * tf.random_uniform([3])
    self.shuffled_prediction = tf.argmax(shuffled_y, 1)
    # Reset...
    self.sess.run(tf.global_variables_initializer())
    if restore:
      try:
        self.saver.restore(self.sess, self.checkpoint_path)
      except:
        pass
    else:
      subprocess.call('rm -rf /tmp/rain*', shell=True)

  def train_with_samples(self, samples):
    '''Trains the neural network.'''
    number_of_samples = len(samples['x'])
    self.average_score -= self.average_score / 50.0
    self.average_score += number_of_samples / 50.0
    print('Avg Score: {}'.format(self.average_score))
    sys.stdout.flush()
    for i in range(number_of_samples):
      if i < number_of_samples - 1:
        sample_a = 1
      else:
        # The last move was a mistake...
        sample_a = -1
      sample_x = samples['x'][i]
      sample_t = samples['y'][i]
      self.sess.run(
          self.train,
          {
              x: [sample_x],
              t: [sample_t],
              a: sample_a,
              keep_prob: 0.5,
          })
    self.saver.save(self.sess, self.checkpoint_path)

  def get_prediction(self, sample_x):
    '''Simple prediction.'''
    return self.sess.run(
        self.prediction,
        {x: [sample_x], keep_prob: 1})

  def get_shuffled_prediction(self, sample_x):
    '''My attempt at adding some chaos/exploration.'''
    return self.sess.run(
        self.shuffled_prediction,
        {x: [sample_x], keep_prob: 1})
