# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Artificial intelligence that plays games."""
from __future__ import print_function
from glob import glob
import json
import sys
import subprocess
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

size_input = 5
size_hidden = 200
size_output = 3

a = tf.compat.v1.placeholder(tf.float32)
t = tf.compat.v1.placeholder(tf.float32, [None, size_output])
x = tf.compat.v1.placeholder(tf.float32, [None, size_input])
keep_prob = tf.compat.v1.placeholder(tf.float32)

class AI(object):
  '''AI for the rain game's player.'''
  def __init__(self, restore=True, save=True):
    # Define variables.
    weights_1 = tf.Variable(tf.random.truncated_normal([size_input, size_hidden]))
    weights_2 = tf.Variable(tf.random.truncated_normal([size_hidden, size_output]))
    biases_1 = tf.Variable(tf.zeros([size_hidden]))
    biases_2 = tf.Variable(tf.zeros([size_output]))

    # Define model.
    model = tf.matmul(x, weights_1) + biases_1
    model = tf.nn.relu(model)
    model = tf.nn.dropout(model, 1 - (keep_prob))
    model = tf.matmul(model, weights_2) + biases_2
    self.prediction = tf.argmax(input=model, axis=1)
    self.loss = a * tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
        labels=t, logits=model))
    self.average_score = 0.0
    self.train = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(self.loss)
    self.sess = tf.compat.v1.Session()
    if save:
      self.saver = tf.compat.v1.train.Saver()
    self.checkpoint_path = '/tmp/rain.ckpt'

    # Add a pinch of chaos.
    min_prob = tf.reduce_min(input_tensor=model, axis=1)
    boosted_y = min_prob + model
    shuffled_y = boosted_y * tf.random.uniform([3])
    self.shuffled_prediction = tf.argmax(input=shuffled_y, axis=1)

    # Initialize/reset global variables.
    self.sess.run(tf.compat.v1.global_variables_initializer())
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
    for i in range(number_of_samples):
      if i < number_of_samples - 1:
        # Reward.
        sample_a = 1
      else:
        # Punish.
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
    if getattr(self, 'saver', None):
      print('Game over. Learning from game... Average lifespan is {:.3f} moves.'.format(self.average_score))
      sys.stdout.flush()
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
