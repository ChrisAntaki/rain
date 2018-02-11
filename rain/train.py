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

"""Trains an AI."""
from __future__ import print_function
from time import sleep
from sys import stdout
import argparse
import random
from ai import AI
from termcolor import colored

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    help='Sets difficulty.',
                    default='hard',
                    choices=['easy', 'hard'],
                    type=str)
parser.add_argument('-v',
                    help='Sets visual mode.',
                    action='store_true')
parser.add_argument('-r',
                    help='Restores the saved model.',
                    default=1,
                    choices=[0, 1],
                    type=int)
args = parser.parse_args()
print(args)

# Initialize variables.
ai = AI(restore=args.r)
player = []
raindrops = []
raindrop_counter = 0
raindrop_counter_target = 0
x = []
y = []

def reset():
  """Resets the game state."""
  global player, raindrops, raindrop_counter, x, y
  player = [9, 4]
  raindrops = []
  if args.d == 'hard':
    add_rain()
    for _ in range(7):
      move_rain()
      add_rain()
  raindrop_counter = 0
  x = []
  y = []

def loop():
  """The main game loop."""
  move_player()
  move_rain()
  if detect_collisions():
    if args.v:
      render()
      print('Score: {}'.format(len(x)))
      stdout.flush()
      sleep(1)

    train()
    reset()
  else:
    add_rain()
  render()
  record()

def move_player():
  """Moves player based on AI predictions."""
  if len(x) is 0:
    return
  prediction = ai.get_prediction(x[-1])[0]
  keys = [0] * 3
  keys[prediction] = 1
  y.append(keys)

  if prediction == 1:
    if player[1] > 0:
      player[1] -= 1
  elif prediction == 2:
    if player[1] < 9:
      player[1] += 1


def move_rain():
  """Moves rain downwards."""
  global raindrops
  raindrops_to_remove = []
  for raindrop in raindrops:
    raindrop[0] += 1
    if raindrop[0] >= 10:
      raindrops_to_remove.append(raindrop)
  for raindrop_to_remove in raindrops_to_remove:
    raindrops.remove(raindrop_to_remove)

def record():
  """Records the inputs.
  There are a total of 5 inputs:
  - Against the wall? There's 2 for left and right.
  - Rain above? There's 3 for up-left, up, and up-right.
  """
  grid = [0] * 5

  # Save wall proximity.
  if player[1] == 0:
    grid[0] = 1
  elif player[1] == 9:
    grid[1] = 1

  # Save nearby overhead raindrops.
  for raindrop in raindrops:
    if raindrop[0] == 8:
      delta_x = raindrop[1] - player[1]
      if abs(delta_x) > 1:
        continue
      grid[3 + delta_x] = 1

  # Save this knowledge.
  x.append(grid)

def detect_collisions():
  """Detects collisions between raindrops and the player."""
  for raindrop in raindrops:
    if raindrop[0] == player[0] and raindrop[1] == player[1]:
      return True
  return False

def add_rain():
  """Spawns new raindrops."""
  global raindrops, raindrop_counter
  if args.d == 'hard':
    # Hard...
    adjustment = random.randrange(0, 2)
    for i in range(5):
      raindrops.append([0, 2 * i + adjustment])
  else:
    # Easy...
    if raindrop_counter == 0:
      raindrops.append([0, random.randrange(0, 10)])
      raindrop_counter = raindrop_counter_target
    else:
      raindrop_counter -= 1

def render():
  """Renders the game."""
  if not args.v:
    return
  # Clear the terminal
  print(chr(27) + "[2J")

  grid = [' '] * 100
  for raindrop in raindrops:
    grid[raindrop[0] * 10 + raindrop[1]] = colored('O', 'blue')
  grid[player[0] * 10 + player[1]] = colored('X', 'yellow')
  for i in range(10):
    row = ''
    for ii in range(10):
      row += grid[i * 10 + ii]
    print(row)
  print('')
  stdout.flush()

def train():
  """Trains the AI based on the last play session."""
  ai.train_with_samples({
      'x': x,
      'y': y,
  })

reset()
while True:
  loop()
  if args.v:
    sleep(1e-1)
  else:
    sleep(1e-7)
