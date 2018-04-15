import unittest
from ai import AI

class TestAI(unittest.TestCase):
  ai = None

  def setUp(self):
    self.ai = AI(restore=False, save=False)

  def test_prediction(self):
    prediction = self.ai.get_prediction([0, 0, 0, 0, 0])
    self.assertEquals(prediction, [0])

  def test_training(self):
    n_steps = 100
    self.ai.train_with_samples({
      'x': [[0, 0, 0, 0, 0]] * n_steps,
      'y': [[0, 0, 1]] * n_steps,
    })
    prediction = self.ai.get_prediction([0, 0, 0, 0, 0])
    self.assertEquals(prediction, [2])

if __name__ == '__main__':
  unittest.main()
