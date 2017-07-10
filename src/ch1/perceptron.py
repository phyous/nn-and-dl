import numpy as np


class Perceptron:
    def __init__(self, w, b):
        """
        Initializer for a perceptron
        http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons

        :param w: Array of weights (size j)
        :param b: perceptron bias
        """
        self.w = w
        self.b = b

    def compute(self, x):
        if not isinstance(x, list):
            raise TypeError("input to perceptron must be a list")
        if len(x) != len(self.w):
            raise Exception("length of input array %d must equal length of weight array %d" % (len(x), len(self.w)))

        return 1 if np.dot(x, self.w) + self.b > 0 else 0
