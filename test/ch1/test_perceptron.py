from unittest import TestCase

from src.ch1.perceptron import Perceptron


class TestPerceptron(TestCase):
    def test_compute(self):
        p = Perceptron(w=[1, 1], b=-1)
        self.assertEqual(p.compute([0, 0]), 0)

    def try_cases(self, perceptron, io):
        for case in io:
            self.assertEqual(perceptron.compute(case[0]), case[1])

    def test_and(self):
        """
        Create a perceptron that mimics an `and` gate
        """
        p = Perceptron(w=[1, 1], b=-1)
        cases = [
            ([0, 0], 0),
            ([0, 1], 0),
            ([1, 0], 0),
            ([1, 1], 1),
        ]
        self.try_cases(perceptron=p, io=cases)

    def test_or(self):
        """
        Create a perceptron that mimics an `or` gate
        """
        p = Perceptron(w=[1, 1], b=0)
        cases = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 1),
        ]
        self.try_cases(perceptron=p, io=cases)

    def test_nand(self):
        """
        Create a perceptron that mimics a `nand` gate
        """
        p = Perceptron(w=[-1, -1], b=2)
        cases = [
            ([0, 0], 1),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]
        self.try_cases(perceptron=p, io=cases)

    def test_bitwise_sum(self):
        """
        Let's build a circuit that can do bitwise sum
        See: http://neuralnetworksanddeeplearning.com/images/tikz3.png
        """

        def create_nand():
            return Perceptron(w=[-1, -1], b=2)

        def bitwise_sum(x):
            nand = create_nand()

            n1_val = nand.compute(x)
            r_sum = nand.compute([nand.compute([x[0], n1_val]), nand.compute([n1_val, x[1]])])
            r_carry = nand.compute([n1_val, n1_val])
            return r_sum, r_carry

        # 0+0 = 0, carry 0
        self.assertEqual(bitwise_sum([0, 0]), (0, 0))
        # 1+0 = 1, carry 0
        self.assertEqual(bitwise_sum([0, 1]), (1, 0))
        self.assertEqual(bitwise_sum([1, 0]), (1, 0))
        # 1+1 = 0, carry 1
        self.assertEqual(bitwise_sum([1, 1]), (0, 1))
