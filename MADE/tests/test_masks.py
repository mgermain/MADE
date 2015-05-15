import unittest
from numpy.testing import assert_array_equal
from nose.tools import assert_false

import numpy as np
import theano
from dataset import Dataset
from MADE.made import MADE


class MasksTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 10
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "None"
        self.direct_output_connect = False

    def setUp(self):
        fake_dataset = Dataset.get_fake(self.input_size, 1)
        self.model = MADE(fake_dataset,
                          hidden_sizes=self.hidden_sizes,
                          batch_size=fake_dataset['train']['data'].shape[0],
                          hidden_activation=theano.tensor.nnet.sigmoid,
                          use_cond_mask=self.use_cond_mask,
                          direct_input_connect=self.direct_input_connect,
                          direct_output_connect=self.direct_output_connect)

        self.nb_shuffle = 50

    def tearDown(self):
        pass

    def _shuffles(self, shuffle_type):
        for i in range(self.nb_shuffle):
            self.model.shuffle(shuffle_type)

    def _get_masks(self):
        return [layer.weights_mask.get_value() for layer in self.model.layers]

    def test_base(self):
        for layer in self.model.layers:
            mask = layer.weights_mask.get_value()
            assert_array_equal(mask, np.ones_like(mask))

    def test_shuffle_once(self):
        shuffle_type = "Once"
        self.model.shuffle("Full")  # Initial shuffle, always Full

        initial_masks = self._get_masks()

        self.model.shuffle(shuffle_type)

        shuffled_once_masks = self._get_masks()

        # Testing that they are shuffled
        for masks in zip(initial_masks, shuffled_once_masks):
            assert_false(np.array_equal(masks[0], masks[1]))

        self._shuffles(shuffle_type)

        shuffled_masks = self._get_masks()

        # Testing that they are not shuffled again
        for masks in zip(shuffled_once_masks, shuffled_masks):
            assert_array_equal(masks[0], masks[1])

    def test_reset(self):
        shuffle_type = "Full"
        self.model.shuffle("Full")  # Initial shuffle, always Full

        initial_masks = self._get_masks()

        self._shuffles(shuffle_type)

        shuffled_masks = self._get_masks()

        # Testing that they are shuffled
        for masks in zip(initial_masks, shuffled_masks):
            assert_false(np.array_equal(masks[0], masks[1]))

        self.model.reset(shuffle_type)

        # Testing that they are resetted
        for i in range(len(self.model.layers)):
            assert_array_equal(initial_masks[i], self.model.layers[i].weights_mask.get_value())

        self._shuffles(shuffle_type)  # Shuffling again

        # Testing that they are reshuffled exactly at the same state
        for i in range(len(self.model.layers)):
            assert_array_equal(shuffled_masks[i], self.model.layers[i].weights_mask.get_value())


# Need more tests for the reset. See TestMade.py:verify_reset_mask()
