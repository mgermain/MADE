import unittest
from nose.tools import assert_equal, assert_raises, assert_false

import numpy as np
from numpy.testing import assert_array_equal

from MADE.mask_generator import MaskGenerator


class MaskGeneratorTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.input_size = 10
        self.hidden_sizes = []

    def setUp(self):
        self.nb_layers = len(self.hidden_sizes) + 2
        self.mask_generator = MaskGenerator(self.input_size, self.hidden_sizes, 0)

    def tearDown(self):
        pass

    def test_mask(self):
        for i in range(self.nb_layers - 1):
            layerToOutput = self.mask_generator.get_mask_layer_UPDATE(i).eval()
            layerConnectivity = self.mask_generator.layers_connectivity[i].get_value()
            nextLayerConnectivity = self.mask_generator.layers_connectivity[i + 1].get_value()

            for i, output in enumerate(layerToOutput.T):
                conn = (layerConnectivity <= nextLayerConnectivity[i])
                assert_equal(output[conn].sum(), conn.sum())
                assert_equal(output[np.logical_not(conn)].sum(), 0)

    def test_direct_output_mask(self):
        # layer0ToOutput = self.mask_generator.get_direct_output_mask_layer_UPDATE(0).eval()
        # VALID_layer0ToOutput = np.triu(np.ones((self.input_size, self.input_size)), 1)
        # assert_array_equal(layer0ToOutput, VALID_layer0ToOutput)

        for i in range(self.nb_layers):
            layerToOutput = self.mask_generator.get_direct_output_mask_layer_UPDATE(i).eval()
            layerConnectivity = self.mask_generator.layers_connectivity[i].get_value()
            for output, conn in zip(layerToOutput, layerConnectivity):
                assert_equal(output[:conn].sum(), 0)
                assert_equal(output[conn:].sum(), self.input_size - conn)

        assert_raises(IndexError, self.mask_generator.get_direct_output_mask_layer_UPDATE, self.nb_layers)

    def test_direct_input_mask(self):
        for i in range(self.nb_layers):
            layerToOutput = self.mask_generator.get_direct_input_mask_layer_UPDATE(i).eval()
            layerConnectivity = self.mask_generator.layers_connectivity[i].get_value()
            for output, conn in zip(layerToOutput.T, layerConnectivity):
                assert_equal(output[:conn].sum(), conn)
                assert_equal(output[conn:].sum(), 0)

        assert_raises(IndexError, self.mask_generator.get_direct_input_mask_layer_UPDATE, self.nb_layers)

    def test_shuffle_ordering(self):
        initial_ordering = self.mask_generator.ordering.get_value()
        initial_input_connectivity = self.mask_generator.layers_connectivity[0].get_value()

        assert_array_equal(initial_ordering, np.arange(self.input_size))
        assert_array_equal(initial_ordering + 1, initial_input_connectivity)

        # Shuffling the mask
        self.mask_generator.shuffle_ordering()

        shuffled_ordering = self.mask_generator.ordering.get_value()
        shuffled_input_connectivity = self.mask_generator.layers_connectivity[0].get_value()

        assert_equal(initial_ordering.sum(), shuffled_ordering.sum())  # Making sure that all the number are there
        assert_array_equal(set(initial_ordering), set(shuffled_ordering))  # Making sure that all the number are there
        assert_false(np.array_equal(initial_ordering, shuffled_ordering))  # Making sure that there was an actual shuffle
        assert_array_equal(shuffled_ordering + 1, shuffled_input_connectivity)

        # Making sure that the mask is not only shuffled once
        self.mask_generator.shuffle_ordering()

        last_ordering = self.mask_generator.ordering.get_value()
        last_input_connectivity = self.mask_generator.layers_connectivity[0].get_value()

        assert_equal(shuffled_ordering.sum(), last_ordering.sum())  # Making sure that all the number are there
        assert_array_equal(set(shuffled_ordering), set(last_ordering))  # Making sure that all the number are there
        assert_false(np.array_equal(shuffled_ordering, last_ordering))  # Making sure that there was an actual shuffle
        assert_array_equal(last_ordering + 1, last_input_connectivity)

    def _reset_test_get_masks(self):
        ordering = self.mask_generator.ordering.get_value()
        masks = [self.mask_generator.get_mask_layer_UPDATE(i).eval() for i in range(self.nb_layers - 1)]
        masks_direct_input = [self.mask_generator.get_direct_input_mask_layer_UPDATE(i).eval() for i in range(self.nb_layers)]
        masks_direct_output = [self.mask_generator.get_direct_output_mask_layer_UPDATE(i).eval() for i in range(self.nb_layers)]
        return ordering, masks, masks_direct_input, masks_direct_output

    def test_reset(self):
        ordering, masks, masks_direct_input, masks_direct_output = self._reset_test_get_masks()

        # shuffle gen a couple iteration of mask
        for i in range(3):
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()

        new_ordering, new_masks, new_masks_direct_input, new_masks_direct_output = self._reset_test_get_masks()

        assert_false(np.array_equal(ordering, new_ordering))
        for m, nm in zip(masks, new_masks):
            assert_false(np.array_equal(m, nm))
        for m, nm in zip(masks_direct_input, new_masks_direct_input):
            assert_false(np.array_equal(m, nm))
        for m, nm in zip(masks_direct_output, new_masks_direct_output):
            assert_false(np.array_equal(m, nm))

        self.mask_generator.reset()

        r_ordering, r_masks, r_masks_direct_input, r_masks_direct_output = self._reset_test_get_masks()

        assert_array_equal(ordering, r_ordering)
        for i in range(len(masks)):
            assert_array_equal(masks[i], r_masks[i])
        for i in range(len(masks_direct_input)):
            assert_array_equal(masks_direct_input[i], r_masks_direct_input[i])
        for i in range(len(masks_direct_output)):
            assert_array_equal(masks_direct_output[i], r_masks_direct_output[i])

        # shuffle gen a couple iteration of mask
        for i in range(3):
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()

        r_new_ordering, r_new_masks, r_new_masks_direct_input, r_new_masks_direct_output = self._reset_test_get_masks()

        assert_array_equal(new_ordering, r_new_ordering)
        for i in range(len(new_masks)):
            assert_array_equal(new_masks[i], r_new_masks[i])
        for i in range(len(new_masks_direct_input)):
            assert_array_equal(new_masks_direct_input[i], r_new_masks_direct_input[i])
        for i in range(len(new_masks_direct_output)):
            assert_array_equal(new_masks_direct_output[i], r_new_masks_direct_output[i])


class MaskGeneratorTests_MNIST_no_hidden(MaskGeneratorTests):

    def __init__(self, *args, **kwargs):
        super(MaskGeneratorTests, self).__init__(*args, **kwargs)
        self.input_size = 784
        self.hidden_sizes = []


class MaskGeneratorTests_MNIST_1_500_hidden(MaskGeneratorTests):

    def __init__(self, *args, **kwargs):
        super(MaskGeneratorTests, self).__init__(*args, **kwargs)
        self.input_size = 784
        self.hidden_sizes = [500]


class MaskGeneratorTests_MNIST_2_500_hidden(MaskGeneratorTests):

    def __init__(self, *args, **kwargs):
        super(MaskGeneratorTests, self).__init__(*args, **kwargs)
        self.input_size = 784
        self.hidden_sizes = [500, 500]


class MaskGeneratorTests_MNIST_3_500_hidden(MaskGeneratorTests):

    def __init__(self, *args, **kwargs):
        super(MaskGeneratorTests, self).__init__(*args, **kwargs)
        self.input_size = 784
        self.hidden_sizes = [500, 500, 500]
