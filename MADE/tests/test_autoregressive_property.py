import unittest
from numpy.testing import assert_array_equal, assert_almost_equal

import numpy as np
import theano
from dataset import Dataset
from MADE.made import MADE


class AutoregressivePropertyTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "None"
        self.direct_output_connect = False

    def setUp(self):
        ### Testing that the sum of all prob is equal to 1 ###
        # This test has to be run in 64bit for accuracy
        self._old_theano_config_floatX = theano.config.floatX
        theano.config.floatX = 'float64'
        self.nb_test = 15

        self._shuffling_type = "Full"

        fake_dataset = Dataset.get_permutation(self.input_size)
        self.model = MADE(fake_dataset,
                          hidden_sizes=self.hidden_sizes,
                          batch_size=fake_dataset['train']['data'].shape[0],
                          hidden_activation=theano.tensor.nnet.sigmoid,
                          use_cond_mask=self.use_cond_mask,
                          direct_input_connect=self.direct_input_connect,
                          direct_output_connect=self.direct_output_connect)

        # Train the model to have more accurate results
        for i in range(2 * self.input_size):
            self.model.shuffle(self._shuffling_type)
            self.model.learn(i, True)

    def tearDown(self):
        theano.config.floatX = self._old_theano_config_floatX

    def test_total_prob(self):
        print "Testing on model: input_size={0} hidden_sizes={1}{2}{3}{4} ".format(self.input_size, self.hidden_sizes, " CondMask" if self.use_cond_mask else "", " DirectInputConnect" + self.direct_input_connect if self.direct_input_connect != "None" else "", " DirectOutputConnect" if self.direct_output_connect else ""),

        # Test the model on all the data
        ps = []
        for i in range(self.nb_test):
            ps += np.exp(self.model.valid_log_prob(False), dtype=theano.config.floatX).sum(dtype=theano.config.floatX),
            self.model.shuffle(self._shuffling_type)
            print ".",
        assert_almost_equal(ps, 1)

    def test_verify_masks(self):
        nb_perm_mask = 10

        # Set all the parameters to one
        for layer in self.model.layers:
            for param in layer.params:
                param.set_value(np.ones(param.shape.eval(), dtype=theano.config.floatX))

        # Make sure that the input do not "see" itself
        for perm in range(nb_perm_mask):
            base = self.model.use(np.zeros((1, self.input_size), dtype=theano.config.floatX), False)

            for i in range(self.input_size):
                inp = np.zeros((1, self.input_size), dtype=theano.config.floatX)
                inp[0][i] = 1
                test = self.model.use(inp, False)

                assert_array_equal(base[0][i], test[0][i])

            self.model.shuffle(self._shuffling_type)


class AutoregressivePropertyTests_CondMask(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = True
        self.direct_input_connect = "None"
        self.direct_output_connect = False


# DirectInput
class AutoregressivePropertyTests_DirectInputOut(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "Output"
        self.direct_output_connect = False


class AutoregressivePropertyTests_DirectInputFull(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "Full"
        self.direct_output_connect = False


class AutoregressivePropertyTests_CondMask_DirectInputOut(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = True
        self.direct_input_connect = "Output"
        self.direct_output_connect = False


class AutoregressivePropertyTests_CondMask_DirectInputFull(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = True
        self.direct_input_connect = "Full"
        self.direct_output_connect = False


# DirectOutput
class AutoregressivePropertyTests_DirectOutput(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "None"
        self.direct_output_connect = True


class AutoregressivePropertyTests_CondMask_DirectOutput(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = True
        self.direct_input_connect = "None"
        self.direct_output_connect = True


class AutoregressivePropertyTests_DirectInputOut_DirectOutput(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "Output"
        self.direct_output_connect = True


class AutoregressivePropertyTests_DirectInputFull_DirectOutput(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = False
        self.direct_input_connect = "Full"
        self.direct_output_connect = True


class AutoregressivePropertyTests_CondMask_DirectInputOut_DirectOutput(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = True
        self.direct_input_connect = "Output"
        self.direct_output_connect = True


class AutoregressivePropertyTests_CondMask_DirectInputFull_DirectOutput(AutoregressivePropertyTests):

    def __init__(self, *args, **kwargs):
        AutoregressivePropertyTests.__init__(self, *args, **kwargs)

        # Setup model
        self.input_size = 12
        self.hidden_sizes = [500]
        self.use_cond_mask = True
        self.direct_input_connect = "Full"
        self.direct_output_connect = True
