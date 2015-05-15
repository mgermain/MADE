import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse


class Layer(object):

    def __init__(self, layerIdx, input, n_in, n_out, weights_initialization, activation=None):
        self.input = input
        self._activation = activation
        self.layerIdx = layerIdx
        self.n_in = n_in
        self.n_out = n_out
        self.weights_initialization = weights_initialization

        # Init weights and biases
        self.W = theano.shared(value=weights_initialization((n_in, n_out)), name='W{}'.format(layerIdx), borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b{}'.format(layerIdx), borrow=True)

        # Output
        self.lin_output = T.dot(input, self.W) + self.b

        # Parameters of the layer
        self.params = [self.W, self.b]

    def _output(self):
        return (self.lin_output if self._activation is None else self._activation(self.lin_output))

    @property
    def output(self):
        return self._output()


class MaskedLayer(Layer):

    def __init__(self, mask_generator, **kargs):
        Layer.__init__(self, **kargs)

        self.mask_generator = mask_generator
        self.weights_mask = theano.shared(value=np.ones((self.n_in, self.n_out), dtype=theano.config.floatX), name='weights_mask{}'.format(self.layerIdx), borrow=True)

        # Output
        self.lin_output = T.dot(self.input, self.W * self.weights_mask) + self.b

        self.shuffle_update = [(self.weights_mask, mask_generator.get_mask_layer_UPDATE(self.layerIdx))]


class ConditionningMaskedLayer(MaskedLayer):

    def __init__(self, use_cond_mask=False, **kargs):
        MaskedLayer.__init__(self, **kargs)

        if use_cond_mask:
            self.U = theano.shared(value=self.weights_initialization((self.n_in, self.n_out)), name='U{}'.format(self.layerIdx), borrow=True)

            # Output
            self.lin_output += T.dot(T.ones_like(self.input), self.U * self.weights_mask)

            self.params += [self.U]


class DirectInputConnectConditionningMaskedLayer(ConditionningMaskedLayer):

    def __init__(self, direct_input, **kargs):
        ConditionningMaskedLayer.__init__(self, **kargs)

        if direct_input is not None:
            self.direct_input_weights_mask = theano.shared(value=np.ones((self.mask_generator._input_size, self.n_out), dtype=theano.config.floatX), name='direct_input_weights_mask{}'.format(self.layerIdx), borrow=True)

            self.D = theano.shared(value=self.weights_initialization((self.mask_generator._input_size, self.n_out)), name='D{}'.format(self.layerIdx), borrow=True)

            # Output
            self.lin_output += T.dot(direct_input, self.D * self.direct_input_weights_mask)

            self.params += [self.D]

            self.shuffle_update += [(self.direct_input_weights_mask, self.mask_generator.get_direct_input_mask_layer_UPDATE(self.layerIdx + 1))]


class DirectOutputInputConnectConditionningMaskedOutputLayer(DirectInputConnectConditionningMaskedLayer):

    def __init__(self, direct_outputs=[], **kargs):
        DirectInputConnectConditionningMaskedLayer.__init__(self, **kargs)

        self.direct_ouputs_masks = []
        for direct_out_layerIdx, n_direct_out, direct_output in direct_outputs:

            direct_ouput_weights_mask = theano.shared(value=np.ones((n_direct_out, self.n_out), dtype=theano.config.floatX), name='direct_output_weights_mask{}'.format(self.layerIdx), borrow=True)
            direct_ouput_weight = theano.shared(value=self.weights_initialization((n_direct_out, self.n_out)), name='direct_ouput_weight{}'.format(self.layerIdx), borrow=True)

            # Output
            self.lin_output += T.dot(direct_output, direct_ouput_weight * direct_ouput_weights_mask)

            self.direct_ouputs_masks += [direct_ouput_weights_mask]
            self.params += [direct_ouput_weight]

            self.shuffle_update += [(direct_ouput_weights_mask, self.mask_generator.get_direct_output_mask_layer_UPDATE(direct_out_layerIdx))]


def dropoutLayerDecorator(layer, trng, is_train, dropout_rate):

    if dropout_rate > 0:
        layer._normal_output = layer._output

        def _output():
            return ifelse(is_train, layer._normal_output() * trng.binomial(size=layer._normal_output().shape, p=1 - dropout_rate, dtype=theano.config.floatX), layer._normal_output() * (1 - dropout_rate))

        layer._output = _output

    return layer
