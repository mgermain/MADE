from __future__ import division
import numpy as np
import theano
import theano.tensor as T

from update_rules import DecreasingLearningRate, AdaGrad, AdaDelta, RMSProp, Adam, Adam_paper
from layer_types import ConditionningMaskedLayer, dropoutLayerDecorator, DirectInputConnectConditionningMaskedLayer, DirectOutputInputConnectConditionningMaskedOutputLayer  # , MaskGenerator
from mask_generator import MaskGenerator
from weights_initializer import WeightsInitializer


class MADE(object):

    def __init__(self, dataset,
                 learning_rate=0.001,
                 decrease_constant=0,
                 hidden_sizes=[500],
                 random_seed=1234,
                 batch_size=1,
                 hidden_activation=T.nnet.sigmoid,
                 use_cond_mask=False,
                 direct_input_connect="None",
                 direct_output_connect=False,
                 update_rule="None",
                 dropout_rate=0,
                 weights_initialization="Uniform",
                 mask_distribution=0):

        input_size = dataset['input_size']
        self.shuffled_once = False

        class SeedGenerator(object):
            # This subclass purpose is to maximize randomness and still keep reproducibility

            def __init__(self, random_seed):
                self.rng = np.random.mtrand.RandomState(random_seed)

            def get(self):
                return self.rng.randint(42424242)
        self.seed_generator = SeedGenerator(random_seed)

        self.trng = T.shared_randomstreams.RandomStreams(self.seed_generator.get())

        weights_initialization = getattr(WeightsInitializer(self.seed_generator.get()), weights_initialization)  # Get the weights initializer by string name

        # Building the model's graph
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        is_train = T.bscalar(name="is_train")

        # Initialize the mask
        self.mask_generator = MaskGenerator(input_size, hidden_sizes, mask_distribution, self.seed_generator.get())

        # Initialize layers
        input_layer = ConditionningMaskedLayer(layerIdx=0,
                                               input=input,
                                               n_in=input_size,
                                               n_out=hidden_sizes[0],
                                               activation=hidden_activation,
                                               weights_initialization=weights_initialization,
                                               mask_generator=self.mask_generator,
                                               use_cond_mask=use_cond_mask)
        self.layers = [dropoutLayerDecorator(input_layer, self.trng, is_train, dropout_rate)]
        # Now the hidden layers
        for i in range(1, len(hidden_sizes)):
            previous_layer = self.layers[i - 1]
            hidden_layer = DirectInputConnectConditionningMaskedLayer(layerIdx=i,
                                                                      input=previous_layer.output,
                                                                      n_in=hidden_sizes[i - 1],
                                                                      n_out=hidden_sizes[i],
                                                                      activation=hidden_activation,
                                                                      weights_initialization=weights_initialization,
                                                                      mask_generator=self.mask_generator,
                                                                      use_cond_mask=use_cond_mask,
                                                                      direct_input=input if direct_input_connect == "Full" and previous_layer.output != input else None)
            self.layers += [dropoutLayerDecorator(hidden_layer, self.trng, is_train, dropout_rate)]
        # And the output layer
        outputLayerIdx = len(self.layers)
        previous_layer = self.layers[outputLayerIdx - 1]
        self.layers += [DirectOutputInputConnectConditionningMaskedOutputLayer(layerIdx=outputLayerIdx,
                                                                               input=previous_layer.output,
                                                                               n_in=hidden_sizes[outputLayerIdx - 1],
                                                                               n_out=input_size,
                                                                               activation=T.nnet.sigmoid,
                                                                               weights_initialization=weights_initialization,
                                                                               mask_generator=self.mask_generator,
                                                                               use_cond_mask=use_cond_mask,
                                                                               direct_input=input if (direct_input_connect == "Full" or direct_input_connect == "Output") and previous_layer.output != input else None,
                                                                               direct_outputs=[(layer.layer_idx, layer.n_in, layer.input) for layerIdx, layer in enumerate(self.layers[1:-1])] if direct_output_connect else [])]

        # The loss function
        output = self.layers[-1].output
        pre_output = self.layers[-1].lin_output
        log_prob = -T.sum(T.nnet.softplus(-target * pre_output + (1 - target) * pre_output), axis=1)
        loss = (-log_prob).mean()

        # How to update the parameters
        self.parameters = [param for layer in self.layers for param in layer.params]
        parameters_gradient = T.grad(loss, self.parameters)

        # Initialize update_rule
        if update_rule == "None":
            self.update_rule = DecreasingLearningRate(learning_rate, decrease_constant)
        elif update_rule == "adadelta":
            self.update_rule = AdaDelta(decay=decrease_constant, epsilon=learning_rate)
        elif update_rule == "adagrad":
            self.update_rule = AdaGrad(learning_rate=learning_rate)
        elif update_rule == "rmsprop":
            self.update_rule = RMSProp(learning_rate=learning_rate, decay=decrease_constant)
        elif update_rule == "adam":
            self.update_rule = Adam(learning_rate=learning_rate)
        elif update_rule == "adam_paper":
            self.update_rule = Adam_paper(learning_rate=learning_rate)
        updates = self.update_rule.get_updates(zip(self.parameters, parameters_gradient))

        # How to to shuffle weights
        masks_updates = [layer_mask_update for layer in self.layers for layer_mask_update in layer.shuffle_update]
        self.update_masks = theano.function(name='update_masks',
                                            inputs=[],
                                            updates=masks_updates)
        #
        # Functions to train and use the model
        index = T.lscalar()
        self.learn = theano.function(name='learn',
                                     inputs=[index, is_train],
                                     outputs=loss,
                                     updates=updates,
                                     givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size], target: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]},
                                     on_unused_input='ignore')  # ignore for when dropout is absent

        self.use = theano.function(name='use',
                                   inputs=[input, is_train],
                                   outputs=output,
                                   on_unused_input='ignore')  # ignore for when dropout is absent

        # Test functions
        self.valid_log_prob = theano.function(name='valid_log_prob',
                                              inputs=[is_train],
                                              outputs=log_prob,
                                              givens={input: dataset['valid']['data'], target: dataset['valid']['data']},
                                              on_unused_input='ignore')  # ignore for when dropout is absent
        self.train_log_prob = theano.function(name='train_log_prob',
                                              inputs=[is_train],
                                              outputs=log_prob,
                                              givens={input: dataset['train']['data'], target: dataset['train']['data']},
                                              on_unused_input='ignore')  # ignore for when dropout is absent
        self.train_log_prob_batch = theano.function(name='train_log_prob_batch',
                                                    inputs=[index, is_train],
                                                    outputs=log_prob,
                                                    givens={input: dataset['train']['data'][index * 1000:(index + 1) * 1000], target: dataset['train']['data'][index * 1000:(index + 1) * 1000]},
                                                    on_unused_input='ignore')  # ignore for when dropout is absent
        self.test_log_prob = theano.function(name='test_log_prob',
                                             inputs=[is_train],
                                             outputs=log_prob,
                                             givens={input: dataset['test']['data'], target: dataset['test']['data']},
                                             on_unused_input='ignore')  # ignore for when dropout is absent

        # Functions for verify gradient
        self.useloss = theano.function(name='useloss',
                                       inputs=[input, target, is_train],
                                       outputs=loss,
                                       on_unused_input='ignore')  # ignore for when dropout is absent
        self.learngrad = theano.function(name='learn',
                                         inputs=[index, is_train],
                                         outputs=parameters_gradient,
                                         givens={input: dataset['train']['data'][index * batch_size:(index + 1) * batch_size], target: dataset['train']['data'][index * batch_size:(index + 1) * batch_size]},
                                         on_unused_input='ignore')  # ignore for when dropout is absent

    def shuffle(self, shuffling_type):
        if shuffling_type == "Once" and self.shuffled_once is False:
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()
            self.update_masks()
            self.shuffled_once = True
            return

        if shuffling_type in ["Ordering", "Full"]:
            self.mask_generator.shuffle_ordering()
        if shuffling_type in ["Connectivity", "Full"]:
            self.mask_generator.sample_connectivity()
        self.update_masks()

    def reset(self, shuffling_type, last_shuffle=0):
        self.mask_generator.reset()

        # Always do a first shuffle so that the natural order does not gives us an edge
        self.shuffle("Full")

        # Set the mask to the requested shuffle
        for i in range(last_shuffle):
            self.shuffle(shuffling_type)

    def sample(self, nb_samples=1, mask_id=0):
        rng = np.random.mtrand.RandomState(self.seed_generator.get())

        self.reset("Full", mask_id)

        swap_order = self.mask_generator.ordering.get_value()
        input_size = self.layers[0].W.shape[0].eval()

        samples = np.zeros((nb_samples, input_size), theano.config.floatX)

        for i in range(input_size):
            inv_swap = np.where(swap_order == i)[0][0]
            out = self.use(samples, False)
            rng.binomial(p=out[:, inv_swap], n=1)
            samples[:, inv_swap] = rng.binomial(p=out[:, inv_swap], n=1)

        return samples
