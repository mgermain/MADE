import numpy as np
import theano
import time as t
from MADE.made import MADE
from dataset import Dataset
from utils import get_done_text


class Tester(object):

    """
    Old tests! Some might not work anymore. Cleaner and complementary one in MADE/tests/
    """

    def _get_fake_model(self, in_size, hidden_sizes):
        fake_dataset = Dataset.get_fake(in_size, 1)

        return MADE(fake_dataset,
                    # learning_rate=trainingparams['learning_rate'],
                    # decrease_constant=trainingparams['decrease_constant'],
                    hidden_sizes=hidden_sizes,
                    # random_seed=hyperparams['random_seed'],
                    # batch_size=trainingparams['batch_size'],
                    hidden_activation=lambda x: x,
                    use_cond_mask=False,
                    direct_input_connect="Output",
                    direct_output_connect=False)

    def print_mask(self, model):
        #import pprint
        for layer in model.layers:
            print "#"
            if "direct_ouputs_masks" in layer.__dict__:
                for l in layer.direct_ouputs_masks:
                    print l.get_value()
            print "W"
            print layer.weights_mask.get_value()
            if "direct_input_weights_mask" in layer.__dict__:
                print "D"
                print layer.direct_input_weights_mask.get_value()
        print ""

    def visual_mask_test(self, in_size=3, hidden_sizes=[2, 4]):
        model = self._get_fake_model(in_size, hidden_sizes)

        self.print_mask(model)
        print "\n|/-\\|/-\\ SHUFFLING |/-\\|/-\\\n"
        model.shuffle("Ordering")
        self.print_mask(model)
        exit()

    def _save_graph(self, name, x_label, y_label, *args):
        import pylab as pl

        pl.xlabel(x_label)
        pl.ylabel(y_label)
        pl.grid(True)

        for data in args:
            pl.plot(data[0], data[1], 'x-', linewidth=.5, color=data[2])

        pl.savefig('{0}.png'.format(name))

        return pl

    def _eval_valid_multiple_shuffle(self, model, nb_test_per_step):
        nb_shuffles = []
        avg_nll = []

        x_label = 'Nb Shuffle'
        y_label = 'Avg NLL'

        start_time = t.time()
        print "#",
        steps = [10, 100, 500]
        nb_test_per_step += 1
        for stepIdx in range(len(steps)):
            step = range_start = steps[stepIdx]
            if stepIdx != 0:
                range_start = steps[stepIdx - 1] * (nb_test_per_step - 1) + step

            for current_nb_shuffle in range(range_start, nb_test_per_step * step, step):
                print "/{0}\\".format(current_nb_shuffle),
                # TODO Fix missing variable due to change of scope
                # avg_valid_nll, avg_valid_nll_std = 0, 0  # get_mean_error_and_std(model, model.valid_log_prob, validset_theano.shape.eval()[0], shuffle_mask, current_nb_shuffle)
                avg_valid_nll = 0
                nb_shuffles += [current_nb_shuffle]
                avg_nll += [round(avg_valid_nll, 2)]
            self._save_graph('{0}shuffle_test_temp'.format(step), x_label, y_label, (nb_shuffles, avg_nll, 'purple'))
            print ".",
        print get_done_text(start_time)
        return nb_shuffles, avg_nll

    def print_valid_shuffle_graph(self, model):
        print "\n### Testing how tne nb of shuffle when validating affect NLL ###"
        nb_test_per_step = 5

        x_label = 'Nb Shuffle'
        y_label = 'Avg NLL'

        plots = []
        colors = ['red', 'green', 'blue']
        for i in range(2):
            nb_shuffles, avg_nll = self._eval_valid_multiple_shuffle(model, nb_test_per_step)
            plot = (nb_shuffles, avg_nll, colors[i])
            plots += plot
            self._save_graph('shuffle_test{0}'.format(i), x_label, y_label, plot)

        self._save_graph('shuffle_test', x_label, y_label, plots).show()
        exit()

    def _get_masks(self, model):
        return np.asarray([layer.weights_mask.get_value() for layer in model.layers])

    def verify_reset_mask(self, in_size=10, hidden_sizes=[10, 3], nb_perm_mask=50):
        model = self._get_fake_model(in_size, hidden_sizes)
        print in_size, hidden_sizes, nb_perm_mask,

        # Set all the "parameters" to one
        for layer in model.layers:
            for param in layer.params:
                param.set_value(np.ones(param.shape.eval(), dtype=theano.config.floatX))

        for perm in range(nb_perm_mask):
            # Get baseline to compare with
            zeroInput = np.zeros((1, in_size), dtype=theano.config.floatX)
            base = model.use(zeroInput, False)

            for i in range(in_size):
                inp = np.zeros((1, in_size), dtype=theano.config.floatX)
                inp[0][i] = 1

                test = model.use(inp, False)

                if base[0][i] != test[0][i]:
                    print "\n# BAM! Mask error #"
                    print "in_size", in_size
                    print hidden_sizes
                    print "After {0} shuffle.".format(perm)
            model.shuffle("Full")
            if not perm % 10:
                model.reset_masks()
                print " R ",
            print ".",
        print "SUCESS!"
        exit()

    def visual_verify_reset_mask(self, in_size=5, hidden_sizes=[4, 2], nb_shuffle=3):
        model = self._get_fake_model(in_size, hidden_sizes)
        print in_size, hidden_sizes, nb_shuffle
        print model.parameters

        for i in range(nb_shuffle):
            self.print_mask(model)
            print "shuffle", i
            model.shuffle("Full")
        self.print_mask(model)
        print "###### RESET ######\n"
        model.reset("Full")
        for i in range(nb_shuffle):
            self.print_mask(model)
            print "shuffle", i
            model.shuffle("Full")
        self.print_mask(model)
        exit()


def verify_gradients(model, data):
    epsilon = 1e-6

    print model.parameters[0], model.parameters[0].get_value()
    model.learn(1, True)
    print model.parameters[0], model.parameters[0].get_value()
    model.shuffle("Full")
    model.shuffle("Full")
    model.shuffle("Full")
    model.shuffle("Full")
    model.shuffle("Full")
    model.shuffle("Full")

    parameters_gradient = model.learngrad(1, True)

    emp_grad_weights = [p.get_value() for p in model.parameters]

    print model.parameters[0], model.parameters[0].get_value()
    # for em in model.parameters:
    #     print em, em.get_value()

    def updateValParam(epsilon):
        param = model.parameters[h].get_value()
        param[idx] += epsilon
        model.parameters[h].set_value(param)

    print model.parameters
    for h in range(len(model.parameters)):
        print "Computing empirical gradient for {}".format(model.parameters[h])

        for idx in np.ndindex(tuple(model.parameters[h].shape.eval())):
            updateValParam(epsilon)
            a = model.useloss(data, data, False)
            updateValParam(-epsilon)

            updateValParam(-epsilon)
            b = model.useloss(data, data, False)
            updateValParam(epsilon)

            emp_grad_weights[h][idx] = (a - b) / (2.0 * epsilon)
        print ""

    for h in range(len(model.parameters)):
        print '{0} grad diff : {1}'.format(model.parameters[h], np.mean(np.abs(parameters_gradient[h].ravel() - emp_grad_weights[h].ravel())))


def _get_conditioning_mask_model(dataset, input_size, hidden_sizes):
    import theano.tensor as T
    return MADE(dataset,
                # learning_rate=trainingparams['learning_rate'],
                # decrease_constant=trainingparams['decrease_constant'],
                hidden_sizes=hidden_sizes,
                # random_seed=hyperparams['random_seed'],
                # batch_size=trainingparams['batch_size'],
                #hidden_activation=lambda x: x,
                hidden_activation=T.nnet.sigmoid,
                use_cond_mask=True,
                direct_input_connect="None",
                direct_output_connect=False,
                weights_initialization="Diagonal")


def visual_conditioning_weight():
    input_size = 7
    hidden_sizes = [5]
    fake_dataset = Dataset.get_permutation(input_size)

    model = _get_conditioning_mask_model(fake_dataset, input_size, hidden_sizes)

    for i, l in enumerate(model.layers):
        print "## layer", i
        for p in l.params:
            print p, ":\n", p.get_value()
        print "weights_mask:"
        print l.weights_mask.get_value()

    # import theano.printing as printing
    # for i, p in enumerate(model.parameters):
    #     model.parameters[i] = printing.Print('{0}{1}'.format(p, i))(model.parameters[i])

    # for i, l in enumerate(model.layers):
    #     model.layers[i].lin_output = printing.Print('output{0}'.format(i))(model.layers[i].lin_output)

    # print fake_dataset['train']['data'][0].eval()
    print model.use([np.ones_like(fake_dataset['train']['data'][0].eval())], False)


if __name__ == '__main__':
    Tester().visual_mask_test(3, [2])  # TEST ##
    # Tester().visual_verify_reset_mask()

    # visual_conditioning_weight()

    ## Difference Fini (verify gradient) This TEST must be run in float64 on the CPU and the activation must be Sigmoid (not hinge)##
    # dataset = Dataset.get_permutation(7)
    # model = _get_conditioning_mask_model(dataset, 7, [5])
    # d, batch_size = 1, 100
    # verify_gradients(model, dataset['train']['data'][d * batch_size:(d + 1) * batch_size].eval())

    # Tester().print_valid_shuffle_graph(model)  # TEST ##
