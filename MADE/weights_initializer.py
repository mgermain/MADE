import numpy as np
import theano


class WeightsInitializer(object):

    def __init__(self, random_seed):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def _init_range(self, dim):
        return np.sqrt(6. / (dim[0] + dim[1]))

    def Uniform(self, dim):
        init_range = self._init_range(dim)
        return np.asarray(self.rng.uniform(low=-init_range, high=init_range, size=dim), dtype=theano.config.floatX)

    def Zeros(self, dim):
        return np.zeros(dim, dtype=theano.config.floatX)

    def Diagonal(self, dim):
        W_values = self.Zeros(dim)
        np.fill_diagonal(W_values, 1)
        return W_values

    def Orthogonal(self, dim):
        max_dim = max(dim)
        return np.linalg.svd(self.Uniform((max_dim, max_dim)))[2][:dim[0], :dim[1]]

    def Gaussian(self, dim):
        return np.asarray(self.rng.normal(loc=0, scale=self._init_range(dim), size=dim), dtype=theano.config.floatX)
