import os
import time as t
import theano
import numpy as np
from utils import get_done_text


class Dataset(object):

    @staticmethod
    def get_permutation(input_size):
        # Generate binary dataset of all possible permutations for size input_size
        import itertools
        print "Generating dataset of all possible permutations for size ({0}) input_size ...".format(input_size),
        dataset = []
        for i in itertools.product("01", repeat=input_size):
            dataset.append(i)
        fake_dataset = {'input_size': input_size}
        fake_dataset['valid'] = {'data': theano.shared(value=np.asarray(dataset, dtype=theano.config.floatX), borrow=True)}
        fake_dataset['train'] = {'data': theano.shared(value=np.asarray(dataset[:int(len(dataset) * 0.10)], dtype=theano.config.floatX), borrow=True)}
        fake_dataset['test'] = {'data': theano.shared(value=np.asarray(dataset, dtype=theano.config.floatX), borrow=True)}
        print "Done. {0} items generated.".format(len(dataset))
        return fake_dataset

    @staticmethod
    def get_fake(in_size=4, dataset_size=1):
        fake_dataset = {'input_size': in_size}
        fake_dataset['valid'] = {'data': theano.shared(value=np.zeros((dataset_size, in_size), dtype=theano.config.floatX), borrow=True), 'length': dataset_size}
        fake_dataset['train'] = {'data': theano.shared(value=np.zeros((dataset_size, in_size), dtype=theano.config.floatX), borrow=True), 'length': dataset_size}
        fake_dataset['test'] = {'data': theano.shared(value=np.zeros((dataset_size, in_size), dtype=theano.config.floatX), borrow=True), 'length': dataset_size}
        return fake_dataset

    @staticmethod
    def _clean(dataset):
        data = []
        for i in dataset:
            data.append(i)
        return np.asarray(data, dtype=theano.config.floatX)

    @staticmethod
    def get(dataset_name):
        datasets = ['adult',
                    'binarized_mnist',
                    'connect4',
                    'dna',
                    'mushrooms',
                    'nips',
                    'ocr_letters',
                    'rcv1',
                    'web']

        if dataset_name not in datasets:
            raise ValueError('Dataset unknown: ' + dataset_name)

        print '### Loading dataset [{0}] ...'.format(dataset_name),
        start_time = t.time()

        raw_dataset = np.load(os.path.join("datasets", dataset_name) + ".npz")
        full_dataset = {'input_size': raw_dataset['inputsize']}

        trainset_theano = theano.shared(value=raw_dataset['train_data'], borrow=True)
        validset_theano = theano.shared(value=raw_dataset['valid_data'], borrow=True)
        testset_theano = theano.shared(value=raw_dataset['test_data'], borrow=True)

        full_dataset['train'] = {'data': trainset_theano, 'length': raw_dataset['train_length']}
        full_dataset['valid'] = {'data': validset_theano, 'length': raw_dataset['valid_length']}
        full_dataset['test'] = {'data': testset_theano, 'length': raw_dataset['test_length']}

        print "(Dim:{0} Train:{1} Valid:{2} Test:{3})".format(full_dataset['input_size'], full_dataset['train']['length'], full_dataset['valid']['length'], full_dataset['test']['length']),
        print get_done_text(start_time), "###"
        return full_dataset
