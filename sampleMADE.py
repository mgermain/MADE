#!/usr/bin/python -u

import numpy as np
import os
import argparse
import time as t

import theano
import theano.tensor as Te

import utils
from dataset import Dataset
from trainMADE import load_model_params, build_model

import matplotlib.pyplot as plt
from matplotlib import cm


def batch_get_nearest_neighbours(samples, dataset):
    sample = Te.matrix(name="sample")
    data = Te.matrix(name="dataset")
    find_nearest_neighbour = theano.function(name="find_nearest_neighbour",
                                             inputs=[sample],
                                             outputs=data[Te.argmin(Te.sum((data[:, None, :] - sample) ** 2, axis=2), axis=0)],
                                             givens={data: dataset['train']['data']})
    return find_nearest_neighbour(samples)


def get_nearest_neighbours(samples, dataset):
    sample = Te.vector(name="sample")
    data = Te.matrix(name="dataset")
    find_nearest_neighbour = theano.function(name="find_nearest_neighbour",
                                             inputs=[sample],
                                             outputs=data[Te.argmin(Te.sum((data - sample) ** 2, axis=1))],
                                             givens={data: dataset['train']['data']})
    neighbours = []
    for s in samples:
        neighbours += [find_nearest_neighbour(s)]
    return neighbours


def get_model(model_path):
    hyperparams = utils.load_dict_from_json_file(os.path.join(model_path, "hyperparams"))
    hyperparams['weights_initialization'] = "Zeros"

    trainingparams = utils.load_dict_from_json_file(os.path.join(model_path, "trainingparams"))
    dataset_name = trainingparams['dataset_name']

    if dataset_name != "binarized_mnist":
        raise(Exception("Invalid dataset. Only model trained on MNIST supported."))

    #
    # LOAD DATASET ####
    dataset = Dataset.get(dataset_name)
    if trainingparams['batch_size'] == -1:
        trainingparams['batch_size'] = dataset['train']['length']

    model = build_model(dataset, trainingparams, hyperparams, hyperparams['hidden_sizes'])

    print "### Loading model | Hidden:{0} CondMask:{1} Activation:{2} ... ".format(hyperparams['hidden_sizes'], hyperparams['use_cond_mask'], hyperparams['hidden_activation']),
    start_time = t.time()
    load_model_params(model, model_path)
    print utils.get_done_text(start_time), "###"

    return model, dataset_name, dataset


def save_samples(x, y, samples, dataset_name, model_name):
    total_final_array = []
    for w in range(y):
        total_final_array += [np.hstack([np.pad(zz.reshape((28, 28)), 1, mode='constant', constant_values=1) for zz in samples[w * x:w * x + x]])]

    plt.imsave('samples_{0}x{1}_{2}_{3}.png'.format(x, y, dataset_name, model_name), np.vstack(total_final_array), cmap=cm.Greys_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from MADE trained on MNIST and generate an image of X by Y samples.')
    parser.add_argument('model_path', help='Path to the experiment folder of the model to sample from.')
    parser.add_argument('nb_samples_per_row', help="X: # of samples on each row.", type=int, default=10)
    parser.add_argument('nb_rows', help="Y: # of rows.", type=int, default=10)
    parser.add_argument('find_nearest_neighbour', metavar='find_nearest_neighbour', type=eval, choices=[False, True], help="Generate the nearest neighbour (in trainset) image of the samples: {%(choices)s}", default=False)
    parser.add_argument('change_mask', metavar='change_mask', help="Change mask for every row when generating samples: {%(choices)s}", type=eval, choices=[False, True])
    parser.add_argument('nb_images', help="# of images to generate.", type=int, default=1)
    args = parser.parse_args()

    model, dataset_name, dataset = get_model(args.model_path)

    for run in range(args.nb_images):
        print "Image {0}".format(run)
        print "### Generating {} samples ...".format(args.nb_samples_per_row * args.nb_rows),
        name = "_samples_run{}".format(run)

        start_time = t.time()
        if args.change_mask:
            name += "_change_mask"
            samples = model.sample(args.nb_samples_per_row, 0)
            for i in range(1, args.nb_rows):
                samples = np.vstack([model.sample(args.nb_samples_per_row, i), samples])
        else:
            samples = model.sample(args.nb_samples_per_row * args.nb_rows)
        print utils.get_done_text(start_time), "###"

        if args.find_nearest_neighbour:
            print "### Finding neighbours ...",
            start_time = t.time()
            samples_neighbours = get_nearest_neighbours(samples, dataset)
            print utils.get_done_text(start_time), "###"

            save_samples(args.nb_samples_per_row, args.nb_rows, samples_neighbours, dataset_name + name + "_neighbours", os.path.basename(args.model_path))

        save_samples(args.nb_samples_per_row, args.nb_rows, samples, dataset_name + name, os.path.basename(args.model_path))
