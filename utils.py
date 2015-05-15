import sys
import time as t
import pickle
import json
import theano
import hashlib
import platform
import os
from contextlib import contextmanager
import logging
if platform.system() != "Windows":
    import fcntl

    @contextmanager
    def open_with_lock(*args, **kwargs):
        """ Context manager for opening file with an exclusive lock. """
        f = open(*args, **kwargs)
        try:
            fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            logging.info("Can't immediately write-lock the file ({0}), blocking ...".format(f.name))
            fcntl.lockf(f, fcntl.LOCK_EX)
        yield f
        fcntl.lockf(f, fcntl.LOCK_UN)
        f.close()
else:
    @contextmanager
    def open_with_lock(*args, **kwargs):
        """ No lock on windows. Assuming one experiment at the time. """
        f = open(*args, **kwargs)
        yield f
        f.close()


def generate_uid_from_string(value):
    """ Create unique identifier from a string. """
    return hashlib.sha256(value).hexdigest()


def saveFeatures(model):
    features = model.layers[0].W.get_value()
    pickle.dump(features, open('W.pkl', 'wb'))


def write_result(dataset_name, model_info, experiment_name):
    header = ["Learning Rate", "Decrease Constant", "Hidden Layers", "Random Seed", "Activation Function", "Max Epoch", "Best Epoch", "Look Ahead", "Batch Size", "Shuffle Mask", "Shuffling Type", "Nb Shuffle Per Valid", "Conditioning Mask", "Direct Input Connect", "Direct Output Connect", "Pre-Training", "Pre-Training Max Epoch", "Update Rule", "Dropout Rate", "Weights Initialization", "Mask Distribution knob", "Training err", "Training err std", "Validation err", "Validation err std", "Test err", "Test err std", "Total Training Time", "Experiment Name"]

    result = map(str, model_info[0:21])
    result += map("{0:.6f}".format, model_info[21:-1])
    result += ["{0:.4f}".format(model_info[-1])]
    result += [experiment_name]

    print "### Saving result to file ...",
    start_time = t.time()
    write_result_file(dataset_name, header, result)

    print get_done_text(start_time)


def write_result_file(dataset_name, header, result):
    result_file = "_results_on_{0}_MADE.csv".format(dataset_name)

    write_header = not os.path.exists(result_file)

    with open_with_lock(result_file, "a") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write('\t'.join(result) + "\n")


def get_done_text(start_time):
    sys.stdout.flush()
    return "DONE in {:.4f} seconds.".format(t.time() - start_time)


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


def printParams(model):
    for l in model.layers:
        print "\t#",
        for p in l.params:
            print p,
        print


def print_computational_graphs(model, hidden_sizes, shuffle_mask, use_cond_mask, direct_input_connect):
    file_name = "MADE_h{}.shuffle{}.cond_mask{}.direct_conn{}_{}_".format(hidden_sizes, shuffle_mask, use_cond_mask, direct_input_connect, theano.config.device)
    theano.printing.pydotprint(model.use, file_name + "use")
    theano.printing.pydotprint(model.learn, file_name + "learn")
    theano.printing.pydotprint(model.shuffle, file_name + "shuffle")
    theano.printing.pydotprint(model.valid_log_prob, file_name + "log_prob")
    exit()
