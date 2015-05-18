# MADE
MADE: Masked Autoencoder for Distribution Estimation

Paper on [arxiv](http://arxiv.org/abs/1502.03509) and at [ICML2015](http://icml.cc/2015/?page_id=710).

## Dependencies:
- python = 2.7
- numpy > 1.7
- scipy > 0.11
- theano > 0.6

## Usage
See `python trainMADE.py --help`

Experiments are saved in : *`./experiments/{experiment_name}/`*.

Datasets need to be in : *`./datasets/{dataset_name}.npz`*.

## Train
Commands to generate the best result from the paper on multiple dataset.

DNA
```
python -u trainMADE.py dna 1e-5 0.95 -1 -1 Full 300 100 30 False 0 adadelta 0 [500] 1234 False Output False hinge Orthogonal 0
```

MNIST
```
python -u trainMADE.py --name mnist_from_paper binarized_mnist 0.01 0 -1 32 Full 300 100 30 False 0 adagrad 0 [8000,8000] 1234 False Output False hinge Orthogonal 0
```

## Sample
Generating an X by Y image of MNIST digit sampled from a model (assuming the one above).
```
python -u sampleMADE.py experiments/mnist_from_paper/ 10 10 True True 1
```

## Datasets
In repo:
- adult
- connect4
- dna
- mushrooms
- nips
- ocr_letters
- web

External download due to size:
- [rcv1](https://github.com/mgermain/MADE/releases/download/ICML2015/rcv1.npz)
- [binarized_mnist](https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz)
