# MADE
MADE: Masked Autoencoder for Distribution Estimation

Paper on [arxiv](http://arxiv.org/abs/1502.03509) and at [ICML2015](http://icml.cc/2015/?page_id=710).

## Dependencies:
- python = 2.7
- numpy >= 1.9.1
- scipy >= 0.14
- theano >= 0.9

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

MNIST (*Warning: Orthogonal initialization takes a long time and a lot of RAM (4gig) with a model that big.*)
```
python -u trainMADE.py --name mnist_from_paper binarized_mnist 0.01 0 -1 32 Full 300 100 30 False 0 adagrad 0 [8000,8000] 1234 False Output False hinge Orthogonal 0
```

## Sample
Generating an 10 by 10 MNIST digits image sampled from a trained model(assuming the one above).
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


## Troubleshooting
**I got a weird cannot convert int to float error. ``TypeError: Cannot convert Type TensorType(float32, matrix) (of Variable Subtensor{int64:int64:}.0) into Type TensorType(float64, matrix)``**

Have you [configured theano](http://deeplearning.net/software/theano/library/config.html#envvar-THEANORC)?
Here is my .theanorc config (use cpu if you do not have a CUDA capable gpu):
```
[global]
device = gpu
floatX = float32
exception_verbosity=high

[nvcc]
fastmath = True
```


**I got an IO error about params.npz. ``IOError: [Errno 2] No such file or directory: './experiments/ef0f220f13f115d34798a5f7e16e8c539e4eaee42564d07ed8f893b7fadaa8a0/params.npz'``**

You are probably trying to resume a failed experiment. Try the --force flag to override the previous one or simply delete the experiments folder.
