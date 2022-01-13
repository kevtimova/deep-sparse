# Sparse Coding with Multi-Layer Decoders using Variance Regularization
This is a PyTorch implementation for the setup described in 
[Sparse Coding with Multi-Layer Decoders using Variance Regularization](https://arxiv.org/abs/2112.09214). 

### Requirements

- Python 3.7
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.6.0 with torchvision 0.7.0
- Other dependencies: numpy, tensorboardX

### Datasets

In our experiments, we use:
- the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. We provide the train and validation splits in 
```data/MNIST_train.npy``` and ```data/MNIST_val.npy```.
- a custom dataset with 200,000 gray-scale natural image patches of size 28x28 extracted from 
[ImageNet](https://www.image-net.org/index.php). The script to generate it is 
[build_imagenet_LCN.sh](https://github.com/kevtimova/deep-sparse/blob/main/scripts/build_ImageNet_LCN.sh).

### Training

The scripts below can be used to train sparse autoencoders with our different setups.

| dataset          | model    | script |
|------------------|----------|--------|
| MNIST            | SDL      | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/MNIST_SDL.sh)       |
| MNIST            | SDL-NL   | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/MNIST_SDL-NL.sh)       |
| MNIST            | VDL      | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/MNIST_VDL.sh)       |
| MNIST            | VDL-NL   | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/MNIST_VDL-NL.sh)       |
| ImageNet_patches | SDL      | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/ImageNet_SDL.sh)       |
| ImageNet_patches | SDL-NL   | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/ImageNet_SDL-NL.sh)       |
| ImageNet_patches | VDL      | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/ImageNet_VDL.sh)       |
| Imagenet_patches | VDL-NL   | [link](https://github.com/kevtimova/deep-sparse/blob/main/scripts/ImageNet_VDL-NL.sh)       |

### Evaluation

We evaluate our pre-trained sparse autoencoders on the downstream tasks of denoising (for MNIST and our custom 
ImageNet patches dataset) and classification in the low-data regime (for MNIST only).

#### Denoising

The denoising perfomance on the test set can be measured at the end of training by providing a list with levels of 
random noise (measured by std of Gaussian noise; the noise is added to the input images) in the ```noise``` argument 
in ```main.py```.

Alternatively, ```eval_denoising.py``` can be used given a pre-trained autoencoder.

#### Classification

To evaluate the linear separability of codes obtained from the sparse autoencoders, we provide the steps below.

Step 1: Given a pre-trained encoder, ```compute_codes.py``` can be used to create a dataset containing the codes 
for each MNIST image.

Step 2: Using the dataset from the previous step, ```eval_classification.py``` can be used to measure classification 
performance with a set number of training samples per class.

There are two options for the classifier - a linear classifier
(located in ```modles/linear_classifier.py```) and a classifier which uses a randomly initialized LISTA encoder module 
followed by a linear classification layer (located in ```modles/lista_classifier.py```).  
