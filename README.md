# KMML challenge
This repository contains all the code used for the [Challenge](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021/overview/evaluation) of the course MVA Kernel Methods for Machine learning. The team was composed of Loïc Chadoutaud and Jules Samaran. The goal of the challenge was to predict whether a DNA sequence region is binding site to a specific transcription factor with kernel methods. The idea was to implement everything without using any machine learning library.

We implemented three algorithms:
- Kernel Ridge Regression (KRR)
- Kernel Logistic Regression (KLR)
- Support Vector Machine (SVM)

You can use the following kernels:
- Linear
- Gaussian
- Spectrum k
- Mismatch k-m

k corresponds to the length of subsequences considered and m to the possible number of mismatch in the subsequences. For the last two kernels, we preferred to compute the features of the samples projected in the corresponding RKHS and save it because it takes a lot of time to compute. The features are saved in the *data/processed folder*. As they are considered like features, if you want to use it, you should use the linear kernel but you could also try with the gaussian kernel. 

As an example, the code is ready to make predictions with a mismatch kernel 7-2 and a KRR algorithm by running the main.py file after following the getting started section. 

# Getting started

## Requirements

The easiest way to use our code is to create a new virtual environment, clone this repository and to install the required packages with the requirements.txt files

## Directory structure 

Make sure to have the following architecture for the folders. 

```bash
data
├── original
│   ├── Xte0.csv
│   ├── Xte0_mat100.csv
│   ├── Xte1.csv
│   ├── Xte1_mat100.csv
│   ├── Xte2.csv
│   ├── Xte2_mat100.csv
│   ├── Xtr0.csv
│   ├── Xtr0_mat100.csv
│   ├── Xtr1.csv
│   ├── Xtr1_mat100.csv
│   ├── Xtr2.csv
│   ├── Xtr2_mat100.csv
│   ├── Ytr0.csv
│   ├── Ytr1.csv
│   └── Ytr2.csv
└── processed
```

You have to put the original files as shown on the above tree in the folder *data/original*, the folder *data/processed* is here to save features that take a long time to compute.

## How to run the code

To run the code, you have to choose which algorithm you want to use among Kernel Ridge Regression, Kernel Logistic Regression and Kernel Support Vector Machines by specifying it in the following line of the main.py file. For instance, in the exemple below, KRR will be used. Instead of *cfg_krr.yaml* it should be *cfg_klr.yaml* or *cfg_svm.yaml*
```
cfg_paths = [os.path.join(os.getcwd(), "cfgs", "cfg_krr.yaml")]
```
All the required parameters to run the code are in the files of the type "cfg_krr.yaml" in the cfgs folder. 

Below is a example of a config file
```
---
MODEL_NAME: KRR

grid_hparams:
  lamb: [0.0001, 0.001, 0.1, 1.]

DATA:
  k_list: ["gaussian"]
  type_list: ["mismatching2-7"]
  scale: [false]

N_FOLDS: 5

DESCRIPTION: ""
```

You can change the number of arguments of *k_list*, *type_list* and *scale* by changing the number of elements in the list. Make sure that the length of the three lists are the same. 

Possible arguments for those lists are:
- k_list: "linear", "gaussian"
- type_list: "mat100", "spectrum{k}", "mismatching{m}-{k}"
- scale: true, false

Finally, run the main.py file and your submission will be saved in the submission folder. 
