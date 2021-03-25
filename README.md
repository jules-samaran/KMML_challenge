# KMML challenge
This repository contains all the code used for the [Challenge](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021/overview/evaluation) of the course MVA Kernel Methods for Machine learning. The goal of the challenge was to predict whether a DNA sequence region is binding site to a specific transcription factor with kernel methods. The idea was to implement everything without using any machine learning librairy.

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

You have to put the original files as shown on the above tree in the folder original, the folder processed is here to save features that are long to compute and to save time.

## How to run the code

To run the code, you have to choose which algorithm you want to use among Kernel Ridge Regression, Kernel Logistic Regression and Kernel Support Vector Machines by specifying it in the following line of the main.py file. For instance, in the exemple below, KRR will be used. 
```
cfg_paths = [os.path.join(os.getcwd(), "cfgs", "cfg_krr.yaml")]
```
All the required parameters to run the code are in the files of the type "cfg_krr.yaml" in the cfgs folder. 

Finally, run the main.py file and your submission will be saved in the submission folder. 


