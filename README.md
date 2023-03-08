# VirExplorer 
## Introduction

VirExplorer is a pipeline used to predict if genomic sequences obtained by a next-generation sequencer, for instance, Illumina, of short length belong to a virus or a human DNA.

It is a binary classifier implemented with a Deep Learning model based on a multilayer Convolutional Neural Network in Keras TensorFlow. The pipeline comprises four scripts that can be run interactively or from the command line that create the databases for the training, encode the data, train the model, and run predictions.

## Environment requirements

VirExplorer has been developed on Ubuntu 20.04 and equipped with an Nvidia RTX-1090Ti GPU, and it requires the following components and packages:

1. Installation of the package bbmap for Ubuntu (sudo apt-get install bbmap)
2. A virtual environment with the latest versions of Python, TensorFlow, biopython, scikit-learn, NumPy, matplotlib, scikit-plot, and seaborn installed.

## Repository Content

Repository Content
This repository contains the following:
1. The four Python scripts comprising the pipeline (dataste_prep.py, encode.py, training.py, VirExlporer.py)
2. 'datasets' folder: contains two examples of files for the reference data needed to train the model on virus and human host genome.
3. 'data_train' folder: an example of the output of the dataset_prep.py and encode.py scripts using the files from the 'datasets' folder.
4. 'data_test' folder: this folder is used to store the file with the sequences to be classified. There are also some sample files that can be used to test the scripts.
5. 'pretrained_model' folder: contains a model trained on a dataset containing about 1,5 million samples of 150bp extracted randomly from the HG38 reference genome and the same number of sequences from a curated human virus dataset obtained from the NCBI data. The model has not been completely optimized yet.
6. 'plots' folder: contains the pre-trained model plots for accuracy, loss, confusion matrix, and auc-roc. It also contains the console output with the metrics: F1 score, Precision, Recall, and Brier Score.

## Previous version of VirExplorer video presentation

VirExplorer was originally developed for my capstone project to classify genome sequences belonging to bacterias and their viruses. Here following there is the link to the youTube video presentation of the project (13 min).

(https://youtu.be/pl0N43weySo "VirExplorer Video Presentation")



