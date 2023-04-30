# Deep-Learning-CRISPR-Model (Word2VecCRISPR)
The details of this project are described in "Deep Learning-based Approach on sgRNA off-target Prediction in CRISPR/Cas9", presented at the International Conference on Computer Science, Information Technology & Engineering (ICCoSITE) in Jakarta - Indonesia, 16 February 2023

This project worked by Alyssa Imani and intented for Seminar coursework at Binus University.

# Background
Designing the appropriate sgRNA can improve the efficacy of CRISPR/Cas9 to make on-target knockout. 
Where on-target knockout means a successful making cleavage for insertion or deletion to target DNA. 
However, in implementing CIRSPR/Cas9 it is possible to get cleavage on a mismatched target site which is called off-target, 
which can cause genomic instability or cell death.To solve the challenge of designing sgRNA for CRISPR, 
this study will explain how the deep learning model can affect the prediction of sgRNA off-target (high specificity). 
The deep learning model combines Word2Vec embedding and a convolutional neural network (CNN). 
Different RNN models will also be included in the model to see if they will improve performance. 
Like the previous study, the prediction target was divided into on-target and off-target sgRNA. 
This project only focusing on the analysis of the sgRNA off-target prediction.

# Resources

Datasets and reference for CNN model architecture are taken from paper [Deep learning improves the ability of sgRNA off-target propensity prediction](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3395-z) 
and their [GitHub](https://github.com/LQYoLH/CnnCrispr).

# Description
This project is intended for study of implementing deep learning to a binary classification of sgRNA off-targetprediction. 
Five different models were compared in this study to get the best performance in classification prediction for off-target sgRNA. 
All models implement Word2Vec embedding to get better feature vector representation compared to the traditional one-hot encoding method. 
The five models were constructed by different RNN models: biLSTM, LSTM, GRU, biGRU and without RNN Layer (NoRNN). 
Each model was trained with a learning rate default of Adam optimizer 0.001 and tested with two different datasets: HEK293T dataset and K562 Dataset.

