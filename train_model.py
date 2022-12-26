import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from cnn_model import select_model
# function for split positive (label=1) and negative (label=0) dataset
def pos_neg_data(file):
    dataset = []
    positive = []
    negative = []

    file = open(file)
    for line in file:
        data = [i for i in line.strip().split(',')]
        label = data[3]
        if label == "1":
            positive.append(data)
        else:
            negative.append(data)
        dataset.append(data)
    
    return positive,negative

# function for load dataset x (23 integer code) and dataset y (label) for regression scheme
def load_data_reg(file):
    data_x = []
    data_y = []

    file = open(file)
    for line in file:
        data = [i for i in line.strip().split(',')]
        label = np.float(data[3])
        data_y.append(label)
        integer_code = [int(i) for i in data[4:]]
        data_x.append(integer_code)
    
    return data_x,data_y

# function for load dataset x (23 integer code) and dataset y (label) for classification scheme
def load_data_class(file):
    data_x = []
    data_y = []

    file = open(file)
    for line in file:
        data = [i for i in line.strip().split(',')]
        label = data[3]
        data_y.append(label)
        integer_code = [int(i) for i in data[4:]]
        data_x.append(integer_code)
    
    return data_x,data_y

# load positive and negative dataset
positive, negative = pos_neg_data("encoded_all_data.csv")
positive_hek293t, negative_hek293t = pos_neg_data("encoded_hek293t.csv")
positive_K562, negative_K562 = pos_neg_data("encoded_K562.csv")

# load data regression scheme
reg_dataset_x, reg_dataset_y = load_data_reg("encoded_all_data.csv")
reg_hek293t_x, reg_hek293t_y = load_data_reg("encoded_hek293t.csv")
reg_K562_x, reg_K562_y = load_data_reg("encoded_K562.csv")

# load data classification scheme
class_dataset_x, class_dataset_y = load_data_class("encoded_all_data.csv")
class_hek293t_x, class_hek293t_y = load_data_class("encoded_hek293t.csv")
class_K562_x, class_K562_y = load_data_class("encoded_K562.csv")

# Split dataset with test ratio = 0.2

# Split dataset for regression scheme
reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(reg_dataset_x,reg_dataset_y,test_size=0.2,random_state=50) 

reg_hek293t_x_train, reg_hek293t_x_test, reg_hek293t_y_train, reg_hek293t_y_test = train_test_split(reg_hek293t_x,reg_hek293t_y,test_size=0.2,random_state=50) 

reg_K562_x_train, reg_K562_x_test, reg_K562_y_train, reg_K562_y_test = train_test_split(reg_K562_x,reg_K562_y,test_size=0.2,random_state=50)

# Split dataset for classification scheme
class_x_train, class_x_test, class_y_train, class_y_test = train_test_split(class_dataset_x,class_dataset_y,test_size=0.2,random_state=50) 

class_hek293t_x_train, class_hek293t_x_test, class_hek293t_y_train, class_hek293t_y_test = train_test_split(class_hek293t_x,class_hek293t_y,test_size=0.2,random_state=50) 

class_K562_x_train, class_K562_x_test, class_K562_y_train, class_K562_y_test = train_test_split(class_K562_x,class_K562_y,test_size=0.2,random_state=50)

# Embedding

from gensim.models import KeyedVectors, Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from model_eval import roc_auc, model_rep, model_matrix, GetKS
from tensorflow.keras.losses import  binary_crossentropy
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Parameter
EPOCHS = 1
BATCH_SIZE = 25
VECTOR_SIZE = 100
WINDOW = 5
# Word2Vec
def count_w2v_weight(data_x_train):
    model_w2v = Word2Vec(sentences=data_x_train, vector_size=VECTOR_SIZE, window=WINDOW, min_count=1)
    model_w2v.train(data_x_train, total_examples=len(data_x_train), epochs=EPOCHS)

    w2v_weights =  np.zeros((len(model_w2v.wv), 100))
    for i in range(len(model_w2v.wv)):
        w2v_weights[i, :] = model_w2v.wv[i]
    return w2v_weights

#Regression Train
model = Sequential()
model.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(reg_x_train)],
        trainable=True))
model = select_model(model, "GRU_model")
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x = reg_x_train, y=reg_y_train, epochs = EPOCHS, verbose=1, batch_size = BATCH_SIZE, shuffle=True)
reg_y_pred = model.predict(reg_x_test)
print(reg_y_pred)

#Classification Train

# model = Sequential()
# model.add(Embedding(
#         input_dim=16,
#         output_dim=100,
#         input_length=23,
#         weights=[count_w2v_weight(class_x_train)],
#         trainable=True))
# model = select_model(model, "GRU_model")
# model.compile(loss=binary_crossentropy,
#               optimizer='adam', metrics=['acc'])
# model.fit(x = class_x_train, y=class_y_train, epochs = EPOCHS, verbose=1, batch_size = BATCH_SIZE, shuffle=True)
# class_y_pred = model.predict(class_x_test)
# print(class_y_pred)

#Keras
# model = Sequential()
# model.add(Embedding(input_dim=16, output_dim=256, input_length=23))
# model = select_model(model, "GRU_model")
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# model.fit(x = reg_x_train, y=reg_y_train, epochs = EPOCHS, verbose=1, batch_size = BATCH_SIZE, shuffle=True)


# score = model.evaluate(reg_x_test, reg_y_test)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])


