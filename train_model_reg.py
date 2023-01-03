import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from cnn_model import select_model
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding

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

# function for load dataset x (23 integer code) and dataset y (label)
def load_data(dataset):
    data_x = []
    data_y = []

    for data in dataset:
        label = np.float(data[3])
        data_y.append(label)
        integer_code = [int(i) for i in data[4:]]
        data_x.append(integer_code)
    
    return data_x,data_y

# load positive and negative dataset
positive_hek293t, negative_hek293t = pos_neg_data("encoded_hek293t.csv")
positive_K562, negative_K562 = pos_neg_data("encoded_K562.csv")

# Split x and y dataset
x_positive_hek293t, y_positive_hek293t = load_data(positive_hek293t)
x_negative_hek293t, y_negative_hek293t = load_data(negative_hek293t)
x_positive_K562, y_positive_K562 = load_data(positive_K562)
x_negative_K562, y_negative_K562 = load_data(negative_K562)

# Split dataset with test ratio = 0.2
train_x_positive_hek293t, test_x_positive_hek293t, train_y_positive_hek293t, test_y_positive_hek293t = train_test_split(x_positive_hek293t,y_positive_hek293t,test_size=0.2,random_state=50) 

train_x_negative_hek293t, test_x_negative_hek293t, train_y_negative_hek293t, test_y_negative_hek293t = train_test_split(x_negative_hek293t,y_negative_hek293t,test_size=0.2,random_state=50) 

train_x_positive_K562, test_x_positive_K562, train_y_positive_K562, test_y_positive_K562 = train_test_split(x_positive_K562,y_positive_K562,test_size=0.2) 

train_x_negative_K562, test_x_negative_K562, train_y_negative_K562, test_y_negative_K562 = train_test_split(x_negative_K562,y_negative_K562,test_size=0.2) 

# Dataset train
dataset_x = train_x_positive_hek293t + train_x_negative_hek293t + train_x_positive_K562 + train_x_negative_K562

dataset_y = train_y_positive_hek293t + train_y_negative_hek293t + train_y_positive_K562 + train_y_negative_K562

dataset_x_train, dataset_x_valid, dataset_y_train, dataset_y_valid = train_test_split(dataset_x,dataset_y,test_size=0.2,random_state=50)

# Dataset test
x_test_hek293t = test_x_positive_hek293t + test_x_negative_hek293t
x_test_K562 = test_x_positive_K562 + test_x_negative_K562

y_test_hek293t = test_y_positive_hek293t + test_y_negative_hek293t
y_test_K562 = test_y_positive_K562 + test_y_negative_K562

dataset_x_test = x_test_hek293t + x_test_K562
dataset_y_test = y_test_hek293t + y_test_K562


# Embedding

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

# Regression Train

# word2vec, biLSTM
model_reg_1 = Sequential()
model_reg_1.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model_reg_1 = select_model(model_reg_1, "biLSTM_model_reg")
model_reg_1.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model_reg_1.fit(x = dataset_x_train, y=dataset_y_train, epochs = EPOCHS, verbose=1, validation_data=(dataset_x_valid, dataset_y_valid), batch_size = BATCH_SIZE, shuffle=True)


# word2vec, LSTM
model_reg_2 = Sequential()
model_reg_2.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model_reg_2 = select_model(model_reg_2, "LSTM_model")
model_reg_2.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model_reg_2.fit(x = dataset_x_train, y=dataset_y_train, epochs = EPOCHS, verbose=1, validation_data=(dataset_x_valid, dataset_y_valid), batch_size = BATCH_SIZE, shuffle=True)


# word2vec, GRU
model_reg_3 = Sequential()
model_reg_3.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model_reg_3 = select_model(model_reg_3, "GRU_model")
model_reg_3.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model_reg_3.fit(x = dataset_x_train, y=dataset_y_train, epochs = EPOCHS, verbose=1, validation_data=(dataset_x_valid, dataset_y_valid), batch_size = BATCH_SIZE, shuffle=True)


# word2vec, biGRU
model_reg_4 = Sequential()
model_reg_4.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model_reg_4 = select_model(model_reg_4, "biGRU_model")
model_reg_4.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model_reg_4.fit(x = dataset_x_train, y=dataset_y_train, epochs = EPOCHS, verbose=1, validation_data=(dataset_x_valid, dataset_y_valid), batch_size = BATCH_SIZE, shuffle=True)


# word2vec, noRNN
model_reg_5 = Sequential()
model_reg_5.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model_reg_5 = select_model(model_reg_5, "noRNN_model")
model_reg_5.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model_reg_5.fit(x = dataset_x_train, y=dataset_y_train, epochs = EPOCHS, verbose=1, validation_data=(dataset_x_valid, dataset_y_valid), batch_size = BATCH_SIZE, shuffle=True)


# regression score
score_1 = model_reg_1.evaluate(dataset_x_test, dataset_y_test)
print('Test Score Model 1')
print('Test Loss: ', score_1[0])
print('Test accuracy: ', score_1[1])

score_2 = model_reg_2.evaluate(dataset_x_test, dataset_y_test)
print('Test Score Model 2')
print('Test Loss: ', score_2[0])
print('Test accuracy: ', score_2[1])

score_3 = model_reg_3.evaluate(dataset_x_test, dataset_y_test)
print('Test Score Model 3')
print('Test Loss: ', score_3[0])
print('Test accuracy: ', score_3[1])

score_4 = model_reg_4.evaluate(dataset_x_test, dataset_y_test)
print('Test Score Model 4')
print('Test Loss: ', score_4[0])
print('Test accuracy: ', score_4[1])

score_5 = model_reg_5.evaluate(dataset_x_test, dataset_y_test)
print('Test Score Model 5')
print('Test Loss: ', score_5[0])
print('Test accuracy: ', score_5[1])
