import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import  binary_crossentropy
from sklearn.model_selection import train_test_split
from cnn_model import select_model
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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

# One Hot Encoding for Dataset y for Classification scheme
class_y_train = keras.utils.to_categorical(dataset_y_train)
class_y_test = keras.utils.to_categorical(dataset_y_test)
class_y_valid = keras.utils.to_categorical(dataset_y_valid)
class_y_test_hek293t = keras.utils.to_categorical(y_test_hek293t)
class_y_test_K562 = keras.utils.to_categorical(y_test_K562)

# Parameter
EPOCHS = 25
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

#Classification Train

# word2vec, biLSTM
model1 = Sequential()
model1.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model1 = select_model(model1, "biLSTM_model")
model1.compile(loss=binary_crossentropy,optimizer='adam', metrics=['acc'])
model1.fit(x = np.array(dataset_x_train), y=np.array(class_y_train), epochs = EPOCHS, verbose=1, validation_data=(np.array(dataset_x_valid), np.array(class_y_valid)), batch_size = BATCH_SIZE, shuffle=True)

# word2vec, LSTM
model2 = Sequential()
model2.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model2 = select_model(model2, "LSTM_model")
model2.compile(loss=binary_crossentropy,optimizer='adam', metrics=['acc'])
model2.fit(x = np.array(dataset_x_train), y=np.array(class_y_train), epochs = EPOCHS, verbose=1, validation_data=(np.array(dataset_x_valid), np.array(class_y_valid)), batch_size = BATCH_SIZE, shuffle=True)

# word2vec, GRU
model3 = Sequential()
model3.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model3 = select_model(model3, "GRU_model")
model3.compile(loss=binary_crossentropy,optimizer='adam', metrics=['acc'])
model3.fit(x = np.array(dataset_x_train), y=np.array(class_y_train), epochs = EPOCHS, verbose=1, validation_data=(np.array(dataset_x_valid), np.array(class_y_valid)), batch_size = BATCH_SIZE, shuffle=True)

# word2vec, biGRU
model4 = Sequential()
model4.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model4 = select_model(model4, "biGRU_model")
model4.compile(loss=binary_crossentropy,optimizer='adam', metrics=['acc'])
model4.fit(x = np.array(dataset_x_train), y=np.array(class_y_train), epochs = EPOCHS, verbose=1, validation_data=(np.array(dataset_x_valid), np.array(class_y_valid)), batch_size = BATCH_SIZE, shuffle=True)

# word2vec, noRNN_model
model5 = Sequential()
model5.add(Embedding(
        input_dim=16,
        output_dim=100,
        input_length=23,
        weights=[count_w2v_weight(dataset_x_train)],
        trainable=True))
model5 = select_model(model5, "noRNN_model")
model5.compile(loss=binary_crossentropy,optimizer='adam', metrics=['acc'])
model5.fit(x = np.array(dataset_x_train), y=np.array(class_y_train), epochs = EPOCHS, verbose=1, validation_data=(np.array(dataset_x_valid), np.array(class_y_valid)), batch_size = BATCH_SIZE, shuffle=True)


# Test score
score_model1 = model1.evaluate(np.array(dataset_x_test), np.array(class_y_test))
print("Test Score model 1")
print('Test Loss: ', score_model1[0])
print('Test accuracy: ', score_model1[1])

score_model2 = model2.evaluate(np.array(dataset_x_test), np.array(class_y_test))
print("Test Score model 2")
print('Test Loss: ', score_model2[0])
print('Test accuracy: ', score_model2[1])

score_model3 = model3.evaluate(np.array(dataset_x_test), np.array(class_y_test))
print("Test Score model 3")
print('Test Loss: ', score_model3[0])
print('Test accuracy: ', score_model3[1])

score_model4 = model4.evaluate(np.array(dataset_x_test), np.array(class_y_test))
print("Test Score model 4")
print('Test Loss: ', score_model4[0])
print('Test accuracy: ', score_model4[1])

score_model5 = model5.evaluate(np.array(dataset_x_test), np.array(class_y_test))
print("Test Score model 5")
print('Test Loss: ', score_model5[0])
print('Test accuracy: ', score_model5[1])

# Test score hek293t
hek293t_score_model1 = model1.evaluate(np.array(x_test_hek293t), np.array(class_y_test_hek293t))
print("Test Score model 1 for dataset hek293t")
print('Test Loss: ', hek293t_score_model1[0])
print('Test accuracy: ', hek293t_score_model1[1])

hek293t_score_model2 = model2.evaluate(np.array(x_test_hek293t), np.array(class_y_test_hek293t))
print("Test Score model 2 for dataset hek293t")
print('Test Loss: ', hek293t_score_model2[0])
print('Test accuracy: ', hek293t_score_model2[1])

hek293t_score_model3 = model3.evaluate(np.array(x_test_hek293t), np.array(class_y_test_hek293t))
print("Test Score model 3 for dataset hek293t")
print('Test Loss: ', hek293t_score_model3[0])
print('Test accuracy: ', hek293t_score_model3[1])

hek293t_score_model4 = model4.evaluate(np.array(x_test_hek293t), np.array(class_y_test_hek293t))
print("Test Score model 4 for dataset hek293t")
print('Test Loss: ', hek293t_score_model4[0])
print('Test accuracy: ', hek293t_score_model4[1])

hek293t_score_model5 = model5.evaluate(np.array(x_test_hek293t), np.array(class_y_test_hek293t))
print("Test Score model 5 for dataset hek293t")
print('Test Loss: ', hek293t_score_model5[0])
print('Test accuracy: ', hek293t_score_model5[1])

# Test score K562
K562_score_model1 = model1.evaluate(np.array(x_test_K562), np.array(class_y_test_K562))
print("Test Score model 1 for dataset K562")
print('Test Loss: ', K562_score_model1[0])
print('Test accuracy: ', K562_score_model1[1])

K562_score_model2 = model2.evaluate(np.array(x_test_K562), np.array(class_y_test_K562))
print("Test Score model 2 for dataset K562")
print('Test Loss: ', K562_score_model2[0])
print('Test accuracy: ', K562_score_model2[1])

K562_score_model3 = model3.evaluate(np.array(x_test_K562), np.array(class_y_test_K562))
print("Test Score model 3 for dataset K562")
print('Test Loss: ', K562_score_model3[0])
print('Test accuracy: ', K562_score_model3[1])

K562_score_model4 = model4.evaluate(np.array(x_test_K562), np.array(class_y_test_K562))
print("Test Score model 4 for dataset K562")
print('Test Loss: ', K562_score_model4[0])
print('Test accuracy: ', K562_score_model4[1])

K562_score_model5 = model5.evaluate(np.array(x_test_K562), np.array(class_y_test_K562))
print("Test Score model 5 for dataset K562")
print('Test Loss: ', K562_score_model5[0])
print('Test accuracy: ', K562_score_model5[1])

# Prediction Result
class_y_pred1 = model1.predict(dataset_x_test)
class_y_pred2 = model2.predict(dataset_x_test)
class_y_pred3 = model3.predict(dataset_x_test)
class_y_pred4 = model4.predict(dataset_x_test)
class_y_pred5 = model5.predict(dataset_x_test)

# Predict dataset hek293t
hek293t_y_pred1 = model1.predict(x_test_hek293t)
hek293t_y_pred2 = model2.predict(x_test_hek293t)
hek293t_y_pred3 = model3.predict(x_test_hek293t)
hek293t_y_pred4 = model4.predict(x_test_hek293t)
hek293t_y_pred5 = model5.predict(x_test_hek293t)

# Predict dataset K562
K562_y_pred1 = model1.predict(x_test_K562)
K562_y_pred2 = model2.predict(x_test_K562)
K562_y_pred3 = model3.predict(x_test_K562)
K562_y_pred4 = model4.predict(x_test_K562)
K562_y_pred5 = model5.predict(x_test_K562)

# ROC_AUC score
roc_auc_1 = roc_auc_score(dataset_y_test, np.array(class_y_pred1[:,1]))
roc_auc_2 = roc_auc_score(dataset_y_test, np.array(class_y_pred2[:,1]))
roc_auc_3 = roc_auc_score(dataset_y_test, np.array(class_y_pred3[:,1]))
roc_auc_4 = roc_auc_score(dataset_y_test, np.array(class_y_pred4[:,1]))
roc_auc_5 = roc_auc_score(dataset_y_test, np.array(class_y_pred5[:,1]))

# ROC_AUC score hek293t
hek293t_roc_auc_1 = roc_auc_score(y_test_hek293t, np.array(hek293t_y_pred1[:,1]))
hek293t_roc_auc_2 = roc_auc_score(y_test_hek293t, np.array(hek293t_y_pred2[:,1]))
hek293t_roc_auc_3 = roc_auc_score(y_test_hek293t, np.array(hek293t_y_pred3[:,1]))
hek293t_roc_auc_4 = roc_auc_score(y_test_hek293t, np.array(hek293t_y_pred4[:,1]))
hek293t_roc_auc_5 = roc_auc_score(y_test_hek293t, np.array(hek293t_y_pred5[:,1]))

# ROC_AUC score K562
K562_roc_auc_1 = roc_auc_score(y_test_K562, np.array(K562_y_pred1[:,1]))
K562_roc_auc_2 = roc_auc_score(y_test_K562, np.array(K562_y_pred2[:,1]))
K562_roc_auc_3 = roc_auc_score(y_test_K562, np.array(K562_y_pred3[:,1]))
K562_roc_auc_4 = roc_auc_score(y_test_K562, np.array(K562_y_pred4[:,1]))
K562_roc_auc_5 = roc_auc_score(y_test_K562, np.array(K562_y_pred5[:,1]))

#print result roc auc
print('roc_auc model 1: ', roc_auc_1)
print('roc_auc model 2: ', roc_auc_2)
print('roc_auc model 3: ', roc_auc_3)
print('roc_auc model 4: ', roc_auc_4)
print('roc_auc model 5: ', roc_auc_5)

print('Dataset hek293t')
print('roc_auc model 1: ', hek293t_roc_auc_1)
print('roc_auc model 2: ', hek293t_roc_auc_2)
print('roc_auc model 3: ', hek293t_roc_auc_3)
print('roc_auc model 4: ', hek293t_roc_auc_4)
print('roc_auc model 5: ', hek293t_roc_auc_5)

print('Dataset K562')
print('roc_auc model 1: ', K562_roc_auc_1)
print('roc_auc model 2: ', K562_roc_auc_2)
print('roc_auc model 3: ', K562_roc_auc_3)
print('roc_auc model 4: ', K562_roc_auc_4)
print('roc_auc model 5: ', K562_roc_auc_5)

# label = ["biLSTM model", "LSTM model", "GRU model", "biGRU model", "noRNN model"]
# ROC_AUC_score = [0.84,0.72,0.73,0.86,0.74]

# ax = plt.axes()
# ax.set_facecolor("#F8F4EA")
# plt.bar(label, ROC_AUC_score, color=["#144272","#1D5997","#256DB8","#2D85E1","#4099F5"],edgecolor=("#0F3257"),linewidth=2, width=0.5)
# plt.xlabel("Model")
# plt.ylabel("ROC AUC")
# plt.title("ROC AUC Score")
# plt.savefig("roc_auc.jpg")
# plt.show()
