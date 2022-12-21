import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, BatchNormalization, GRU
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv1D

def select_model(model, model_name):
    list_model = ["biLSTM_model", "LSTM_model", "GRU_model", "biGRU_model", "noRNN_model"]
    if model_name == list_model[0]: return biLSTM_model(model)
    if model_name == list_model[2]: return LSTM_model(model)
    if model_name == list_model[3]: return GRU_model(model)
    if model_name == list_model[4]: return biGRU_model(model)
    if model_name == list_model[5]: return noRNN_model(model)
    

def biLSTM_model(model):
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def LSTM_model(model):
    model.add(LSTM(40, return_sequences=True))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def biGRU_model(model):
    model.add(Bidirectional(GRU(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def GRU_model(model):
    model.add(GRU(40, return_sequences=True))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def noRNN_model(model):
    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model