# Build and test RNN models on imdb reviews
#
# Uses code from
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# Joel Faubert
# 2560106
# CSI5138 HW3
# Ottawa University
# Fall 2018
#



import keras
from keras.layers import Flatten, Dense, Input, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Dropout
#from keras.layers import CuDNNLSTM
from keras.models import Model
from keras.models import Sequential
from keras.initializers import Constant
import time

def build_and_test_rnn(x_train, y_train, x_test, y_test, word_index, max_seq_length,
    glove_dim, embed_matrix, state_dim, num_dense, num_epochs, batch_size):

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, glove_dim,
        embeddings_initializer=Constant(embed_matrix),
        input_length=max_seq_length, mask_zero=True,
        trainable=False))
    model.add(SimpleRNN(state_dim, activation='tanh'))
    model.add(Dense(num_dense, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    start_fit = time.time()
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
    result = model.evaluate(x_test, y_test)
    fit_time = time.time() - start_fit
    result = ([num_epochs, state_dim, batch_size, model.count_params(),
                    history.history['loss'][-1], history.history['acc'][-1],
                    result[0], result[1], fit_time])
    return result

def build_and_test_lstm(x_train, y_train, x_test, y_test, word_index, max_seq_length,
    glove_dim, embed_matrix, state_dim, num_dense, num_epochs, batch_size):

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, glove_dim,
        embeddings_initializer=Constant(embed_matrix),
        input_length=max_seq_length, mask_zero=True,
        trainable=False))
    #model.add(CuDNNLSTM(state_dim, activation='tanh'))
    model.add(LSTM(state_dim, activation='tanh'))
    model.add(Dense(num_dense, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    start_fit = time.time()
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
    result = model.evaluate(x_test, y_test)
    fit_time = time.time() - start_fit
    result = ([glove_dim, state_dim, batch_size, model.count_params(),
                    history.history['loss'][-1], history.history['acc'][-1],
                    result[0], result[1], fit_time])
    return result

def build_and_test_lstm_dropout(x_train, y_train, x_test, y_test, word_index, max_seq_length,
    glove_dim, embed_matrix, state_dim, num_dense, num_epochs, batch_size, dropout):

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, glove_dim,
        embeddings_initializer=Constant(embed_matrix),
        input_length=max_seq_length, mask_zero=True,
        trainable=False))
    #model.add(CuDNNLSTM(state_dim, activation='tanh'))
    model.add(LSTM(state_dim, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(num_dense, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['acc'])

    start_fit = time.time()
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
    result = model.evaluate(x_test, y_test)
    fit_time = time.time() - start_fit
    result = ([dropout, history.history['loss'][-1],
        history.history['acc'][-1], result[0], result[1], fit_time])
    return result
