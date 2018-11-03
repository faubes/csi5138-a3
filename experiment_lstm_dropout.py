# test dropout rates

import preprocess_data
import imdb_rnn_lstm
import os
from itertools import product

DATASET_PATH = "/home/joel/datasets/"
GLOVE_PATH = "/home/joel/datasets/glove"
GLOVE_DIM = 100
#MAX_NB_WORDS = 400000
MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 1000

DROPOUT = [0.01, 0.1, 0.5, 0.2]

#def preprocess_data(data_path, glove_path, glove_dim, max_num_words, max_seq_length, embed_dim):
x_train, y_train, x_test, y_test, word_index, embed_matrix = preprocess_data.load_and_process_data(DATASET_PATH, GLOVE_PATH, GLOVE_DIM,
    MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

results_lstm = []
num_dense = 64
glove_dim = GLOVE_DIM
num_epochs = 10
batch_size = 64
state_dim = 200
max_seq_length = MAX_SEQUENCE_LENGTH


for dropout in DROPOUT:
    print("LSTM Dropout Test: ".format(dropout))

    results_lstm.append(imdb_rnn_lstm.build_and_test_lstm_dropout(x_train, y_train, x_test, y_test, word_index,
        max_seq_length, glove_dim, embed_matrix, state_dim, num_dense, num_epochs, batch_size, dropout))

    with open('lstm_dropout_test.log', 'a') as f:
        f.write("%s\n" % results_lstm[-1])
