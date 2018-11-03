import preprocess_data
import imdb_rnn
import os
from itertools import product

DATASET_PATH = "D:/datasets/"
GLOVE_PATH = "D:/datasets/glove/"
GLOVE_DIM = 100
#MAX_NB_WORDS = 400000
MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 1000
NUM_EPOCHS = [15, 25]
BATCH_SIZE = [128, 256]
STATE_DIM = [20, 50, 100, 200, 500]

#def preprocess_data(data_path, glove_path, glove_dim, max_num_words, max_seq_length, embed_dim):
x_train, y_train, x_test, y_test, word_index, embed_matrix = preprocess_data.load_and_process_data(DATASET_PATH, GLOVE_PATH, GLOVE_DIM,
    MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

results_rnn = []
results_lstm = []
num_dense = 32
glove_dim = GLOVE_DIM
#num_epochs = NUM_EPOCHS[0]
#batch_size = BATCH_SIZE[0]
#state_dim = STATE_DIM[0]
max_seq_length = MAX_SEQUENCE_LENGTH

for num_epochs, batch_size, state_dim in product(NUM_EPOCHS, BATCH_SIZE, STATE_DIM):
    print("RNN Test: Epochs {}, Batch_size {}, State_dim {}".format(num_epochs, batch_size, state_dim))
    results_rnn.append(imdb_rnn.build_and_test_rnn(x_train, y_train, x_test, y_test, word_index,
        max_seq_length, glove_dim, embed_matrix, state_dim, num_dense, num_epochs, batch_size))
	
    with open('rnn_test_2.log', 'a') as f:
        f.write("%s\n" % results_rnn[-1])
