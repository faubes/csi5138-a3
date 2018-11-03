import preprocess_data
import os
from itertools import product
import keras
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import SimpleRNN
#from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.models import Model
from keras.models import Sequential
from keras.initializers import Constant
import time

DATASET_PATH = "D:/datasets/"
GLOVE_PATH = "D:/datasets/glove"
GLOVE_DIM = 200
#MAX_NB_WORDS = 400000
MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE = 256
NUM_EPOCHS = 25
STATE_DIM = 500
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.1
#def preprocess_data(data_path, glove_path, glove_dim, max_num_words, max_seq_length, embed_dim):
x_train, y_train, x_test, y_test, word_index, embed_matrix = preprocess_data.load_and_process_data(DATASET_PATH, GLOVE_PATH, GLOVE_DIM,
    MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)


results_lstm = []
num_dense = 64
glove_dim = GLOVE_DIM
num_epochs = NUM_EPOCHS
batch_size = BATCH_SIZE
state_dim = STATE_DIM
max_seq_length = MAX_SEQUENCE_LENGTH
learn_rate = LEARNING_RATE
dropout_rate = DROPOUT_RATE

print("LSTM Test: Epochs {}, Batch_size {}, State_dim {}, lr {}".format(num_epochs, batch_size, state_dim, learn_rate))
optimizer = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
model = Sequential()
model.add(Embedding(len(word_index) + 1, glove_dim,
	embeddings_initializer=Constant(embed_matrix),
	input_length=max_seq_length,
	trainable=False))
regularizer = keras.regularizers.l1(0.01)
model.add(CuDNNLSTM(state_dim, kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer,
	activity_regularizer=regularizer))
model.add(Dropout(dropout_rate))
#model.add(LSTM(state_dim, activation='tanh'))
model.add(Dense(num_dense, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
			  optimizer=optimizer,
			  metrics=['acc'])

start_fit = time.time()
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
result = model.evaluate(x_test, y_test)
fit_time = time.time() - start_fit
result = ([glove_dim, state_dim, batch_size, model.count_params(),
				history.history['loss'][-1], history.history['acc'][-1],
				result[0], result[1], fit_time])
results_lstm.append(result)

with open('cudnn_lstm_dropout_test.log', 'a') as f:
	f.write("%s"%datetime.now())
	f.write("%s\n" % results_lstm_model[-1])
