from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
#import time # get an idea of time required to preprocess
import os # for loading the GLOVE file
import load_imdb_data # load the imdb reviews

def load_and_process_data(data_path, glove_path, glove_dim, max_num_words,
    max_seq_length):

    print('Loading dataset')
    x_train, y_train, x_test, y_test = load_imdb_data.load_imdb_dataset(data_path)

    print('Tokenizing texts')
    tokenizer = Tokenizer(num_words=max_num_words,
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',
        lower=True, split=' ', char_level=False, oov_token=None, document_count=0)

    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(x_train)
    train_sequences = pad_sequences(train_sequences, maxlen=max_seq_length)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    test_sequences = pad_sequences(test_sequences, maxlen=max_seq_length)

    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    print('Loading word vectors')
    # from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    embeddings_index = {}
    glove_file = os.path.join(glove_path, "glove.6B." + str(glove_dim) + "d.txt")
    f = open(os.path.join(glove_path, glove_file), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, glove_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return train_sequences, y_train, test_sequences, y_test, word_index, embedding_matrix
