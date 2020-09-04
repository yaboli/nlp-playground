import tensorflow as tf

import numpy as np
import os
import time


class Dataset(object):

    def __init__(self, batch_size, buffer_size):
        self.path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def build_dataset(self):
        # Read, then decode for py2 compat.
        text = open(self.path_to_file, 'rb').read().decode(encoding='utf-8')
        # length of text is the number of characters in it
        print('Length of text: {} characters'.format(len(text)))

        # Take a look at the first 250 characters in text
        print(text[:250])

        # The unique characters in the file
        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        # Creating a mapping from unique characters to indices
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        print('{')
        for char, _ in zip(char2idx, range(20)):
            print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('  ...\n}')

        # The maximum length sentence we want for a single input in characters
        seq_length = 100
        examples_per_epoch = len(text) // (seq_length + 1)

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        for input_example, target_example in dataset.take(1):
            print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
            print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

            for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
                print("Step {:4d}".format(i))
                print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
                print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

        dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)

        return dataset


class CharRNN(object):
    def __init__(self, dataset):
        self.batch_size = dataset.batch_size
        self.vocab = dataset.vocab
        # Length of the vocabulary in chars
        self.vocab_size = len(self.vocab)
        # The embedding dimension
        self.embedding_dim = 256
        # Number of RNN units
        self.rnn_units = 1024

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim,
                                      batch_input_shape=[self.batch_size, None]),
            tf.keras.layers.GRU(self.rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(self.vocab_size)
        ])
        return model


if __name__ == '__main__':
    pass
