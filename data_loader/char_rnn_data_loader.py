from base.keras.base_data_loader import BaseDataLoader
import tensorflow as tf
from utils.utils import write_binary
import os
import numpy as np


class CharRnnDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(CharRnnDataLoader, self).__init__(config)
        self.dataset = self.load_data()

    def get_train_data(self):
        return self.dataset

    def get_test_data(self):
        pass

    def load_data(self):
        data_dir = self.config.data_loader.data_dir
        path_to_file = tf.keras.utils.get_file(self.config.data_loader.file_name,
                                               self.config.data_loader.url)
        # Read, then decode for py2 compat.
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

        # The unique characters in the file
        vocab = sorted(set(text))
        vocab_path = os.path.join(data_dir, self.config.data_loader.vocab)
        write_binary(vocab, vocab_path)

        # Creating a mapping from unique characters to indices
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        char2idx_path = os.path.join(data_dir, self.config.data_loader.char2idx)
        idx2char_path = os.path.join(data_dir, self.config.data_loader.idx2char)
        write_binary(char2idx, char2idx_path)
        write_binary(idx2char, idx2char_path)

        text_as_int = np.array([char2idx[c] for c in text])

        # The maximum length sentence we want for a single input in characters
        seq_length = self.config.data_loader.seq_length

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)

        buffer_size = self.config.data_loader.buffer_size
        batch_size = self.config.trainer.batch_size
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        return dataset
