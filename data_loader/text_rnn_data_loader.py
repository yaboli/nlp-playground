from base.keras.base_data_loader import BaseDataLoader
import tensorflow as tf
import tensorflow_datasets as tfds
from utils.utils import write_binary
import os


class TextRnnDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(TextRnnDataLoader, self).__init__(config)
        self.train_data, self.test_data = self.load_data()

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def load_data(self):
        dataset, info = tfds.load(self.config.data_loader.dataset, with_info=True,
                                  as_supervised=True)

        data_dir = self.config.data_loader.data_dir
        encoder_path = os.path.join(data_dir, self.config.data_loader.encoder)
        encoder = info.features['text'].encoder
        write_binary(encoder, encoder_path)

        train_dataset, test_dataset = dataset['train'], dataset['test']

        train_dataset = train_dataset.shuffle(self.config.data_loader.buffer_size)
        train_dataset = train_dataset.padded_batch(self.config.trainer.batch_size,
                                                   tf.compat.v1.data.get_output_shapes(train_dataset))

        test_dataset = test_dataset.padded_batch(self.config.trainer.batch_size,
                                                 tf.compat.v1.data.get_output_shapes(test_dataset))

        return train_dataset, test_dataset
