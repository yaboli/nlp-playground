from base.keras.base_model import BaseModel
import tensorflow as tf
from utils.utils import load_binary, pad_to_size
import os


class TextRnnModel(BaseModel):
    def __init__(self, config, mode='training'):
        super(TextRnnModel, self).__init__(config)
        self.mode = mode
        self.encoder = None
        self.init_vocab()
        self.build_model()

    def init_vocab(self):
        data_dir = self.config.data_loader.data_dir
        assert os.path.isdir(data_dir), "Data directory does not exist!"

        encoder_path = os.path.join(data_dir, self.config.data_loader.encoder)
        self.encoder = load_binary(encoder_path)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.encoder.vocab_size, self.config.model.embedding_dim),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.config.model.rnn_units)),
            tf.keras.layers.Dense(self.config.model.hidden_dim_1, activation='relu'),
            tf.keras.layers.Dense(self.config.model.hidden_dim_2)
        ])

        if self.mode == 'training':
            self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                               optimizer=tf.keras.optimizers.Adam(self.config.model.learning_rate),
                               metrics=['accuracy'])

    def sample_predict(self, sample_pred_text, pad):
        encoded_sample_pred_text = self.encoder.encode(sample_pred_text)

        if pad:
            encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
        encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
        predictions = self.model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

        return (predictions)
