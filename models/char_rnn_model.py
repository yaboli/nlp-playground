from base.keras.base_model import BaseModel
import tensorflow as tf
from utils.utils import load_binary
import os


class CharRnnModel(BaseModel):
    def __init__(self, config, mode='training'):
        super(CharRnnModel, self).__init__(config)
        self.vocab_size = 0
        self.char2idx = None
        self.idx2char = None
        self.init_vocab()
        self.mode = mode
        self.build_model()

    def init_vocab(self):
        data_dir = self.config.data_loader.data_dir
        assert os.path.isdir(data_dir), "Data directory does not exist!"

        vocab_path = os.path.join(data_dir, self.config.data_loader.vocab)
        vocab = load_binary(vocab_path)
        self.vocab_size = len(vocab)

        char2idx_path = os.path.join(data_dir, self.config.data_loader.char2idx)
        idx2char_path = os.path.join(data_dir, self.config.data_loader.idx2char)
        self.char2idx = load_binary(char2idx_path)
        self.idx2char = load_binary(idx2char_path)

    def build_model(self):

        vocab_size = self.vocab_size
        batch_size = self.config.trainer.batch_size if self.mode == 'training' else 1
        embedding_dim = self.config.model.embedding_dim
        rnn_units = self.config.model.rnn_units

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])

        if self.mode == 'training':
            def loss(labels, logits):
                return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

            self.model.compile(optimizer=self.config.model.optimizer,
                               loss=loss)

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

        self.model.build(tf.TensorShape([1, None]))

    def generate_text(self, start_string, num_generate=1000, temperature=1.0):
        """
        :param start_string: Start text
        :param num_generate: Number of characters to generate
        :param temperature: Low temperatures results in more predictable text.
                            Higher temperatures results in more surprising text.
                            Experiment to find the best setting.
        :return: Generated text
        """
        # Evaluation step (generating text using the learned model)

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        return start_string + ''.join(text_generated)
