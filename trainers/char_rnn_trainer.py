from base.keras.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class CharRnnModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(CharRnnModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, "ckpt_{epoch}"),
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        history = self.model.fit(
            self.data,
            epochs=self.config.trainer.num_epochs,
            callbacks=self.callbacks
        )
        self.loss.extend(history.history['loss'])
