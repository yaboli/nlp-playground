from data_loader.text_rnn_data_loader import TextRnnDataLoader
from models.text_rnn_model import TextRnnModel
from trainers.text_rnn_trainer import TextRnnModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([
        config.data_loader.data_dir,
        config.callbacks.tensorboard_log_dir,
        config.callbacks.checkpoint_dir
    ])

    print('Create the data generator.')
    data_loader = TextRnnDataLoader(config)

    print('Create the model.')
    model = TextRnnModel(config)

    print('Create the trainer')
    data = (data_loader.get_train_data(), data_loader.get_test_data())
    trainer = TextRnnModelTrainer(model.model, data, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
