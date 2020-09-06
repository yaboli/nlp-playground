from models.char_rnn_model import CharRnnModel
from utils.config import process_config
from utils.utils import get_args
import tensorflow as tf


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    print('Create the model.')
    model = CharRnnModel(config, mode='infer')
    model.load(tf.train.latest_checkpoint(config.callbacks.checkpoint_dir))

    print("\n" + "*" * 20 + " Generated Text " + "*" * 20)
    start_string = u"ROMEO: "
    print(model.generate_text(start_string))
    print("*" * 20 + " The End " + "*" * 20 + "\n")


if __name__ == '__main__':
    main()
