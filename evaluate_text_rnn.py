from models.text_rnn_model import TextRnnModel
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
    model = TextRnnModel(config, mode='infer')
    model.load(tf.train.latest_checkpoint('experiments/2020-09-07/text_rnn/checkpoints'))

    sample_pred_text = ('The movie was cool. The animation and the graphics '
                        'were out of this world. I would recommend this movie.')

    # predict on a sample text without padding.
    predictions = model.sample_predict(sample_pred_text, pad=False)
    print(predictions)

    # predict on a sample text with padding
    predictions = model.sample_predict(sample_pred_text, pad=True)
    print(predictions)


if __name__ == '__main__':
    main()
