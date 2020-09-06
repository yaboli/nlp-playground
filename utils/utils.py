import argparse
import pickle


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def write_binary(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_binary(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
