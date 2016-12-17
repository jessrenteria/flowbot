import argparse
from utils.config import get_config
from data.preprocessor import Preprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', default='lstm_attention')
    args = parser.parse_args()

    config = get_config(args.bot)
    config['reuse_data'] = False
    p = Preprocessor(config)

if __name__ == '__main__':
    main()
