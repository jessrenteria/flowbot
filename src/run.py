import argparse

from utils.config import get_configs
from model.flowbot import Flowbot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    testing = not args.train
    config = get_configs()
    flowbot = Flowbot(config, testing)

    if testing:
        flowbot.interact()
    else:
        flowbot.train()

if __name__=='__main__':
    main()
