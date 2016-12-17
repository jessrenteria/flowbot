import argparse

from utils.config import get_config
from model.flowbot import Flowbot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--bot', choices=['lstm', 'dizzy'], default='lstm')
    args = parser.parse_args()

    testing = not args.train
    config = get_config(args.bot)
    flowbot = Flowbot(config, testing)

    if testing:
        flowbot.interact()
    else:
        flowbot.train()

if __name__=='__main__':
    main()
