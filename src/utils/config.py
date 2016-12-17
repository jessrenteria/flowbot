"""Helper file for parsing config files.
"""
import configparser

class _ConfigParserWrapper:
    """Wrapper for handling parsing config files.
    """
    def __init__(self):
        self._parser = configparser.ConfigParser()
        self._config = {}

    def parse(self, config_file):
        self._parser.read(config_file)

        for section in self._parser.sections():
            for key in self._parser[section]:
                assert key not in self._config

                if section == 'ints':
                    self._config[key] = self._parser[section].getint(key)
                elif section == 'floats':
                    self._config[key] = self._parser[section].getfloat(key)
                elif section == 'bools':
                    self._config[key] = self._parser[section].getboolean(key)
                else:
                    self._config[key] = self._parser[section][key]

            self._parser.remove_section(section)

    def get_config(self):
        return self._config

def get_config(bot_name):
    """Casts values in configs file and places into a single dict
    """
    bot_dir = 'bots/' + bot_name
    parser = _ConfigParserWrapper()
    parser.parse('global_config.ini')
    parser.parse(bot_dir + '/config.ini')
    config = parser.get_config()

    config['train_info'] = bot_dir + '/train.info'
    config['checkpoint_dir'] = bot_dir + '/checkpoints'
    config['checkpoint_file'] = config['checkpoint_dir'] + '/flowbot'
    config['tensorboard_dir'] = bot_dir + '/tensorboard'

    return config
