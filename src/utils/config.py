"""Helper file for parsing config file.
"""
import configparser

def get_configs():
    """Casts values in config file and places into a single dict
    """
    parser = configparser.ConfigParser()
    parser.read('config.ini')
    config = {}

    for section in parser.sections():
        for key in parser[section]:
            if section == 'ints':
                config[key] = parser[section].getint(key)
            elif section == 'floats':
                config[key] = parser[section].getfloat(key)
            elif section == 'bools':
                config[key] = parser[section].getboolean(key)
            else:
                config[key] = parser[section][key]

    return config
