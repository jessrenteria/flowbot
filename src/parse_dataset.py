from utils.config import get_configs
from data.preprocessor import Preprocessor

config = get_configs()
config['reuse_data'] = False
p = Preprocessor(config)
