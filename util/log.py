__author__ = 'matt'

import os
import json
import logging.config


def setup_logging(
    default_path='logging.json',
    default_level=logging.INFO,
    env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)

    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            raw_data = f.read()
            config = json.loads(raw_data)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
