import logging
import sys

def load_logger_conf():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='loggings.log', level=logging.INFO, format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    # add the handler to the root logger
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    console = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger('').addHandler(console)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.FileHandler('loggings.log')