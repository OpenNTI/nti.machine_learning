import click
import logging
import glob
import json

from sqlalchemy.exc import IntegrityError

from nti.data.algorithms import DB_SCAN
from nti.data.algorithms import KMEANS
from nti.data.algorithms import ENTROPY

from nti.data.database.oubound import get_data_from_json

from nti.data.algorithms.supervised.neural_network import NeuralNetwork

from nti.data.fake import SupervisedGenerator

# Available algorithm options
ALGOS = [DB_SCAN, KMEANS, ENTROPY]



def _read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    if not 'full.json' in file_name:
        get_data_from_json(data)
    

def _load(file_name, is_directory=False):
    if is_directory:
        logging.info('Opening directory %s...' % file_name)
        for f in glob.glob(file_name+"/*.json"):
            logging.info('Reading file %s...' % f)
            _read_json(f)
    else:
        logging.info('Reading file %s...' % file_name)
        _read_json(file_name)
    logging.info('Done.')

def _interest():
    sg = SupervisedGenerator(500)
    points = sg.generate_sample_data()
    x_s = [p.values for p in points]
    y_s = [p.correct for p in points]
    nn = NeuralNetwork(x_s, y_s, [3, 5, 1], training_size=.55)
    nn.train(10)
    #nn.classify([95, 12, 44])

@click.group()
def finance():
    pass

@click.command()
@click.argument('file_name', nargs=1)
@click.option('-d', '--directory', help="Read as directory", is_flag=True)
def load(file_name, directory):
    _load(file_name, directory)
    
@click.command()
def interest(): 
    """
    Analyize essay sentiments against interest levels
    """   
    _interest()

finance.add_command(load)
finance.add_command(interest)

if __name__ == '__main__':
    finance()