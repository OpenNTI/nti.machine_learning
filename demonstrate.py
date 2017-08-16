import click
import logging

from nti.data import FORMAT

from nti.data.algorithms import DB_SCAN
from nti.data.algorithms import KMEANS
from nti.data.algorithms import ENTROPY
from nti.data.algorithms import SVM

from nti.data.algorithms.clustering.geometric import KMeans

from nti.data.algorithms.clustering.density import DBScan
from nti.data.algorithms.clustering.density import Entropic

from nti.data.algorithms.supervised.support_vector_machine import SupportVectorMachine

from nti.data.fake import DataGenerator
from nti.data.fake import Plotter
from nti.data.fake import Plotter2D
from nti.data.fake import Functions

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

# Available algorithms
U_ALGOS = [DB_SCAN, KMEANS, ENTROPY]
S_ALGOS = [SVM]

def _unsupervised(algo, algo_args, point_number, non_uniform):
    """
    Performs the demonstration of an unsupervised algorithm using fake data
    """
    logging.info('Generating %s points...' % point_number)
    generator = DataGenerator(point_number)
    points = generator.generate_sample_data(non_uniform=non_uniform)

    logging.info('Performing %s...' % algo)
    arg_list = [points]+list(algo_args)
    algo = eval(algo)(*arg_list)
    algo.cluster()

    plotter = Plotter(points)
    plotter.plot(algo.__class__.__name__ + " Demonstration")

BELOW = 0
ABOVE = 1

def _get_classification(function, point):
    return "Above" if point.get(1) >= point.get(0) else "Below"

def _get_prediction(val):
    return "Above" if val == ABOVE else "Below"

def _supervised(algo, point_num, stddev):
    """
    Performs the demonstration of a supervised algorithm using fake data
    """
    logging.info('Generating points...')
    generator = DataGenerator(point_num)
    points = generator.generate_with_function(Functions.LINEAR, 100, stddev)
    
    correct_predicts = [_get_classification(Functions.LINEAR, p) for p in points]
    passable_points = [p.values for p in points]
    
    plotter = Plotter2D(points, correct_predicts)
    plotter.plot("Sample Points for SVM")
    
    logging.info('Building %s...' % algo)
    algo = eval(algo)(passable_points, correct_predicts, classes=["Above", "Below"])
    algo.train()
    logging.info('Success rate: %s%%' % algo.get_success_rate())
    x = 0
    while True:
        x, y = input("Enter two values x,y:").split(',')
        to_classify = [int(x), int(y)]
        print("Prediction: %s" % algo.classify(to_classify))
    

@click.group()
def demonstrate():
    """
    Root command
    """
    pass

@click.command()
@click.argument('algorithm', nargs=1, type=click.Choice(U_ALGOS))
@click.argument('algo_args', nargs=-1)
@click.option('-p', '--point-num', help="Number of points to use", type=int)
@click.option('-n', '--non-uniform', help="Uniformity of the points", type=bool)
def unsupervised(algorithm, algo_args, point_num, non_uniform):
    """
    Runs a demonstration of an unsupervised algorithm
    """
    point_num = 1000 if point_num is None else point_num
    non_uniform = False if non_uniform is None else non_uniform
    _unsupervised(algorithm, algo_args, point_num, non_uniform)

@click.command()
@click.argument('algorithm', nargs=1, type=click.Choice(S_ALGOS))
@click.option('--point-num', '-p', help="Number of points for train.", type=int)
@click.option('--stddev', '-s', help="Variance in point from function", type=int)
def supervised(algorithm, point_num, stddev):
    """
    Runs a demonstration of a supervised algorithm
    """
    point_num = 1000 if point_num is None else point_num
    stddev = 50 if stddev is None else stddev
    _supervised(algorithm, point_num, stddev)

demonstrate.add_command(unsupervised)
demonstrate.add_command(supervised)

if __name__ == '__main__':
    demonstrate()
