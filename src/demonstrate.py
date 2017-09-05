import click
import logging

from nti.data import FORMAT

from nti.data.algorithms import NeuralNetwork
from nti.data.algorithms import KMeans
from nti.data.algorithms import DBScan
from nti.data.algorithms import Entropic

from nti.data.fake import DataGenerator
from nti.data.fake import Plotter

from nti.data import NTIDataFrame

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

# Available algorithms
U_ALGOS = [KMeans.__name__, DBScan.__name__, Entropic.__name__]

def _unsupervised(algo, algo_args, point_number, non_uniform):
    """
    Performs the demonstration of an unsupervised algorithm using fake data
    """
    logging.info('Generating %s points...' % point_number)
    generator = DataGenerator(point_number)
    points = generator.generate_sample_data(non_uniform=non_uniform)
    
    data_frame = NTIDataFrame(points, columns=['A', 'B', 'C'])

    logging.info('Performing %s...' % algo)
    arg_list = [data_frame]+list(algo_args)
    algo = eval(algo)(*arg_list)
    new_frame = algo.cluster()

    plotter = Plotter(new_frame)
    plotter.plot(algo.__class__.__name__ + " Demonstration")
    

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

demonstrate.add_command(unsupervised)

if __name__ == '__main__':
    demonstrate()
        
        
