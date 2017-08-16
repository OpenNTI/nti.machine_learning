import click
import logging
import codecs

from csv import reader

from nti.data.algorithms import DB_SCAN
from nti.data.algorithms import KMEANS
from nti.data.algorithms import ENTROPY

from nti.data.algorithms.clustering.geometric import KMeans

from nti.data.algorithms.clustering.density import DBScan
from nti.data.algorithms.clustering.density import Entropic

from nti.data.problems.oubound import OUBoundEssayStats
from nti.data.problems.oubound import build_essay_classifier
from nti.data.problems.oubound import predict_essay

from nti.data.database.oubound import OUBoundEssayDB

# Available algorithm options
ALGOS = [DB_SCAN, KMEANS, ENTROPY]

def _do_essay_analysis(algorithm, algo_args, out_file, size):
    """
    Call the analysis with a given algorithm, its arguments, output file,
    and size.
    """
    algo_class = eval(algorithm)
    stats = OUBoundEssayStats(algo_class, *algo_args)
    stats.build(out_file, size)

def _load_file(file_name):
    """
    Loads a csv file of OUBound essays into the MySQL database.
    """
    db = OUBoundEssayDB()
    logging.info('Reading file %s and building objects...' % file_name)
    with codecs.open(file_name, "r", encoding='utf-8', errors='ignore') as fdata:
        lines = [line for line in reader(fdata)]
        length = len(lines)
        count = 0
        for line in lines[1:]:
            db.insert(line)
            if count % 1000 == 0:
                logging.info('%s%%' % int((count / length) * 100))
            count += 1
    db.close()
    logging.info('Done.')

def _build_predict(title):
    """
    Uses essay sentiments to predict type of essay.
    (Experimental)
    """
    build_essay_classifier(title)

def _predict(soonerid, classifier_name):
    """
    Predict essay type of person soonerid
    """
    predict_essay(soonerid, classifier_name)

@click.group()
def oubound():
    """
    Root command
    """
    pass

@click.command()
@click.argument('algorithm', nargs=1, type=click.Choice(ALGOS))
@click.argument('algo_args', nargs=-1)
@click.option('--file', '-f', help='Output file for statistics')
@click.option('--size', '-s', help='How many factors will be shows in analysis', type=int)
def ouboundessay(algorithm, algo_args, file, size):
    """
    Performs analysis on the OUBoundEssay MySQL database.
    """
    file = file if file is not None else "statsfile.csv"
    size = size if size is not None else 3
    _do_essay_analysis(algorithm, algo_args, file, size)

@click.command()
@click.argument('file_name', nargs=1)
def load(file_name):
    """
    Loads a csv file of OUBound essays into the MySQL database.
    """
    _load_file(file_name)

@click.command()
@click.option('-t', '--title', help="Title of the resulting model")
def build_predict(title):
    """
    Uses essay sentiments to predict type of essay.
    (Experimental)
    """
    title = "Essay_Classified" if title is None else title
    _build_predict(title)

@click.command()
@click.argument('soonerid', type=int)
@click.argument('classifier_name', type=str)
def predict(soonerid, classifier_name):
    """
    Predict essay type of person soonerid
    """
    _predict(soonerid, classifier_name)

oubound.add_command(ouboundessay)
oubound.add_command(load)
oubound.add_command(build_predict)
oubound.add_command(predict)

if __name__ == '__main__':
    oubound()
