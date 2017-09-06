import click
import logging
import codecs
import glob
import json

from csv import reader

from nti.data.algorithms import DBScan
from nti.data.algorithms import Entropic
from nti.data.algorithms import KMeans

from nti.data.database.oubound import get_data_from_json
from nti.data.database.oubound import get_columns
from nti.data.database.oubound import insert_obj

from nti.data.problems.oubound import OUBoundEssayStats
from nti.data.problems.oubound import build_essay_classifier
from nti.data.problems.oubound import get_response_aid_correlation
from nti.data.problems.oubound import get_interest_aid_correlation

from nti.data.database.oubound import OUBoundEssayDB

# Available algorithm options
ALGOS = [DBScan.__name__, KMeans.__name__, Entropic.__name__]

def _load_interest(file_name):
    logging.info('Loading file %s...' % file_name)
    with open(file_name, 'r')as f:
        lines = [line for line in reader(f)]
    keys = get_columns("Interests")
    for line in lines[1:]:
        keyword_args = dict(zip(keys, line))
        for key in keyword_args:
            if key != 'sooner_id':
                keyword_args[key] = True if keyword_args[key] == '1' else False
        insert_obj("Interests", **keyword_args)
    logging.info('Done.')

def _read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    if not 'full.json' in file_name:
        get_data_from_json(data)
    

def _load_finance(file_name, is_directory=False):
    if is_directory:
        logging.info('Opening directory %s...' % file_name)
        for f in glob.glob(file_name+"/*.json"):
            logging.info('Reading file %s...' % f)
            _read_json(f)
    else:
        logging.info('Reading file %s...' % file_name)
        _read_json(file_name)
    logging.info('Done.')

def _do_essay_analysis(algorithm, algo_args, out_file, size):
    """
    Call the analysis with a given algorithm, its arguments, output file,
    and size.
    """
    algo_class = eval(algorithm)
    stats = OUBoundEssayStats(algo_class, *algo_args)
    stats.build(out_file, size)

def _load_essay(file_name):
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
            db.insert_essay(line)
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

def _get_response_aid_correlation(essay):
    get_response_aid_correlation(essay)

def _get_interest_aid_correlation():
    get_interest_aid_correlation()

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
@click.option('--size', '-s', help='How many factors will be shown in analysis', type=int)
def ouboundessay(algorithm, algo_args, file, size):
    """
    Performs analysis on the OUBoundEssay MySQL database.
    """
    file = file if file is not None else "statsfile.csv"
    size = size if size is not None else 3
    _do_essay_analysis(algorithm, algo_args, file, size)

@click.command()
@click.argument('file_name', nargs=1)
@click.option('-e', '--essay', is_flag=True)
@click.option('-f', '--finance', is_flag=True)
@click.option('-i', '--interest', is_flag=True)
def load(file_name, essay, finance, interest):
    """
    Loads a resource into the OUBoundEssay MySQL db.
    """
    if essay:
        _load_essay(file_name)
    elif finance:
        _load_finance(file_name, is_directory=True)
    elif interest:
        _load_interest(file_name)

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
@click.argument('predictor', nargs=1, type=click.Choice(['interest', 'response']))
@click.option('-e', '--essay', help='The essay variable to examine.', type=click.Choice(['Freshman.Essay.Response', 'Scholarship.Leadership.Essay',
                                                                                         'Scholarship.Community.Essay', 'Scholarship.Academic.Major.Essay']))
def aid_correlation(predictor, essay):
    """
    Finds the correlation coefficient between a predictor
    and the amount of aid received
    """
    if predictor == 'interest':
        _get_interest_aid_correlation()
    else:
        _get_response_aid_correlation(essay)

oubound.add_command(ouboundessay)
oubound.add_command(load)
oubound.add_command(build_predict)
oubound.add_command(aid_correlation)

if __name__ == '__main__':
    oubound()
