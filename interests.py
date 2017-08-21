import click
import logging

from csv import reader

from nti.data import FORMAT

from nti.data.database.oubound import get_columns
from nti.data.database.oubound import insert_interest

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

def _load(file_name):
    logging.info('Loading file %s...' % file_name)
    with open(file_name, 'r')as f:
        lines = [line for line in reader(f)]
    keys = get_columns("Interests")
    for line in lines[1:]:
        keyword_args = dict(zip(keys, line))
        for key in keyword_args:
            if key != 'sooner_id':
                keyword_args[key] = True if keyword_args[key] == '1' else False
        insert_interest(**keyword_args)
    logging.info('Done.')
    
    

@click.command()
@click.argument('file_name', nargs=1)
def load(file_name):
    """
    Load CSV file of student interests
    """
    _load(file_name)

@click.group()
def cli():
    """
    Base command
    """
    pass

cli.add_command(load)

if __name__ == '__main__':
    cli()