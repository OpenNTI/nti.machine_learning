import click
import logging
import codecs

from csv import reader

from nti.data.algorithms import DB_SCAN
from nti.data.algorithms import KMEANS
from nti.data.algorithms import ENTROPY

from nti.data.database.oubound import get_columns
from nti.data.database.oubound import insert_interest

def _load(file_name):
    with open(file_name, 'r')as f:
        lines = [line for line in reader(f)]
    keys = get_columns("Interests")
    for line in lines[1:]:
        keyword_args = dict(zip(keys, line))
        for key in keyword_args:
            if key != 'sooner_id':
                keyword_args[key] = True if keyword_args[key] == '1' else False
        insert_interest(**keyword_args)
    
    

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