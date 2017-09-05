import logging

from sqlalchemy import create_engine

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import sessionmaker

from nti.data import FORMAT

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

# Base of sqlalchemy based MySQL tables
PersistentBase = declarative_base()


class AbstractDatabase():
    """
    Basic database operations
    """
    def __init__(self, conn_str):
        self.engine = create_engine(
            conn_str, echo=False)
        self.session = sessionmaker(bind=self.engine)()
        
    def __enter__(self):
        try:
            self.__init__()
        except AttributeError:
            logging.error('Must provide database connection string')
        return self
            
    def __exit__(self, type, value, traceback):
        self.close()

    def engine(self):
        return self.engine

    def session(self):
        return self.session
    
    def insert_obj(self, obj, **kwargs):
        new_obj = obj(**kwargs)
        try:
            self.session.add(new_obj)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Could not insert new object into database.')
            
    def close(self):
        self.session.close_all()

class AbstractTable():
    """
    Basic Table Object
    """
    
    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)
