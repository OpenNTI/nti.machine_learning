from sqlalchemy import create_engine

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import sessionmaker

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

    def engine(self):
        return self.engine

    def session(self):
        return self.session
