import logging

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import ForeignKeyConstraint

from sqlalchemy.types import BigInteger
from sqlalchemy.types import Float

from nti.machine_learning import FORMAT

from nti.machine_learning.database import AbstractTable
from nti.machine_learning.database import PersistentBase

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

class Student(AbstractTable,
              PersistentBase):
    
    KEYS = ['sooner_id', 'realname', 'fbyf', 'total_expenses', 'total_aid']
    
    __tablename__ = "Student"
    
    sooner_id = Column(BigInteger, primary_key=True)
    realname = Column(String(100))
    fbyf = Column(String(8))
    total_expenses = Column(Float)
    total_aid = Column(Float)

