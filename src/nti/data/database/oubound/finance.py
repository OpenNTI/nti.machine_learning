import logging

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import ForeignKeyConstraint

from sqlalchemy.types import BigInteger
from sqlalchemy.types import Float
from sqlalchemy.types import Text
from sqlalchemy.types import Integer

from nti.data import FORMAT

from nti.data.database import AbstractTable
from nti.data.database import PersistentBase

from nti.data.database.oubound.base import Student

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

class Expense(PersistentBase,
              AbstractTable):
    
    KEYS = ['sooner_id', 'title', 'description', 'amount']
    
    __tablename__ = "Expense"
    
    sooner_id = Column(BigInteger, primary_key=True)
    title = Column(String(100), primary_key=True)
    description = Column(Text)
    amount = Column(Float)
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])

class Scholarship(PersistentBase,
                  AbstractTable):
    
    KEYS = ['sooner_id', 'title', 'description', 'amount']
    
    __tablename__ = 'Scholarship'
    
    sooner_id = Column(BigInteger, primary_key=True)
    title = Column(String(100), primary_key=True)
    description = Column(Text)
    amount = Column(Float)
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])

class Award(PersistentBase,
            AbstractTable):
    
    KEYS = ['sooner_id', 'title', 'description', 'amount']
    
    __tablename__ = "Award"
    
    sooner_id = Column(BigInteger, primary_key=True)
    title = Column(String(100), primary_key=True)
    description = Column(Text)
    amount = Column(Float)
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])
    
class WorkContrib(PersistentBase,
                  AbstractTable):
    
    __tablename__ = 'Work'
    
    sooner_id = Column(BigInteger, primary_key=True)
    hours_per_week = Column(Integer)
    amount = Column(Float)
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])
    
class FamilyContrib(PersistentBase,
                    AbstractTable):
    
    __tablename__ = 'Family'
    
    sooner_id = Column(BigInteger, primary_key=True)
    cents_per_installment = Column(Float)
    installment_frequency = Column(String(50))
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])
