import logging

from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint

from sqlalchemy.types import BigInteger
from sqlalchemy.types import Boolean

from nti.data import FORMAT

from nti.data.database import PersistentBase

from nti.data.database.oubound.base import Student

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

class Interests(PersistentBase):
    
    KEYS = ['sooner_id', 'clubs_or_orgs','community_services',
            'greek', 'intramural_sports', 'honors', 'study_abroad',
            'employment', 'leadership', 'research', 'cultural_orgs',
            'student_government', 'student_publications', 'marching_band',
            'choir', 'debate', 'no_interest', 'rotc']
    
    __tablename__ = "Interests"
    
    sooner_id = Column(BigInteger, primary_key=True)
    clubs_or_orgs = Column(Boolean)
    community_services = Column(Boolean)
    greek = Column(Boolean)
    intramural_sports = Column(Boolean)
    honors = Column(Boolean)
    study_abroad = Column(Boolean)
    employment = Column(Boolean)
    leadership = Column(Boolean)
    research = Column(Boolean)
    cultural_orgs = Column(Boolean)
    student_government = Column(Boolean)
    student_publications = Column(Boolean)
    marching_band = Column(Boolean)
    choir = Column(Boolean)
    debate = Column(Boolean)
    no_interest = Column(Boolean)
    rotc = Column(Boolean)
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])