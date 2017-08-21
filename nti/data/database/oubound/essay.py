import logging

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import ForeignKeyConstraint

from sqlalchemy.types import BigInteger
from sqlalchemy.types import Float
from sqlalchemy.types import Text
from sqlalchemy.types import Integer
from sqlalchemy.types import PickleType

from nti.data import FORMAT

from nti.data.database import AbstractTable

from nti.data.database import PersistentBase

from nti.data.database.oubound.base import Student

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

class Essay(PersistentBase):
    """
    Object representation of the Essay table in the 
    OUBoundEssay database.
    """
    KEYS = ['sooner_id', 'essay_choice', 'variable', 'value']

    __tablename__ = 'Essay'
    
    sooner_id = Column(BigInteger, primary_key=True)
    essay_choice = Column(Text)
    variable = Column(String(100), primary_key=True)
    value = Column(Text)
    
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])

class Sentiments(PersistentBase):
    """
    Object representaiton of the Sentiments table in the 
    OUBoundEssay database.
    """
    KEYS = ['sooner_id', 'variable', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'emotional_range',
            'adventurousness', 'artistic_interests', 'emotionality', 'imagination', 'intellect', 'authority_challenging',
            'achievement_striving', 'cautiousness', 'dutifulness', 'orderliness', 'self_discipline', 'self_efficacy',
            'activity_level', 'assertiveness', 'cheerfulness', 'excitement_seeking', 'outgoing', 'gregariousness', 'altruism',
            'cooperation', 'modesty', 'uncompromising', 'sympathy', 'trust', 'fiery', 'prone_to_worry', 'melancholy', 'immoderation',
            'self_consciousness', 'susceptible_to_stress', 'challenge', 'closeness', 'curiosity', 'excitement', 'harmony',
            'ideal', 'liberty', 'love', 'practicality', 'self_expression', 'stability', 'structure', 'conservation', 'openness_to_change',
            'hedonism', 'self_enhancement', 'self_transcendence']

    __tablename__ = 'Sentiments'
    
    sooner_id = Column(BigInteger, primary_key=True)
    variable = Column(String(100), primary_key=True)
    openness = Column(Float)
    conscientiousness = Column(Float)
    extraversion = Column(Float)
    agreeableness = Column(Float)
    emotional_range = Column(Float)
    adventurousness = Column(Float)
    artistic_interests = Column(Float)
    emotionality = Column(Float)
    imagination = Column(Float)
    intellect = Column(Float)
    authority_challenging = Column(Float)
    achievement_striving = Column(Float)
    cautiousness = Column(Float)
    dutifulness = Column(Float)
    orderliness = Column(Float)
    self_discipline = Column(Float)
    self_efficacy = Column(Float)
    activity_level = Column(Float)
    assertiveness = Column(Float)
    cheerfulness = Column(Float)
    excitement_seeking = Column(Float)
    outgoing = Column(Float)
    gregariousness = Column(Float)
    altruism = Column(Float)
    cooperation = Column(Float)
    modesty = Column(Float)
    uncompromising = Column(Float)
    sympathy = Column(Float)
    trust = Column(Float)
    fiery = Column(Float)
    prone_to_worry = Column(Float)
    melancholy = Column(Float)
    immoderation = Column(Float)
    self_consciousness = Column(Float)
    susceptible_to_stress = Column(Float)
    challenge = Column(Float)
    closeness = Column(Float)
    curiosity = Column(Float)
    excitement = Column(Float)
    harmony = Column(Float)
    ideal = Column(Float)
    liberty = Column(Float)
    love = Column(Float)
    practicality = Column(Float)
    self_expression = Column(Float)
    stability = Column(Float)
    structure = Column(Float)
    conservation = Column(Float)
    openness_to_change = Column(Float)
    hedonism = Column(Float)
    self_enhancement = Column(Float)
    self_transcendence = Column(Float)

    ForeignKeyConstraint([sooner_id, variable], [
                         Essay.sooner_id, Essay.variable])
    ForeignKeyConstraint([sooner_id], [Student.sooner_id])

    def as_list(self, include_variable=False):
        """
        Get a row of the table as a list of values
        not including the key
        """
        numeric_keys = self.KEYS[2:]
        result = []
        for key in numeric_keys:
            result.append(getattr(self, key))
        if include_variable:
            result.append(self.variable)
        return result

class Model(AbstractTable,
            PersistentBase):
    
    KEYS = ['model_id', 'pickle', 'title']
    
    __tablename__ = 'Models'
    
    pickle = Column(PickleType)
    title = Column(String(20), primary_key=True)

    
