import logging
import _pickle as cPickle

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import ForeignKeyConstraint

from sqlalchemy.exc import IntegrityError

from sqlalchemy.orm.exc import FlushError

from sqlalchemy.types import BigInteger
from sqlalchemy.types import Float
from sqlalchemy.types import Text
from sqlalchemy.types import Integer
from sqlalchemy.types import PickleType

from nti.data import FORMAT

from nti.data.database import PersistentBase
from nti.data.database import AbstractDatabase

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

    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)


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

    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

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
            
class Model(PersistentBase):
    
    KEYS = ['model_id', 'pickle', 'title']
    
    __tablename__ = 'Models'
    
    pickle = Column(PickleType)
    title = Column(String(20), primary_key=True)

    def __init__(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

class OUBoundEssayDB(AbstractDatabase):
    """
    Object interface for dealing with the OUBoundEssay
    database.
    """
    CONN_STR = 'mysql://root@localhost:3306/OUBoundEssay'
    
    def __init__(self):
        super(OUBoundEssayDB, self).__init__(self.CONN_STR)
    
    def _build_essay_from_str(self, line):
        args = dict(zip(Essay.KEYS, line[1:5]))
        return Essay(**args)
        
    def _build_sentiment_from_str(self, line):
        values = [line[1]] + [line[3]] + line[5:]
        values = [0 if x == 'NA' else x for x in values]
        args = dict(zip(Sentiments.KEYS, values))
        return Sentiments(**args)    
    
    def _insert_essay_from_str(self, essay_str):
        """
        Given a string, insert it as an Essay object.
        """
        obj = self._build_essay_from_str(essay_str)
        try:
            self.session.add(obj)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Found duplicate essay instance with id %s. Skipping...' % obj.sooner_id)
    
    def _insert_sentiment_from_str(self, sent_str):
        """
        Given a string, insert it as a Sentiment object
        """
        obj = self._build_sentiment_from_str(sent_str)
        try:
            self.session.add(obj)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Found duplicate essay instance with id %s. Skipping...' % obj.sooner_id)
    
    def insert(self, line):
        """
        Given a line, pull the essay and sentiment objects
        """
        self._insert_essay_from_str(line)
        self._insert_sentiment_from_str(line)
    
    def get_sentiments_as_tuple(self, variable=None, include_variable=False):
        """
        Get all sentiments of a variable form the database as a list
        of lists.
        """
        if variable is not None:
            sentiments = self.session.query(Sentiments).filter_by(variable=variable)
        else:
            sentiments = self.session.query(Sentiments)
        result = []
        for s in sentiments:
            att_list = s.as_list(include_variable=include_variable)
            if not all([x == 0 for x in att_list]):
                result.append(att_list)
        return result
    
    def get_sentiments_as_tuple_with_soonerid(self, sooner_id):
        """
        Get list of sentiments by sooner_id
        """
        sentiment = self.session.query(Sentiments).filter_by(sooner_id=sooner_id).first()
        try:
            return sentiment.as_list()
        except AttributeError:
            return None
    
    def insert_model(self, model_pickle, title):
        m = Model(pickle=model_pickle, title=title)
        self.session.add(m)
        self.session.commit()
    
    def get_model(self, title):
        return self.session.query(Model).filter_by(title=title).first()
    
    def close(self):
        self.session.close_all()

def get_sentiments(variable=None, include_variable=False):
    """
    Function for abstracting db interaction to get sentiments
    """
    db = OUBoundEssayDB()
    result = db.get_sentiments_as_tuple(variable, include_variable)
    db.close()
    return result

def get_sentiments_by_soonerid(sooner_id):
    """
    Function for abstracting db interaction to get sentiments
    by sooner id
    """
    db = OUBoundEssayDB()
    result = db.get_sentiments_as_tuple_with_soonerid(sooner_id)
    db.close()
    return result

def get_model(title):
    """
    Gets a model by title
    """
    db = OUBoundEssayDB()
    result = db.get_model(title)
    db.close()
    return result.pickle
    