import logging

from nti.data import FORMAT

from nti.data.database import AbstractDatabase

from nti.data.database.oubound.essay import Model
from nti.data.database.oubound.essay import Essay
from nti.data.database.oubound.essay import Sentiments

from nti.data.database.oubound.base import Student

from nti.data.database.oubound.finance import Expense
from nti.data.database.oubound.finance import Scholarship
from nti.data.database.oubound.finance import Award
from nti.data.database.oubound.finance import FamilyContrib
from nti.data.database.oubound.finance import WorkContrib

from nti.data.database.oubound.interests import Interests

from nti.data.database.oubound._utils import _build_essay_from_str
from nti.data.database.oubound._utils import _build_sentiment_from_str
from nti.data.database.oubound._utils import _save_expenses_from_json
from nti.data.database.oubound._utils import _save_scholarships_from_json
from nti.data.database.oubound._utils import _save_awards_from_json
from nti.data.database.oubound._utils import _save_contribs_from_json

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

def get_columns(table):
    table = eval(table)
    return table.KEYS

class OUBoundEssayDB(AbstractDatabase):
    """
    Object interface for dealing with the OUBoundEssay
    database.
    """
    CONN_STR = 'mysql://root@localhost:3306/OUBoundEssay'
    TABLES = ["Model", "Essay", "Sentiments",
              "Student", "Expense", "Scholarship",
              "Award", "FamilyContrib", "WorkContrib",
              "Interests"]
    
    def __init__(self):
        super(OUBoundEssayDB, self).__init__(self.CONN_STR)    
    
    def _insert_essay_from_str(self, essay_str):
        """
        Given a string, insert it as an Essay object.
        """
        obj = _build_essay_from_str(essay_str)
        self.insert_obj(Essay, **obj)
    
    def _insert_sentiment_from_str(self, sent_str):
        """
        Given a string, insert it as a Sentiment object
        """
        obj = _build_sentiment_from_str(sent_str)
        self.insert_obj(Sentiments, **obj)
    
    def insert_essay(self, line):
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
    
    def get_model(self, title):
        return self.session.query(Model).filter_by(title=title).first()
    
    def update_student(self, sooner_id, total_expenses, total_aid):
        student = self.session.query(Student).filter_by(sooner_id=sooner_id).first()
        student.total_expenses = total_expenses
        student.total_aid = total_aid
        self.session.commit()
    
    def close(self):
        self.session.close_all()

def insert_obj(table, **kwargs):
    if table not in OUBoundEssayDB.TABLES:
        raise ValueError('Must give valid DB table.')
    obj = eval(table)
    db = OUBoundEssayDB()
    db.insert_obj(obj, **kwargs)
    db.close()

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

def get_data_from_json(json_snippet):
    """
    Save student information from a json snippet or file
    """
    db = OUBoundEssayDB()
    user_item = json_snippet["user"]
    sooner_id = user_item["SoonerID"]
    realname = user_item["realname"]
    fbyf = user_item["OUNetID"]
    db.insert_student(sooner_id=sooner_id, realname=realname, fbyf=fbyf)
    total_expenses = _save_expenses_from_json(sooner_id, json_snippet, db)
    total_aid = 0
    total_aid += _save_scholarships_from_json(sooner_id, json_snippet, db)
    total_aid += _save_awards_from_json(sooner_id, json_snippet, db)
    total_aid += _save_contribs_from_json(sooner_id, json_snippet, db)
    db.update_student(sooner_id, total_expenses, total_aid)
    db.close()
