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
    
    def insert_model(self, model_pickle, title):
        m = Model(pickle=model_pickle, title=title)
        self.session.add(m)
        self.session.commit()
    
    def insert_student(self, sooner_id, realname, fbyf, total_expenses=0, total_aid=0):
        new_student = Student(sooner_id=sooner_id, realname=realname, fbyf=fbyf, total_expenses=total_expenses, total_aid=total_aid)
        try:
            self.session.add(new_student)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Student %s already in database.' % realname)
        
    def insert_expense(self, sooner_id, title, description, amount):
        new_expense = Expense(sooner_id=sooner_id, title=title, description=description, amount=amount)
        try:
            self.session.add(new_expense)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Expense %s, %s already in database.' % (sooner_id, title))
            
    def insert_scholarship(self, sooner_id, title, description, amount):
        new_scholarship = Scholarship(sooner_id=sooner_id, title=title, description=description, amount=amount)
        try:
            self.session.add(new_scholarship)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Scholarship %s, %s already in database.' % (sooner_id, title))
            
    def insert_award(self, sooner_id, title, description, amount):
        new_scholarship = Award(sooner_id=sooner_id, title=title, description=description, amount=amount)
        try:
            self.session.add(new_scholarship)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Award %s, %s already in database.' % (sooner_id, title))
    
    def insert_family_contrib(self, sooner_id, cents_per_installment, installment_frequency):
        new_family_contrib = FamilyContrib(sooner_id=sooner_id, cents_per_installment=cents_per_installment, installment_frequency=installment_frequency)
        try:
            self.session.add(new_family_contrib)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Family Contrib %s already in database.' % (sooner_id,))
            
    def insert_work_contrib(self, sooner_id, hours_per_week, amount):
        new_work_contrib = WorkContrib(sooner_id=sooner_id, hours_per_week=hours_per_week, amount=amount)
        try:
            self.session.add(new_work_contrib)
            self.session.commit()
        except:
            self.session.rollback()
            logging.error('Work Contrib %s already in database.' % (sooner_id,))
            
    def insert_interest(self, **kwargs):
        new_interest = Interests(**kwargs)
        try:
            self.session.add(new_interest)
            self.session.commit()
        except Exception:
            self.session.rollback()
            logging.error('Error inserting Interest into the database')
        
    
    def get_model(self, title):
        return self.session.query(Model).filter_by(title=title).first()
    
    def update_student(self, sooner_id, total_expenses, total_aid):
        student = self.session.query(Student).filter_by(sooner_id=sooner_id).first()
        student.total_expenses = total_expenses
        student.total_aid = total_aid
        self.session.commit()
    
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

def _save_expenses_from_json(sooner_id, json_snippet, db):
    """
    Given a json snippet, save expenses into the database
    """
    cost_group = json_snippet["Groups"][0]
    for item in cost_group["PlanItems"]:
        title = item["title"]
        description = item["description"]
        amount = float(item['amount']['cents'])/100.0
        db.insert_expense(sooner_id=sooner_id, title=title, description=description, amount=amount)
    return float(cost_group["TotalCents"])/100.0

def _save_scholarships_from_json(sooner_id, json_snippet, db):
    """
    Given a json snippet, will save the scholarships into the database
    """
    external_aid_group = json_snippet["Groups"][3]
    for item in external_aid_group["PlanItems"]:
        title = item["title"]
        description = item["description"]
        amount = item["amount"]
        amount = -1.0 if amount is None else float(amount['cents'])/100.0
        db.insert_scholarship(sooner_id=sooner_id, title=title, description=description, amount=amount)
    return float(external_aid_group["TotalCents"])/100.0

def _save_awards_from_json(sooner_id, json_snippet, db):
    aid_group = json_snippet["Groups"][1]
    for item in aid_group["PlanItems"]:
        title = item["title"]
        description = item["description"]
        amount = item["amount"]
        amount = -1.0 if amount is None else float(amount['cents'])/100.0
        db.insert_award(sooner_id=sooner_id, title=title, description=description, amount=amount)
    return float(aid_group["TotalCents"])/100.0

def _get_family_contrib(json_snippet):
    amount = json_snippet["amount"]
    if amount is None:
        return (-1, "")
    try:
        cents_per_installment = float(amount["cents_per_installment"])/100.0
        frequency = amount["frequency"]
    except KeyError:
        cents_per_installment = -1
        frequency = ""
    return (cents_per_installment, frequency)

def _get_work_contrib(json_snippet):
    amount = json_snippet["amount"]
    if amount is None:
        return (-1, -1)
    try:
        hours_per_week = amount["hours_per_week"]
        cents = float(amount["cents"])/100.0
    except KeyError:
        hours_per_week = -1
        cents = -1
    return (hours_per_week, cents)

def _save_contribs_from_json(sooner_id, json_snippet, db):
    contrib_group = json_snippet["Groups"][2]
    item = contrib_group["PlanItems"][0]
    family = _get_family_contrib(item)
    db.insert_family_contrib(sooner_id, family[0], family[1])
    item = contrib_group["PlanItems"][1]
    work = _get_work_contrib(item)
    db.insert_work_contrib(sooner_id, work[0], work[1])
    return float(contrib_group["TotalCents"])/100.0

def insert_interest(**kwargs):
    db = OUBoundEssayDB()
    db.insert_interest(**kwargs)
    logging.info("Loaded interests for %s into database." % kwargs.get('sooner_id'))
    db.close()
