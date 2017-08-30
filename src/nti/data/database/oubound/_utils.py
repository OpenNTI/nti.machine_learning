from nti.data.database.oubound.essay import Essay
from nti.data.database.oubound.essay import Sentiments

from nti.data.database.oubound.finance import Expense
from nti.data.database.oubound.finance import Scholarship
from nti.data.database.oubound.finance import Award
from nti.data.database.oubound.finance import FamilyContrib
from nti.data.database.oubound.finance import WorkContrib

def _build_essay_from_str(self, line):
        args = dict(zip(Essay.KEYS, line[1:5]))
        return args
    
def _build_sentiment_from_str(self, line):
        values = [line[1]] + [line[3]] + line[5:]
        values = [0 if x == 'NA' else x for x in values]
        args = dict(zip(Sentiments.KEYS, values))
        return args

def _save_expenses_from_json(sooner_id, json_snippet, db):
    """
    Given a json snippet, save expenses into the database
    """
    cost_group = json_snippet["Groups"][0]
    for item in cost_group["PlanItems"]:
        title = item["title"]
        description = item["description"]
        amount = float(item['amount']['cents'])/100.0
        db.insert_obj(Expense, sooner_id=sooner_id, title=title, description=description, amount=amount)
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
        db.insert_obj(Scholarship, sooner_id=sooner_id, title=title, description=description, amount=amount)
    return float(external_aid_group["TotalCents"])/100.0

def _save_awards_from_json(sooner_id, json_snippet, db):
    aid_group = json_snippet["Groups"][1]
    for item in aid_group["PlanItems"]:
        title = item["title"]
        description = item["description"]
        amount = item["amount"]
        amount = -1.0 if amount is None else float(amount['cents'])/100.0
        db.insert_obj(Award, sooner_id=sooner_id, title=title, description=description, amount=amount)
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
    db.insert_obj(FamilyContrib, sooner_id, family[0], family[1])
    item = contrib_group["PlanItems"][1]
    work = _get_work_contrib(item)
    db.insert_obj(WorkContrib, sooner_id, work[0], work[1])
    return float(contrib_group["TotalCents"])/100.0