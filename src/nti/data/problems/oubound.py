"""
This file provides the solutions and anlaysis for problems
given by OUBound.
"""

import logging

from pandas import concat
from pandas import merge

import statsmodels.api as sm

from sqlalchemy.exc import IntegrityError

from nti.data import FORMAT
from nti.data import NTIDataFrame

from nti.data.algorithms.utils import variance

from nti.data.algorithms.supervised.support_vector_machine import SupportVectorMachine

from nti.data.database.oubound import get_sentiments
from nti.data.database.oubound import get_sentiments_by_soonerid
from nti.data.database.oubound import Sentiments
from nti.data.database.oubound import insert_obj
from nti.data.database.oubound import get_avg_student_response_measures
from nti.data.database.oubound import get_student_total_aid
from nti.data.database.oubound import get_student_interests
from nti.data.database.oubound import get_student_responses_with_variable

logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger(__name__)

# Map values from 0-1 to a severity string
severities = {0: "Very Low", 1: "Low",
              2: "Medium", 3: "High", 4: "Very High"}

# Variables of student essays in OUBoundEssays
variables = ['Freshman.Essay.Response', 'Scholarship.Academic.Major.Essay',
             'Scholarship.Community.Essay', 'Scholarship.Leadership.Essay']

def _get_severity(quantity):
    """
    Given a 0-1 quanitity, bucket it into 
    a qualitative severity measure
    """
    bucket = int((quantity - (quantity % .2)) / .2)
    bucket = bucket if bucket <= 4 else bucket
    return severities[bucket]

class OUBoundClusterStats():
    """
    Object that gets and contains relevant
    status about a cluster.
    """
    label_keys = Sentiments.KEYS[2:]
    
    def _get_variances(self):
        labeled_data = [p.provide_labels(self.label_keys) for p in self.points]
        
        # Find the variance for each sentiment in this cluster
        variance_data = {}
        for key in self.label_keys:
            _mean = self.center[key]
            vals = [p[key] for p in labeled_data]
            variance_data[key] = variance(vals, _mean)
        
        # Return the variances
        return variance_data
    
    def _get_ident_factors(self, size):
        # Sort the variances, get the top three
        ident_keys = sorted(self.variances, key=self.variances.__getitem__)[:size]
        
        # Return a dictionary that buckets the mean sentiment into severity levels
        return {i: _get_severity(self.center[i]) for i in ident_keys}
    
    def __init__(self, cluster, frame, points, size):
        self.points = points
        self.cluster = cluster
        self.center = frame.get_cluster_centers()[cluster["cluster"][0]]
        self.size = frame.size()
        self.variances = self._get_variances()
        self.identifying_factors = self._get_ident_factors(size)
        self.identifying_variances = {key: self.variances[key] for key in self.identifying_factors.keys()}

class OUBoundEssayStats():
    """
    Performs clustering and fetches and prints the resulting 
    statistics using a user-given clustering algorithm
    """
    def _construct_clustering(self, algo, args):
        algo_args = [self.points] + list(args)
        return algo(*algo_args)

    def __init__(self, clustering_algo, *args):
        self.clustering_algo = clustering_algo
        self.args = args
    
    def _print_cluster(self, cluster, frame, out_file, size):
        """
        Print results
        """
        stats = OUBoundClusterStats(cluster, frame, self.points, size)
        out_file.write("Group Size: %s\n" % stats.size)
        out_file.write("Group Identifying Factors: \n")
        for key, value in stats.identifying_factors.items():
            out_file.write("%s: %s with variance %s\n" % (key, value, stats.identifying_variances[key]))
        out_file.write("\n")

    def build(self, out_file, size):
        """
        Build the stats
        """
        with open(out_file, 'w') as out:
            out.write('Run Parameters: \nSize: %s\n\n' % size)
            for v in variables:
                self.points = NTIDataFrame(get_sentiments(variable=v), columns=Sentiments.KEYS[2:])
                self.algo = self._construct_clustering(self.clustering_algo, self.args)
                logging.info('Clustering essay sentiments for variable %s...' % v)
                exit_data = self.algo.cluster()
                clusters = exit_data.get_clusters()
                logging.info('Clustered. Printing statistics...')
                out.write("Variable %s -----\n" % v)
                out.write("Number of groups: %s\n" % len(clusters))
                for c in clusters:
                    self._print_cluster(c, exit_data, out, size)
        logging.info('Statistics written to %s.' % out_file)
        logging.info('Finished.')

def build_essay_classifier(title):
    logging.info('Pulling sentiment tuples...')
    sentiments = NTIDataFrame(get_sentiments(include_variable=True), columns=Sentiments.KEYS[2:]+["variable"])
    logging.info('Training classifier...')
    svm = SupportVectorMachine(sentiments, "variable")
    svm.train()
    logging.info('Support Vector Machine trained with %.2f%% accuracy on validation.' % (svm.success_rate*100.0,))
    logging.info('Getting classifier pickle...')
    pickle = svm.get_pickle()
    logging.info('Saving model...')
    try:
        insert_obj("Model", pickle=pickle, title=title)
        logging.info('Saved as %s.' % title)
    except IntegrityError:
        logging.error('%s already used as classifier name.' % title)

def get_response_aid_correlation(variable=None):
    logging.info('Pulling data...')
    if variable is None:
        avg_response = get_avg_student_response_measures()
    else:
        avg_response = get_student_responses_with_variable(variable)
    aid = get_student_total_aid()
    ids = avg_response.sooner_id.unique()
    results = []
    for id in ids:
        sub_frame = avg_response.loc[avg_response.sooner_id == id]
        results.append(sub_frame.mean(axis=0).to_frame().transpose())
    total_frame = concat(results)
    result = merge(total_frame, aid, on="sooner_id").drop("sooner_id", axis=1)
    X = list(result.columns.values)[:-1]
    X = result[X]
    Y = [list(result.columns.values)[-1]]
    Y = result[Y]
    model = sm.OLS(Y,X).fit()
    logging.info('Summary:')
    print(model.summary())

def get_interest_aid_correlation():
    logging.info('Pulling data...')
    interests = get_student_interests()
    aid = get_student_total_aid()
    result = merge(interests, aid, on="sooner_id").drop("sooner_id", axis=1)
    X = list(result.columns.values)[:-1]
    X = result[X]
    Y = [list(result.columns.values)[-1]]
    Y = result[Y]
    model = sm.OLS(Y,X).fit()
    logging.info('Summary:')
    print(model.summary())
    
