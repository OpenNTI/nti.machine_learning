"""
This file provides the solutions and anlysis for problems
given by OUBound.
"""

import logging

from sqlalchemy.exc import IntegrityError

from nti.data import FORMAT

from nti.data.algorithms.common import Point

from nti.data.algorithms.utils import variance

from nti.data.algorithms.supervised.support_vector_machine import SupportVectorMachine

from nti.data.database.oubound import get_sentiments
from nti.data.database.oubound import get_sentiments_by_soonerid
from nti.data.database.oubound import get_model
from nti.data.database.oubound import Sentiments
from nti.data.database.oubound import OUBoundEssayDB

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
    
    def __init__(self, cluster, points, size):
        self.points = points
        self.cluster = cluster
        self.number = cluster.num
        self.center = self.cluster.center.provide_labels(self.label_keys)
        self.size = self.cluster.length()
        self.variances = self._get_variances()
        self.identifying_factors = self._get_ident_factors(size)
        self.identifying_variances = {key: self.variances[key] for key in self.identifying_factors.keys()}

class OUBoundEssayStats():
    """
    Performs clustering and fetches and prints the resulting 
    statistics using a user-given clustering algorithm
    """
    def _to_points(self, sentiment_tuples):
        return [Point(*s) for s in sentiment_tuples]

    def _construct_clustering(self, algo, args):
        algo_args = [self.points] + list(args)
        return algo(*algo_args)

    def __init__(self, clustering_algo, *args):
        self.clustering_algo = clustering_algo
        self.args = args
        
    def _print_cluster(self, cluster, out_file, size):
        """
        Print results
        """
        stats = OUBoundClusterStats(cluster, self.points, size)
        out_file.write("Group Number: %s\n" % stats.number)
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
                self.points = self._to_points(get_sentiments(variable=v))
                self.algo = self._construct_clustering(self.clustering_algo, self.args)
                logging.info('Clustering essay sentiments for variable %s...' % v)
                clusters = self.algo.cluster()
                logging.info('Clustered. Printing statistics...')
                out.write("Variable %s -----\n" % v)
                out.write("Number of groups: %s\n" % len(clusters))
                for c in clusters:
                    self._print_cluster(c, out, size)
        logging.info('Statistics written to %s.' % out_file)
        logging.info('Finished.')

def build_essay_classifier(title):
    logging.info('Pulling sentiment tuples...')
    sentiments = get_sentiments(include_variable=True)
    inputs = [s[:-1] for s in sentiments]
    outputs = [s[-1] for s in sentiments]
    logging.info('Training classifier...')
    svm = SupportVectorMachine(inputs, outputs)
    svm.train()
    logging.info('Support Vector Machine trained with %.2f%% accuracy on validation.' % svm.get_success_rate())
    logging.info('Getting classifier pickle...')
    pickle = svm.get_pickle()
    logging.info('Saving model...')
    db = OUBoundEssayDB()
    try:
        db.insert_model(pickle, title)
        logging.info('Saved as %s.' % title)
    except IntegrityError:
        logging.error('%s already used as classifier name.' % title)
    db.close()

def predict_essay(soonerid, classifier_name):
    logging.info('Pulling sentiment tuple...')
    sentiment = get_sentiments_by_soonerid(soonerid)
    if sentiment is None:
        logging.error('No student found with that student id.')
        return
    logging.info('Pulling model %s...' % classifier_name)
    model = get_model(classifier_name)
    if model is None:
        logging.error('No model found by that title.')
        return
    result = model.classify(sentiment)
    logging.info('Student %s predicted with essay type %s.' % (soonerid, result))
    
    
    
