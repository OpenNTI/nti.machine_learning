#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.cross_validation import cross_val_score

from zope import interface

from nti.machine_learning.evaluation.interfaces import IKFoldCrossValidation


@interface.implementer(IKFoldCrossValidation)
class KFoldCrossValidation(object):
    """
    This class will estimate likely performance of a machine learning model.
    It splits datasets into k equal partitions (folds).
    For each iteration, a partition/fold serves as testing data and the rest as training data
    Testing accuracy is calculated on each iteration.
    The iteration is repeated until k times and each time different partition serves as testing data.
    After k iteration, the average testing accuracy estimates the out-of sample accuracy.

    It accepts parameters :
    estimator :
        machine learning type
        (could be Neural Network, k-Nearest Neighbor Classfifier, Linear Regressor, etc)
    feature_data :
        the independent variable X
    target :
        the dependent variable y
    cv:
        the number of k fold cross validation (generally cv is set to 10 --> 10 fold cross validation)
    scoring:
        the metric used to evaluate the performance.
        When dealing with classification, the metric could be classification accuracy, precision, ROC/AUC etc.
        The score is calculated using the function cross_val_score (imported from sklearn.cross_validation),
        the scoring value for classification accuracy = 'accuracy',
        the scoring value for ROC/AUC = 'roc_auc'
    """

    def __init__(self, estimator, feature_data, target, cv, scoring):
        self.cv = cv
        self.target = target
        self.scoring = scoring
        self.estimator = estimator
        self.feature_data = feature_data

    def compute_scores(self):
        scores = cross_val_score(self.estimator,
                                 self.feature_data,
                                 self.target,
                                 cv=self.cv,
                                 scoring=self.scoring)
        return scores
