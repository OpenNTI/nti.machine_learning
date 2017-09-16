#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from zope import interface

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import IKNearestNeighborsRegressor
from nti.machine_learning.algorithms.supervised.interfaces import IKNearestNeighborsClassifier

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation


@interface.implementer(IKNearestNeighborsClassifier)
class KNearestNeighborsClassifier(SupervisedModel):

    def __init__(self, data_frame, prediction_column, **kwargs):
        super(KNearestNeighborsClassifier, self).__init__(data_frame, prediction_column)
        self.classifier = KNeighborsClassifier(**kwargs)

    def train(self, metric='accuracy', k=10):
        kf = KFoldCrossValidation(self.classifier, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.scores = kf.compute_scores()


@interface.implementer(IKNearestNeighborsRegressor)
class KNearestNeighborsRegressor(SupervisedModel):

    def __init__(self, data_frame, prediction_column, **kwargs):
        super(KNearestNeighborsClassifier, self).__init__(data_frame, prediction_column)
        self.classifier = KNeighborsRegressor(**kwargs)

    def train(self, metric='neg_mean_squared_error', k=10):
        kf = KFoldCrossValidation(self.classifier, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.scores = kf.compute_scores()
