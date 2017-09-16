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

    def classify(self, inputs):
        return self.cls.predict(inputs)

    def train(self, data_frame, prediction_column, metric='accuracy', k=10, **kwargs):
        super(KNearestNeighborsClassifier, self).train(data_frame, prediction_column)
        self.cls = KNeighborsClassifier(**kwargs)
        kf = KFoldCrossValidation(self.cls, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.success_rate = kf.compute_scores().mean()

@interface.implementer(IKNearestNeighborsRegressor)
class KNearestNeighborsRegressor(SupervisedModel):

    def classify(self, inputs):
        return self.cls.predict(inputs)

    def train(self, data_frame, prediction_column, metric='neg_mean_squared_error', k=10, **kwargs):
        super(KNearestNeighborsClassifier, self).train(data_frame, prediction_column)
        self.cls = KNeighborsRegressor(**kwargs)
        kf = KFoldCrossValidation(self.cls, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.success_rate = kf.compute_scores().mean()
