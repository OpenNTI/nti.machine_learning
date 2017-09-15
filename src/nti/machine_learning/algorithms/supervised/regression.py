#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from numpy import sqrt

from sklearn.linear_model import LinearRegression

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import IRegressor

from nti.machine_learning.model_evaluation.cross_validation import KFoldCrossValidation

@interface.implementer(IRegressor)
class Regressor(SupervisedModel):

    def __init__(self, data_frame, prediction_columns, **kwargs):
        super(Regressor, self).__init__(data_frame,
                                        prediction_columns)
        self.clf = LinearRegression(**kwargs)

    def train(self):
        kf = KFoldCrossValidation(self.clf, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), 10, 'neg_mean_squared_error')
        scores = kf.compute_scores()
        self.rmse = sqrt(abs(scores.mean()))

    def classify(self, inputs):
        return self.clf.predict(inputs)
