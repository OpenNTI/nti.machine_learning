#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from numpy import dot
from numpy import sqrt

from sklearn.cross_validation import KFold

from sklearn.linear_model import LinearRegression

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import IRegressor

@interface.implementer(IRegressor)
class Regressor(SupervisedModel):

    def __init__(self, data_frame, prediction_columns, **kwargs):
        super(Regressor, self).__init__(data_frame,
                                        prediction_columns)
        self.clf = LinearRegression(**kwargs)

    def train(self):
        kf = KFold(self._data.total_size(), n_folds=10)
        err = 0
        x = self._data.get_frame_no_predictor()
        y = self._data.get_predictors()
        for train, test in kf:
            self.clf.fit(x[train], y[train])
            test_vals = self.classify(x[test])
            out = p - y[test]
            err += dot(out, out)
        self.rmse = err/self._data.total_size()

    def classify(self, inputs):
        return self.clf.predict(inputs)
