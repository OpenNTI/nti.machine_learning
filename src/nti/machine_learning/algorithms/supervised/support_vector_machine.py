#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import ISVM
from nti.machine_learning.algorithms.supervised.interfaces import ILinearSupportVectorClassification

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation

logger = __import__('logging').getLogger(__name__)


@interface.implementer(ISVM)
class SupportVectorMachine(SupervisedModel):
    """
    Abstraction of the SciKit Learn Support Vector Machine.
    """

    def classify(self, inputs):
        return self.classifier.predict([inputs])

    def train(self, data_frame, prediction_columns, metric='accuracy', k=10, **kwargs):
        super(SupportVectorMachine, self).train(data_frame, prediction_columns)
        self.clf = SVC(**kwargs)
        kf = KFoldCrossValidation(self.clf, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        scores = kf.compute_scores()
        self.success_rate = scores.mean()


@interface.implementer(ILinearSupportVectorClassification)
class LinearSupportVectorClassification(SupervisedModel):

    def classify(self, inputs):
        return self.cls.predict(inputs)

    def train(self, data_frame, prediction_column, metric='accuracy', k=10, **kwargs):
        super(LinearSupportVectorClassification, self).train(data_frame, prediction_column)
        self.cls = LinearSVC(**kwargs)
        kf = KFoldCrossValidation(self.cls, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.success_rate = kf.compute_scores().mean()
