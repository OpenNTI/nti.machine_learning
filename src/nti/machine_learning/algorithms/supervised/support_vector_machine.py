#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.svm import SVC

from zope import interface

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import ISVM

from nti.machine_learning.model_evaluation.cross_validation import KFoldCrossValidation


@interface.implementer(ISVM)
class SupportVectorMachine(SupervisedModel):
    """
    Abstraction of the SciKit Learn Support Vector Machine.
    """

    def __init__(self, data_frame, prediction_column, **kwargs):
        super(SupportVectorMachine, self).__init__(data_frame,
                                                   prediction_column)
        self.clf = SVC(**kwargs)

    def classify(self, inputs):
        return self.clf.predict([inputs])

    def train(self):
        kf = KFoldCrossValidation(self.clf, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), 10, 'accuracy')
        scores = kf.compute_scores()
        self.success_rate = scores.mean()
