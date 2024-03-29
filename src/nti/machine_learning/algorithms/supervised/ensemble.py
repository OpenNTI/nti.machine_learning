#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from zope import interface

from sklearn.ensemble import RandomForestClassifier

from nti.machine_learning.algorithms.supervised import SupervisedModel

from nti.machine_learning.algorithms.supervised.interfaces import IEnsembleRandomForestClassifier

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation

logger = __import__('logging').getLogger(__name__)


@interface.implementer(IEnsembleRandomForestClassifier)
class EnsembleRandomForestClassifier(SupervisedModel):

    def classify(self, inputs):
        self.cls.predict(inputs)

    def train(self, data_frame, prediction_column, metric='accuracy', k=10, **kwargs):
        super(EnsembleRandomForestClassifier, self).train(data_frame, prediction_column)
        self.cls = RandomForestClassifier(**kwargs)
        kf = KFoldCrossValidation(self.cls, self._data.get_frame_no_predictor(),
                                  self._data.get_predictors(), k, metric)
        self.cls.fit(self._data.get_frame_no_predictor(), self._data.get_predictors())
        self.success_rate = kf.compute_scores().mean()
