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

from nti.machine_learning.supervised.interfaces import ISVM

@interface.implementer(ISVM)
class SupportVectorMachine(SupervisedModel):
    """
    Abstraction of the SciKit Learn Support Vector Machine.
    """

    def __init__(self, data_frame, prediction_column, training_size=.7, **kwargs):
        super(SupportVectorMachine, self).__init__(data_frame,
                                                   prediction_column, 
                                                   training_set_ratio=training_size)
        self.svc = SVC(**kwargs)

    def classify(self, inputs):
        pred_ans = self.svc.predict([inputs])
        return pred_ans

    def train(self):
        self.svc.fit(self._training_set_inputs, self._training_set_outputs)
        self._run_validation()
