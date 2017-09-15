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
