#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.cross_validation import cross_val_score

class KFoldCrossValidation(object):
	 def __init__(self, estimator, feature_data, target, cv, scoring):
	 	self.estimator = estimator
	 	self.feature_data = feature_data
	 	self.target = target
	 	self.cv = cv
	 	self.scoring =scoring
	 
	 def compute_scores(self):
	 	scores = cross_val_score(self.estimator,
	 		                          self.feature_data,
	 		                          self.target,
	 		                          cv = self.cv,
	 		                          scoring = self.scoring)
	 	return scores




