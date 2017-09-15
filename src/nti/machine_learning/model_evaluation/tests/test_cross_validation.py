#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import is_
from hamcrest import assert_that

from sklearn.neighbors import KNeighborsClassifier

from nti.testing.matchers import validly_provides

from nti.machine_learning.tests import ModelEvalutionLayerTest

from nti.machine_learning.model_evaluation.cross_validation import KFoldCrossValidation


class TestKFoldCrossValidation(ModelEvalutionLayerTest):
    def test_basic_10fold_cross_validation(self):
    	knn = KNeighborsClassifier(n_neighbors=5)
    	ten_cv = KFoldCrossValidation(knn, self.feature_data, self.target, 10,'accuracy')
    	scores = ten_cv.compute_scores()
    	assert_that(scores[0], 1)