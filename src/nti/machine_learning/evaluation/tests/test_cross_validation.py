#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import is_
from hamcrest import assert_that
from hamcrest import greater_than

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from nti.machine_learning.evaluation.cross_validation import KFoldCrossValidation

from nti.machine_learning.tests import ModelEvalutionLayerTest


class TestKFoldCrossValidation(ModelEvalutionLayerTest):

    def test_basic_10fold_cross_validation(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        ten_cv = KFoldCrossValidation(knn, self.feature_data,
                                      self.target, 10, 'accuracy')
        scores = ten_cv.compute_scores()
        assert_that(len(scores), is_(10))

    def test_10fold_cross_validation_model_selection_accuracy(self):
        knn = KNeighborsClassifier(n_neighbors=5)
        ten_cv_knn = KFoldCrossValidation(knn, self.feature_data,
                                          self.target, 10, 'accuracy')
        knn_scores = ten_cv_knn.compute_scores()

        logreg = LogisticRegression()
        ten_cv_log_reg = KFoldCrossValidation(logreg, self.feature_data,
                                              self.target, 10, 'accuracy')
        logreg_scores = ten_cv_log_reg.compute_scores()

        assert_that(len(knn_scores), is_(10))
        assert_that(len(logreg_scores), is_(10))
        assert_that(knn_scores.mean(), greater_than(logreg_scores.mean()))
