#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import assert_that
from hamcrest import greater_than
from hamcrest import has_property

from zope import component

from nti.testing.matchers import validly_provides

from nti.machine_learning.algorithms.supervised.interfaces import ISVM
from nti.machine_learning.algorithms.supervised.interfaces import IRegressor
from nti.machine_learning.algorithms.supervised.interfaces import INeuralNetwork
from nti.machine_learning.algorithms.supervised.interfaces import IKNearestNeighborsRegressor
from nti.machine_learning.algorithms.supervised.interfaces import IKNearestNeighborsClassifier


from nti.machine_learning.tests import SupervisedLearningLayerTest


class TestSupervisedModels(SupervisedLearningLayerTest):
    """
    Test the validity of the various supervised models.
    """

    def test_svm(self):
        svm = component.getUtility(ISVM)
        assert_that(svm, validly_provides(ISVM))
        # train
        svm.train(self.example_frame, self.example_prediction_columns)
        assert_that(has_property(svm, 'success_rate'))
        assert_that(svm.success_rate, 1.0)

    def test_regression(self):
        reg = component.getUtility(IRegressor)
        assert_that(reg, validly_provides(IRegressor))
        # train
        reg.train(self.example_frame, self.example_prediction_columns)
        assert_that(has_property(reg, 'rmse'))
        assert_that(reg.rmse, greater_than(0.0))

    def test_neural_network(self):
        nn = component.getUtility(INeuralNetwork)
        assert_that(nn, validly_provides(INeuralNetwork))
        # train
        nn.train(self.example_frame, self.example_prediction_columns,
                 layers=(3,), max_iter=500, solver='sgd')
        assert_that(nn, has_property('success_rate'))
        assert_that(nn.success_rate, greater_than(0.0))

    def test_k_nearest_neighbors_regressor(self):
        knn_regressor = component.getUtility(IKNearestNeighborsRegressor)
        assert_that(knn_regressor, 
                    validly_provides(IKNearestNeighborsRegressor))
        # train
        knn_regressor.train(self.example_frame,
                            self.example_prediction_columns)
        assert_that(has_property(knn_regressor, 'success_rate'))

    def test_k_nearest_neighbors_classifier(self):
        knn_classifier = component.getUtility(IKNearestNeighborsClassifier)
        assert_that(knn_classifier, 
                    validly_provides(IKNearestNeighborsClassifier))
        # train
        knn_classifier.train(self.example_frame,
                             self.example_prediction_columns)
        assert_that(knn_classifier, has_property('success_rate'))
