#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import assert_that
from hamcrest import greater_than

from zope import component

from nti.testing.matchers import validly_provides

from nti.machine_learning.algorithms.supervised.k_nearest_neighbors import KNearestNeighborsClassifier

from nti.machine_learning.algorithms.supervised.interfaces import IKNearestNeighborsClassifier

from nti.machine_learning.tests import BinaryClassifierLayerTest
from nti.machine_learning.tests import MultiClassClassifierLayerTest


class TestKNearestNeighborsBinaryClassifier(BinaryClassifierLayerTest):

    def test_basic_knn_classifier(self):
        knn_classifier = component.getUtility(IKNearestNeighborsClassifier)
        assert_that(knn_classifier,
                    validly_provides(IKNearestNeighborsClassifier))

        knn_classifier.train(self.data_frame, self.prediction_column)

        assert_that(knn_classifier.success_rate, greater_than(0))


class TestKNearestNeighborsClassifier(MultiClassClassifierLayerTest):
    def test_basic_knn_classifier(self):
        knn_classifier = component.getUtility(IKNearestNeighborsClassifier)

        assert_that(knn_classifier,
                    validly_provides(IKNearestNeighborsClassifier))

        knn_classifier.train(self.data_frame, self.prediction_column)

        assert_that(knn_classifier.success_rate, greater_than(0))