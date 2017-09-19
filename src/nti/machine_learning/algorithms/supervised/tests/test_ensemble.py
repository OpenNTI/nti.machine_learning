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

from nti.machine_learning.algorithms.supervised.interfaces import IEnsembleRandomForestClassifier

from nti.machine_learning.tests import BinaryClassifierLayerTest
from nti.machine_learning.tests import MultiClassClassifierLayerTest


class TestEnsembleRandomForestClassifier(BinaryClassifierLayerTest):
    def test_basic_random_forest_classifier(self):
        random_forest_classifier = component.getUtility(IEnsembleRandomForestClassifier)
        assert_that(random_forest_classifier,
                    validly_provides(IEnsembleRandomForestClassifier))

        random_forest_classifier.train(self.data_frame, self.prediction_column)
        assert_that(random_forest_classifier.success_rate, greater_than(0))


class TestMultiClassEnsembleRandomForestClassifier(MultiClassClassifierLayerTest):
    def test_basic_multi_class_random_forest_classifier(self):
        random_forest_classifier = component.getUtility(IEnsembleRandomForestClassifier)
        assert_that(random_forest_classifier,
                    validly_provides(IEnsembleRandomForestClassifier))

        random_forest_classifier.train(self.data_frame, self.prediction_column)
        assert_that(random_forest_classifier.success_rate, greater_than(0))
