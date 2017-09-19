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

from nti.machine_learning.algorithms.supervised.interfaces import ISVM
from nti.machine_learning.algorithms.supervised.interfaces import ILinearSupportVectorClassification

from nti.machine_learning.tests import BinaryClassifierLayerTest
from nti.machine_learning.tests import MultiClassClassifierLayerTest


class TestLinearSupportVectorClassificationBinary(BinaryClassifierLayerTest):

    def test_basic_linear_svc(self):
        linear_svc = component.getUtility(ILinearSupportVectorClassification)
        assert_that(linear_svc,
                    validly_provides(ILinearSupportVectorClassification))

        linear_svc.train(self.data_frame, self.prediction_column)
        assert_that(linear_svc.success_rate, greater_than(0))


class TestLinearSupportVectorClassificationMultiClass(MultiClassClassifierLayerTest):

    def test_basic_multiclass_linear_svc(self):
        linear_svc = component.getUtility(ILinearSupportVectorClassification)
        assert_that(linear_svc,
                    validly_provides(ILinearSupportVectorClassification))

        linear_svc.train(self.data_frame, self.prediction_column)
        assert_that(linear_svc.success_rate, greater_than(0))


class TestSVM(BinaryClassifierLayerTest):
    def test_basic_svm(self):
        svm = component.getUtility(ISVM)
        assert_that(svm,
                    validly_provides(ISVM))
        svm.train(self.data_frame, self.prediction_column)
        assert_that(svm.success_rate, greater_than(0))


class TestMultiClassSVM(MultiClassClassifierLayerTest):
    def test_basic_multiclass_svm(self):
        svm = component.getUtility(ISVM)
        assert_that(svm,
                    validly_provides(ISVM))
        svm.train(self.data_frame, self.prediction_column)
        assert_that(svm.success_rate, greater_than(0))
