#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import assert_that

from nti.testing.matchers import validly_provides

from nti.machine_learning.algorithms.supervised.interfaces import ISVM

from nti.machine_learning.algorithms.supervised.support_vector_machine import SupportVectorMachine

from nti.machine_learning.tests import SupervisedLearningLayerTest


class TestSupervisedModels(SupervisedLearningLayerTest):

    def test_svm(self):
        svm = SupportVectorMachine(self.example_frame,
                                   self.example_prediction_columns)
        assert_that(svm, validly_provides(ISVM))
        # train
        svm.train()
        assert_that(svm.success_rate, 1.0)
