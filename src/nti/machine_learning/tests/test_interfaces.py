#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from nti.machine_learning.algorithms.supervised import ISVM
from nti.machine_learning.algorithms.supervised import INeuralNetwork

from nti.machine_learning.tests import MachineLearningLayerTest

class TestRegistration(MachineLearningLayerTest):

    def test_svm(self):
        pass
