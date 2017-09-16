#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

import unittest

import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

from random import randint

from zope.component.hooks import setHooks

from nti.testing.layers import GCLayerMixin
from nti.testing.layers import ZopeComponentLayer
from nti.testing.layers import ConfiguringLayerMixin

import zope.testing.cleanup


class SharedConfiguringTestLayer(ZopeComponentLayer,
                                 GCLayerMixin,
                                 ConfiguringLayerMixin):

    set_up_packages = ('nti.machine_learning',)

    @classmethod
    def setUp(cls):
        setHooks()
        cls.setUpPackages()

    @classmethod
    def tearDown(cls):
        cls.tearDownPackages()
        zope.testing.cleanup.cleanUp()

    @classmethod
    def testSetUp(cls, unused_test=None):
        setHooks()

    @classmethod
    def testTearDown(cls):
        pass


from nti.machine_learning import DataFrame


class SupervisedLearningLayerTest(unittest.TestCase):

    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        examples = []
        for _ in range(100):
            x = randint(0, 1)
            y = randint(0, 1)
            xor = x ^ y
            examples.append([x, y, xor])
        self.example_frame = DataFrame(examples, columns=['x', 'y', 'xor'])
        self.example_prediction_columns = ['xor']


class UnsupervisedLearningLayerTest(unittest.TestCase):

    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        points = []
        for _ in range(500):
            x = randint(0, 30)
            y = randint(0, 100)
            z = randint(0, 100)
            points.append([x, y, z])
        for _ in range(500):
            x = randint(70, 100)
            y = randint(0, 100)
            z = randint(0, 100)
            points.append([x, y, z])
        self.example_frame = DataFrame(points, columns=['x', 'y', 'z'])


class ModelEvaluationLayerTest(unittest.TestCase):
    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        iris = load_iris()
        self.feature_data = iris.data
        self.target = iris.target
        self.feature_names = iris.feature_names


class MultiClassClassifierEstimatorLayerTest(unittest.TestCase):
    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        iris = load_iris()
        self.data_frame = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
            columns = iris['feature_names'] + ['target'])
        self.prediction_column=['target']