#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

import unittest

from random import randint

from zope.component.hooks import setHooks

from nti.machine_learning import NTIDataFrame

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

class MachineLearningLayerTest(unittest.TestCase):

    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        examples = []
        for i in range(100):
            x = randint(0,1)
            y = randint(0,1)
            xor = x ^ y
            examples.append([x, y, xor])
        self.example_frame = NTIDataFrame(examples, columns=['x', 'y', 'xor'])
