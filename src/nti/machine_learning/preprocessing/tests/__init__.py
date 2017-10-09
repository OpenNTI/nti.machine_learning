#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

import unittest

import numpy as np

from nti.machine_learning.tests import SharedConfiguringTestLayer


class ScalerLayerTest(unittest.TestCase):

    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        self.X_train = np.array([[1., -1., 2.],
                                 [2., 0., 0.],
                                 [0., 1., -1]])


class EncoderLayerTest(unittest.TestCase):

    layer = SharedConfiguringTestLayer

    @classmethod
    def setUp(self):
        self.original_array_numeric = [1, 2, 2, 6]
        self.original_array_nonnumeric = [
            "barcelona", "rome", "sf", "jakarta", "sf"
        ]
