#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import is_
from hamcrest import close_to
from hamcrest import has_length
from hamcrest import assert_that

import unittest

from nti.machine_learning.algorithms.utils import alpha
from nti.machine_learning.algorithms.utils import entropy
from nti.machine_learning.algorithms.utils import distance
from nti.machine_learning.algorithms.utils import variance
from nti.machine_learning.algorithms.utils import similarity

from nti.machine_learning.tests import SharedConfiguringTestLayer


class TestUtilsCrossValidation(unittest.TestCase):

    layer = SharedConfiguringTestLayer

    def test_distance(self):
        assert_that(distance((2, -1), (-2, 2)),
                    is_(5.0))

    def test_alpha(self):
        assert_that(alpha(10),
                    is_(close_to(0.06931, 0.01)))

    def test_similarity(self):
        assert_that(similarity(10, (2, -1), (-2, 2)),
                    is_(close_to(0.7071, 0.01)))

    def test_entropy(self):
        result = entropy(((2, -1), (-2, 2)), 10)
        assert_that(result, has_length(2))
        assert_that(result[0], is_(close_to(0.87242, 0.01)))
        assert_that(result[1], is_(close_to(0.87242, 0.01)))

    def test_variance(self):
        assert_that(variance([600, 470, 170, 430, 300], 394),
                    is_(close_to(21704, 0.01)))
