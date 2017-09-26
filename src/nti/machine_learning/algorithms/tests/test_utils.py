#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import is_
from hamcrest import assert_that
from hamcrest import greater_than

import unittest

from nti.machine_learning.algorithms.utils import distance

from nti.machine_learning.tests import SharedConfiguringTestLayer


class TestUtilsCrossValidation(unittest.TestCase):

    layer = SharedConfiguringTestLayer
    
    def test_distance(self):
        assert_that(distance((2, -1),(-2, 2) ),
                    is_(5.0))