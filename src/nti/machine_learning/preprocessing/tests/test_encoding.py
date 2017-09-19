#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

# disable: accessing protected members, too many methods
# pylint: disable=W0212,R0904

from hamcrest import is_
from hamcrest import assert_that
from hamcrest import less_than

from nti.testing.matchers import validly_provides

from nti.machine_learning.preprocessing.encoding import Encoder

from nti.machine_learning.preprocessing.tests import EncoderLayerTest

class TestEncoder(EncoderLayerTest):
	def test_label_encoder_numeric(self):
		label_encoder = Encoder(self.original_array_numeric, 'label')

		assert_that(label_encoder.original_array[0], is_(1))
		assert_that(label_encoder.original_array[1], is_(2))
		assert_that(label_encoder.original_array[2], is_(2))
		assert_that(label_encoder.original_array[3], is_(6))

		transformed_array = label_encoder.transform(self.original_array_numeric)
		
		assert_that(transformed_array[0], is_(0))
		assert_that(transformed_array[1], is_(1))
		assert_that(transformed_array[2], is_(1))
		assert_that(transformed_array[3], is_(2))

	def test_label_encoder_numeric(self):
		label_encoder = Encoder(self.original_array_nonnumeric, 'label')
	
		assert_that(label_encoder.original_array[0], is_("barcelona"))
		assert_that(label_encoder.original_array[1], is_("rome"))
		assert_that(label_encoder.original_array[2], is_("sf"))
		assert_that(label_encoder.original_array[3], is_("jakarta"))
		assert_that(label_encoder.original_array[4], is_("sf"))

		transformed_array = label_encoder.transform(self.original_array_nonnumeric)
		
		assert_that(transformed_array[0], is_(0))
		assert_that(transformed_array[1], is_(2))
		assert_that(transformed_array[2], is_(3))
		assert_that(transformed_array[3], is_(1))
		assert_that(transformed_array[4], is_(3))



