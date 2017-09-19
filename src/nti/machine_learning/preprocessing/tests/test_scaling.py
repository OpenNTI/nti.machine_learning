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

from nti.machine_learning.preprocessing.scaling import Scaler

from nti.machine_learning.preprocessing.interfaces import IScaler

from nti.machine_learning.preprocessing.tests import ScalerLayerTest

class TestScaler(ScalerLayerTest):
	def test_standar_scaler(self):
		scaler = Scaler(self.X_train, 'standard')
		assert_that(scaler, validly_provides(IScaler))
		assert_that(scaler.scaler_type, is_('standard'))

		X_transform = scaler.transform(self.X_train)
		assert_that(len(X_transform), is_(3))
		assert_that(self.X_train[0][0], is_(1.0))
		assert_that(X_transform[0][0], is_(0.0))

	def test_min_max_scaler(self):
		scaler = Scaler(self.X_train, 'min_max')
		assert_that(scaler, validly_provides(IScaler))
		assert_that(scaler.scaler_type, is_('min_max'))

		X_transform = scaler.transform(self.X_train)
		assert_that(len(X_transform), is_(3))
		assert_that(self.X_train[0][0], is_(1.0))
		assert_that(X_transform[0][0], is_(0.5))

	def test_max_abs_scaler(self):
		scaler = Scaler(self.X_train, 'max_abs')
		assert_that(scaler, validly_provides(IScaler))
		assert_that(scaler.scaler_type, is_('max_abs'))

		X_transform = scaler.transform(self.X_train)
		assert_that(len(X_transform), is_(3))
		assert_that(self.X_train[0][0], is_(1.0))
		assert_that(X_transform[0][1], is_(-1.0))

	def test_robust_scaler(self):
		scaler = Scaler(self.X_train, 'robust')
		assert_that(scaler, validly_provides(IScaler))
		assert_that(scaler.scaler_type, is_('robust'))

		X_transform = scaler.transform(self.X_train)
		assert_that(len(X_transform), is_(3))
		assert_that(self.X_train[0][0], is_(1.0))
		assert_that(X_transform[1][1], is_(0.0))

	def test_normalizer(self):
		scaler = Scaler(self.X_train, 'normal')
		assert_that(scaler, validly_provides(IScaler))
		assert_that(scaler.scaler_type, is_('normal'))

		X_transform = scaler.transform(self.X_train)
		assert_that(len(X_transform), is_(3))
		assert_that(self.X_train[0][0], is_(1.0))
		assert_that(X_transform[2][0], is_(0.0))

	def test_quantile_transformer(self):
		scaler = Scaler(self.X_train, 'quantile')
		assert_that(scaler, validly_provides(IScaler))
		assert_that(scaler.scaler_type, is_('quantile'))

		X_transform = scaler.transform(self.X_train)
		assert_that(len(X_transform), is_(3))
		assert_that(self.X_train[0][0], is_(1.0))
		assert_that(X_transform[0][0], less_than(1.0))

