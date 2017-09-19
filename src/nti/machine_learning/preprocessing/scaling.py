#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from zope import interface

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing.data import QuantileTransformer

from nti.machine_learning.preprocessing.interfaces import IScaler


def standard_scaler(X_train, **kwargs):
    return StandardScaler(**kwargs).fit(X_train)


def min_max_scaler(X_train, **kwargs):
    return MinMaxScaler(**kwargs).fit(X_train)


def max_abs_scaler(X_train, **kwargs):
    return MaxAbsScaler(**kwargs).fit(X_train)


def robust_scaler(X_train, **kwargs):
    return RobustScaler(**kwargs).fit(X_train)


def normalizer(X_train, **kwargs):
    return Normalizer(**kwargs).fit(X_train)


def quantile_transformer(X_train, **kwargs):
    return QuantileTransformer(**kwargs).fit(X_train)


@interface.implementer(IScaler)
class Scaler(object):

    def __init__(self, X_train, scaler_type, **kwargs):
        self.X_train = X_train
        self.scaler_type = scaler_type

        SCALERS = {
            'standard': standard_scaler,
            'min_max': min_max_scaler,
            'max_abs': max_abs_scaler,
            'robust': robust_scaler,
            'normal': normalizer,
            'quantile': quantile_transformer
        }

        if scaler_type in SCALERS:
            self.scaler = SCALERS[self.scaler_type](X_train, **kwargs)
        else:
            self.scaler = None
            logger.warning("Unrecoqnized scaler type")

    def transform(self, inputs):
        return self.scaler.transform(inputs)
