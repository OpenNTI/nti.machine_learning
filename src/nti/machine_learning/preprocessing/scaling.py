#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from zope import interface

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

from nti.machine_learning.preprocessing.interfaces import IScaler

def _standard_scaler(X_train, **kwargs):
    return StandardScaler(**kwargs).fit(X_train)

def _min_max_scaler(X_train, **kwargs):
    return MinMaxScaler(**kwargs).fit(X_train)

def _max_abs_scaler(X_train, **kwargs):
    return MinMaxScaler(**kwargs).fit(X_train)

def _robust_scaler(X_train, **kwargs):
    return RobustScaler(**kwargs).fit(X_train)

def _normalizer(X_train, **kwargs):
    return Normalizer(**kwargs).fit(X_train)

def _quantile_transformer(X_train, **kwargs):
    return QuantileTransformer(**kwargs).fit(X_train)


@interface.implementer(IScaler)
class Scaler(object):
    def __init__(X_train, scaler_type='standard', **kwargs):
        self.X_train = X_train
        self.scaler_type = scaler_type

        SCALERS : {'standard' : _standard_scaler,
                   'min_max'  : _min_max_scaler,
                   'max_abs'  : _max_abs_scaler,
                   'robust'   : _robust_scaler,
                   'normal'   : _normalizer,
                   'quantile' : _quantile_transformer}

        self.scaler = SCALERS[self.scaler_type](X_train, **kwargs)


    def transfrom(self, inputs):
        return self.scaler.transfrom(inputs)
