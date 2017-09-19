#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

logger = __import__('logging').getLogger(__name__)

from sklearn.preprocessing import LabelEncoder


def label_encoder(original_array, **kwargs):
    """
    Encode labels with value between 0 and n_classes-1.
    """
    return LabelEncoder(**kwargs).fit(original_array)


class Encoder(object):

    def __init__(self, original_array, encoder_type, **kwargs):
        self.original_array = original_array

        ENCODERS = {'label': label_encoder}

        if encoder_type in ENCODERS:
            self.encoder = ENCODERS[encoder_type](original_array, **kwargs)
        else:
            self.encoder = None
            logger.warning("Unrecognized encoder type")

    def transform(self, original_array):
        if self.encoder:
            return self.encoder.transform(original_array)
