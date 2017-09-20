#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. $Id$
"""

from __future__ import print_function, absolute_import, division
__docformat__ = "restructuredtext en"

from zope import interface


class IKFoldCrossValidation(interface.Interface):
    """
    K fold cross validation model
    """
