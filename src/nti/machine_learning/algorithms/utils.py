#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file provide mostly math operations for the algorithms.
If any of this gets actually used, it should be heavily optimized, ideally in C or C++.

.. $Id$
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from six.moves import xrange

from math import log
from math import exp
from math import sqrt

LOG_05 = log(0.5)

logger = __import__('logging').getLogger(__name__)


def distance(p1, p2):
    """
    Euclidean distance between two points
    """
    if not len(p1) == len(p2):
        raise ValueError('Points must have the same dimensions.')
    summation = sum((p2[i] - p1[i])**2 for i in xrange(len(p1)))
    return sqrt(summation)


def mean_distance(points):
    """
    Mean Euclidean distance between a set of points
    """
    result = 0
    n = len(points)
    nc2 = float(((n**2) - n) / 2)
    for i in xrange(0, n):
        for j in xrange(i + 1, n):
            result += distance(points[i], points[j])
    return (1 / nc2) * result


def alpha(mean_dist):
    """
    Constant alpha value in similarity equation
    """
    return -(LOG_05 / mean_dist)


def similarity(mean_dist, pointi, pointj):
    """
    Entropic similarity
    """
    power = alpha(mean_dist) * distance(pointi, pointj)
    return exp(-power)


def entropy(points, mean_dist):
    """
    Entropy over a set of points
    """
    entropies = []
    n = len(points)
    for i in xrange(0, n):
        summation = 0
        for j in xrange(0, n):
            if i == j:
                continue
            sim = similarity(mean_dist, points[i], points[j])
            if sim == 1.0:
                # A value of 1 will break, so make
                # it almost one
                sim = .999999
            summation += (sim * log(sim, 2)) + \
                         (1 - sim) * (log(1 - sim, 2))
        entropies.append(-summation)
    return entropies


def variance(values, mean):
    """
    Variance of a set of values.
    """
    N = len(values)
    sum_val = sum((v - mean)**2 for v in values)
    return sum_val / float(N)
