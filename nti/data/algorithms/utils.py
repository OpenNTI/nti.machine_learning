"""
This file provide mostly math operations for the algorithms.
If any of this gets actually used, it should be heavily optimized, ideally in C or C++.
"""

from math import sqrt
from math import log
from math import exp


def distance(p1, p2):
    """
    Euclidean distance between two points
    """
    if not p1.dimensions == p2.dimensions:
        raise ValueError('Points must have the same dimensions.')
    summation = 0
    for i in range(0, p1.dimensions):
        summation += (p2.get(i) - p1.get(i))**2
    return sqrt(summation)


def mean_distance(points):
    """
    Mean Euclidean distance between a set of points
    """
    result = 0
    n = len(points)
    nc2 = float(((n**2) - n) / 2)
    for i in range(0, n):
        for j in range(i + 1, n):
            result += distance(points[i], points[j])
    return (1 / nc2) * result


def alpha(mean_dist):
    """
    Constant alpha value in similarity equation
    """
    return -(log(0.5) / mean_dist)


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
    for i in range(0, n):
        summation = 0
        for j in range(0, n):
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
    sum_val = sum([(v - mean)**2 for v in values])
    return sum_val / float(N)
