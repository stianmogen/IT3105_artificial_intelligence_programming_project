import numpy


def normalize(x: numpy.array):
    return x / x.sum()
