# -*- coding: utf-8 -*-
import numpy as np


def calc_eucl_dist(pos1, pos2):
    # calc Euclidean Distance
    _coords1 = np.array(pos1)  # location of me
    _coords2 = np.array(pos2)
    # alternative way: np.linalg.norm(_coords1 - _coords2)
    eucl_dist = np.sqrt(np.sum((_coords1 - _coords2) ** 2))
    return eucl_dist
