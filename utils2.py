import numba as nb
import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
from collections import defaultdict
import numpy as np
import torch
import math

def get_link(filename, shift_id=0):
    links = []
    for line in open(filename + 'sup_pairs', 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2) - shift_id))
    for line in open(filename + 'ref_pairs', 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2) - shift_id))
    return links