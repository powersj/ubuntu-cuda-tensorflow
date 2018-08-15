#!/usr/bin/env python3
"""Run matrix multiplication with Tensorflow on a GPU and CPU."""

import sys
import time

from tabulate import tabulate
import tensorflow as tf


DEVICES = [
    '/gpu:0',
    '/cpu:0'
]

MATRIX_SHAPES = [
    (100, 100),    
    (1000, 1000),
    (5000, 5000)
]


def create_matrix(matrix_shape):
    """Create a matrix of specified shape."""
    return tf.random_uniform(
        shape=matrix_shape,
        minval=0,
        maxval=1,
        dtype=tf.float16
    )


def time_tf_matrixmul(device, matrix_shape):
    """Run tensorflow with device and matrix size."""
    with tf.device(device):
        matrix_a = create_matrix(matrix_shape)
        matrix_b = create_matrix(matrix_shape)
        matmul = tf.matmul(matrix_a, matrix_b)

    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    start = time.time()
    session.run(matmul)
    end = time.time()

    return round(end - start, 2)


def bench():
    """Run CPU and GPU times."""
    results = []
    for device in DEVICES:
        for shape in MATRIX_SHAPES:
            print('%s with %s' % (device, shape))
            time = time_tf_matrixmul(device, shape)
            results.append([device, shape, time])

    print(tabulate(results, headers=['Device', 'Shape', 'Time (s)']))


if __name__ == '__main__':
    sys.exit(bench())
